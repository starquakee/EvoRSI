"""Read-only trusted-validation collection; never loads a final answer file."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from research.calibration.gates import read_json
from research.calibration.specs import CalibrationError, RunSpec, protocol_identity, sha256_file
from research.comparison import deploy
from research.comparison.protocol import ModelProfile
from research.comparison.runner import comparison_config_identity
from research.comparison.transport import comparison_request_profile, _profile_value_matches
from research.evaluator.service import EvaluatorRegistry
from research.search.sandbox_task import _single_file_tree_sha256


def json_lines(path):
    if path.is_symlink() or not path.is_file():
        raise CalibrationError('trace_missing_or_symlink')
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def classification_rows(path):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['id', 'label']:
            raise CalibrationError('prediction_columns_invalid')
        rows = {}
        for row in reader:
            row_id = row.get('id')
            if not isinstance(row_id, str) or not row_id or row_id in rows or set(row) != {'id', 'label'}:
                raise CalibrationError('prediction_ids_invalid')
            try:
                label = float(row['label'])
            except (ValueError, TypeError) as exc:
                raise CalibrationError('prediction_label_invalid') from exc
            if not math.isfinite(label) or label not in (0, 1):
                raise CalibrationError('prediction_label_invalid')
            rows[row_id] = int(label)
    if not rows:
        raise CalibrationError('prediction_rows_empty')
    return rows


def recompute_accuracy(prediction, answers):
    predicted, truth = classification_rows(prediction), classification_rows(answers)
    if set(predicted) != set(truth):
        raise CalibrationError('prediction_id_set_mismatch')
    correct = sum(predicted[key] == label for key, label in truth.items())
    return {'correct': correct, 'rows': len(truth), 'exact_accuracy': correct / len(truth),
            'trusted_precision_accuracy': round(correct / len(truth), 6)}


def mechanisms(trace, nodes, decoded):
    by_id = {node['id']: node for node in nodes}
    if len(by_id) != len(nodes):
        raise CalibrationError('journal_node_identity_duplicate')
    counts = Counter(event['operator'] for event in trace)
    selections, draft_scores = [], []
    def softmax(values, temperature):
        if not math.isfinite(temperature) or temperature <= 0:
            raise CalibrationError('selection_temperature_invalid')
        peak = max(values)
        exp = [math.exp((value - peak) / temperature) for value in values]
        return [value / sum(exp) for value in exp]
    for event in trace:
        config = event['config']
        weights = config['experience']['parent_selection']['weights']
        if config['crossover_prob'] != decoded['crossover_prob'] or config['max_debug_depth'] != 1 or weights != {key: decoded[key + '_weight'] for key in ('score', 'delta', 'novelty')}:
            raise CalibrationError('operator_parameter_mismatch')
        child = by_id[event['child_node_id']]
        if hashlib.sha256(child['code'].encode('utf-8', errors='surrogatepass')).hexdigest() != event['child_code_sha256']:
            raise CalibrationError('operator_child_source_mismatch')
        if len(event['parent_node_ids']) != len(event['parent_code_sha256']):
            raise CalibrationError('operator_parent_binding_invalid')
        for parent_id, digest in zip(event['parent_node_ids'], event['parent_code_sha256']):
            if hashlib.sha256(by_id[parent_id]['code'].encode('utf-8', errors='surrogatepass')).hexdigest() != digest:
                raise CalibrationError('operator_parent_source_mismatch')
        if event['operator'] == 'draft' and not child['is_buggy'] and isinstance(child.get('metric'), (float, int)):
            draft_scores.append(child['metric'])
        if event['operator'] not in ('improve', 'crossover'):
            continue
        selection = event.get('selection')
        if not selection or not selection.get('enabled') or selection['weights'] != weights or selection['selected_node_ids'] != event['parent_node_ids']:
            raise CalibrationError('parent_selection_binding_invalid')
        candidates = selection['candidates']
        utilities = [sum(weights[key] * candidate[key + '_component'] for key in weights) - candidate['official_score_missing_penalty'] for candidate in candidates]
        probabilities = softmax(utilities, selection['temperature'])
        for candidate, utility, probability in zip(candidates, utilities, probabilities):
            if any(not math.isfinite(value) for value in (candidate['utility'], utility, candidate['probability'], probability)) or abs(candidate['utility'] - utility) > 1e-10 or abs(candidate['probability'] - probability) > 1e-10 or candidate['fitness'] != by_id[candidate['node_id']]['metric'] or candidate.get('official_score') != candidate['fitness'] or candidate.get('official_status') != 'completed':
                raise CalibrationError('parent_selection_recompute_mismatch')
        changed = [softmax([utility + (weight - weights['score']) * candidate['score_component']
                           for utility, candidate in zip(utilities, candidates)], selection['temperature'])
                   for weight in (0.0, 2.0)]
        spread = max(candidate['score_component'] for candidate in candidates) - min(candidate['score_component'] for candidate in candidates)
        selections.append({'generation': event['generation_id'], 'operator': event['operator'],
            'score_component_informative': spread > 1e-12,
            'score_weight_first_draw_tv_0_to_2': 0.5 * sum(abs(a - b) for a, b in zip(*changed)),
            'entire_parent_set_forced': len(candidates) == len(selection['selected_node_ids'])})
    return {'operator_counts': {key: counts[key] for key in ('draft', 'improve', 'crossover', 'debug')},
            'initial_draft_best': max(draft_scores) if draft_scores else None,
            'parent_selections': selections}


def validate_trial(repo_root, manifest, trial, snapshot):
    spec = RunSpec(trial, manifest['identity'])
    directory = repo_root / spec.run_dir
    result = read_json(directory / 'run-result.json')
    expected = {'schema': 'calibration-inner-result.v1', 'manifest_identity': manifest['identity'],
        'trial_id': trial.trial_id, 'run_key': trial.run_key(), 'cell_key': trial.cell_key(),
        'run_identity': trial.identity(), 'cell_identity': trial.identity(), 'decoded': trial.decoded,
        'inner_seed': trial.inner_seed, 'config_id': trial.record['config_id'], 'repeat': trial.record['repeat'],
        'runner_commit': trial.source_commit, 'protocol_identity': protocol_identity(),
        'config_identity': comparison_config_identity(spec), 'model_profile_identity': ModelProfile().identity(),
        'dataset': manifest['dataset'], 'policy_version': manifest['policy_version'],
        'policy_sha256': manifest['policy_sha256'], 'metric': 'accuracy', 'metric_direction': 'maximize',
        'worker_cleanup_verified': True}
    if any(result.get(key) != value for key, value in expected.items()):
        raise CalibrationError('trial_result_identity_or_cleanup_mismatch')
    for name, key in [('operator-trace.jsonl', 'operator_trace_sha256'), ('request-trace.jsonl', 'request_trace_sha256'),
                      ('journal.jsonl', 'journal_sha256'), ('generation-outcomes.jsonl', 'generation_outcomes_sha256')]:
        if result.get(key) == '' and not (directory / name).exists():
            continue
        if (directory / name).is_symlink() or sha256_file(directory / name) != result.get(key):
            raise CalibrationError('trial_artifact_hash_mismatch')
    live = read_json(directory / 'live-jobs.json')
    if live.get('schema') != 'calibration-live-jobs.v1' or live.get('run_key') != trial.run_key() or any(live.get(key) != [] for key in ('live_jobs', 'uncertain_submissions', 'submission_intents')):
        raise CalibrationError('trial_live_jobs_unresolved')
    requests = [record for record in snapshot['requests'] if record['run_identity'] == trial.identity()]
    costs = {'main': sum(record['kind'] == 'main' for record in requests),
             'debug': sum(record['kind'] == 'debug' for record in requests),
             'charged_tokens': sum(record['charged_tokens'] for record in requests)}
    if result.get('requests') != costs:
        raise CalibrationError('trial_request_accounting_mismatch')
    trace_requests = json_lines(directory / 'request-trace.jsonl') if requests else []
    if [record.get('request_id') for record in trace_requests] != [record['request_id'] for record in requests]:
        raise CalibrationError('trial_request_trace_mismatch')
    profile = comparison_request_profile()
    for attempt, record in zip(trace_requests, requests):
        if attempt.get('kind') != record['kind'] or attempt.get('run_key') != trial.run_key() or attempt.get('reserved_tokens') != record['reserved_tokens']:
            raise CalibrationError('trial_request_trace_mismatch')
        for key in set(profile) - {'base_url'}:  # base_url deliberately excluded from old shared trace allowlist
            if not _profile_value_matches(profile[key], attempt['request'].get(key)):
                raise CalibrationError('trial_request_profile_mismatch')
    nodes = json_lines(directory / 'journal.jsonl') if result.get('journal_sha256') else []
    trace = json_lines(directory / 'operator-trace.jsonl') if result.get('operator_trace_sha256') else []
    behavioral = mechanisms(trace, nodes, trial.decoded)
    if behavioral['operator_counts'] != result.get('operator_counts'):
        raise CalibrationError('trial_operator_counts_mismatch')
    code_by_hash = {hashlib.sha256(node['code'].encode('utf-8', errors='surrogatepass')).hexdigest(): node['code'].encode('utf-8', errors='surrogatepass')
                    for node in nodes if isinstance(node.get('code'), str) and node['code']}
    registry = EvaluatorRegistry.load(repo_root / 'research/evaluator/registry.v1.json')
    evaluator = registry.task('comparison_spambase')
    if sha256_file(evaluator.answer_path) != manifest['dataset']['validation_labels_sha256'] or evaluator.split != 'validation':
        raise CalibrationError('validation_answer_binding_mismatch')
    rescored, artifacts = {}, result['prediction_artifacts']
    job_ids = [job['job_id'] for job in result['jobs'] if job.get('job_id')]
    if len(job_ids) != len(set(job_ids)) or result.get('job_ids') != sorted(job_ids):
        raise CalibrationError('trial_job_id_binding_invalid')
    if (directory / 'predictions').is_symlink():
        raise CalibrationError('prediction_directory_symlink')
    for job in result['jobs']:
        if job.get('source_identity_verified') is False or job.get('worker_cleanup_verified') is False or job.get('evidence_mismatch') is not None:
            raise CalibrationError('job_execution_integrity_contradiction')
        if job.get('evidence_verified') is not True:
            if job.get('score') is not None:
                raise CalibrationError('unverified_job_score')
            continue
        job_id, evaluation, gate = job['job_id'], job['evaluation'], job['source_gate']
        deploy.validate_job_snapshot(evaluation.get('prediction_snapshot'), job_id)
        if any(evaluation.get(key) != value for key, value in {
            'evaluator': 'external_trusted', 'task_id': 'comparison_spambase', 'split': 'validation',
            'metric': 'accuracy', 'direction': 'maximize',
            'answer_sha256': manifest['dataset']['validation_labels_sha256'],
            'registry_version': manifest['dataset']['registry_version'], 'score': job['score'],
        }.items()):
            raise CalibrationError('job_evaluation_binding_mismatch')
        prediction = directory / 'predictions' / (job_id + '.csv')
        backend_prediction = deploy.job_snapshot_host_path(evaluation['prediction_snapshot'], job_id,
            storage_root=repo_root / deploy.DISPATCHER_STORAGE_ROOT)
        if prediction.is_symlink() or sha256_file(prediction) != evaluation['prediction_sha256'] or sha256_file(backend_prediction) != evaluation['prediction_sha256'] or artifacts.get(job_id) != evaluation['prediction_sha256']:
            raise CalibrationError('job_prediction_hash_mismatch')
        source = code_by_hash.get(job.get('submitted_source_sha256'))
        if source is None or any(gate.get(key) != value for key, value in {
            'allowed': True, 'entrypoint': 'main.py', 'entrypoint_sha256': hashlib.sha256(source).hexdigest(),
            'source_sha256': _single_file_tree_sha256('main.py', source),
            'policy_version': manifest['policy_version'], 'policy_sha256': manifest['policy_sha256'],
        }.items()):
            raise CalibrationError('job_source_bytes_mismatch')
        accuracy = recompute_accuracy(prediction, evaluator.answer_path)
        if evaluation['row_count'] != accuracy['rows'] or abs(accuracy['trusted_precision_accuracy'] - job['score']) > 1e-9:
            raise CalibrationError('independent_validation_recompute_mismatch')
        rescored[job_id] = accuracy
    incumbent = result.get('incumbent')
    value = None
    if incumbent is not None:
        matching = [node for node in nodes if node['id'] == incumbent['node_id']]
        if len(matching) != 1 or incumbent['job_id'] not in rescored or hashlib.sha256(matching[0]['code'].encode('utf-8', errors='surrogatepass')).hexdigest() != incumbent['code_sha256']:
            raise CalibrationError('incumbent_source_or_job_unproven')
        value = incumbent['validation_score']
        valid_scores = [node['metric'] for node in nodes if not node['is_buggy'] and isinstance(node.get('code'), str) and node['code'] and type(node.get('metric')) in (int, float)]
        if not valid_scores or value != max(valid_scores) or value != rescored[incumbent['job_id']]['trusted_precision_accuracy'] or artifacts.get(incumbent['job_id']) != incumbent['prediction_sha256']:
            raise CalibrationError('incumbent_score_or_prediction_mismatch')
    completed = result['status'] == 'completed'
    if completed and (costs['main'] != 6 or result['main_candidates'] != 6 or result['generations_completed'] != 3 or any(record['status'] != 'known' for record in requests)):
        raise CalibrationError('completed_trial_shape_invalid')
    status = 'completed' if completed and value is not None else 'terminated' if result['status'] == 'cell_terminated' else 'failed'
    return {'normalized': {'trial_id': trial.trial_id, 'status': status, 'validation_accuracy': value,
                'requests': len(requests), 'charged_tokens': costs['charged_tokens'], 'reason': result['termination']},
            'behavior': behavioral, 'rescored_jobs': rescored, 'job_counts': result['job_counts'],
            'result_sha256': sha256_file(directory / 'run-result.json')}
