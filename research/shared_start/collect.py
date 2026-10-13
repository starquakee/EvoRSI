"""Read-only trusted-validation collection; never loads a final answer file."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from research.shared_start.gates import read_json
from research.shared_start.specs import CalibrationError, RunSpec, protocol_identity, sha256_file, trials
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
        if (not candidates or len({c['node_id'] for c in candidates}) != len(candidates)
                or not set(selection['selected_node_ids']) <= {c['node_id'] for c in candidates}
                or len(set(selection['selected_node_ids'])) != len(selection['selected_node_ids'])):
            raise CalibrationError('parent_selection_candidates_invalid')
        scores = [by_id[c['node_id']]['metric'] for c in candidates]
        if any(type(score) not in (int, float) or not math.isfinite(score) for score in scores):
            raise CalibrationError('parent_selection_score_invalid')
        low, high = min(scores), max(scores)
        by_step = {n['step']: n for n in nodes}
        deltas = []
        for candidate, score in zip(candidates, scores):
            parent_scores = [by_step[s]['metric'] for s in by_id[candidate['node_id']]['parents']
                             if type(by_step[s].get('metric')) in (int, float)]
            deltas.append(max(0.0, score - max(parent_scores)) if parent_scores else 0.0)
        max_delta = max(deltas)
        for candidate, score, delta in zip(candidates, scores, deltas):
            if (candidate.get('component_normalization') != {'score': 'minmax', 'delta': 'minmax', 'hybrid_minmax_weight': 0.5}
                    or abs(candidate['score_component'] - ((score-low)/(high-low) if high > low else 0.5)) > 1e-10
                    or abs(candidate['delta_component'] - (delta/max_delta if max_delta else 0.0)) > 1e-10
                    or candidate.get('official_score_missing_penalty') != 0.0):
                raise CalibrationError('parent_selection_component_mismatch')
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


def source_sha(node):
    return hashlib.sha256(node['code'].encode('utf-8', errors='surrogatepass')).hexdigest()


def validate_node_bindings(nodes, trace, jobs, rescored, imported_ids=()):
    """Bind every scored node to its own trusted job, and every new node to an operator."""
    by_id, by_step = {n['id']: n for n in nodes}, {n['step']: n for n in nodes}
    if len(by_id) != len(nodes) or len(by_step) != len(nodes) or sorted(by_step) != list(range(len(nodes))):
        raise CalibrationError('journal_identity_invalid')
    if nodes and (nodes[0].get('step') != 0 or nodes[0].get('code') != '' or
                  nodes[0].get('metric') is not None or nodes[0].get('parents') != [] or
                  nodes[0].get('is_buggy') is not True):
        raise CalibrationError('journal_root_invalid')
    job_by_id = {j['job_id']: j for j in jobs if j.get('job_id')}
    traced_ids = [event['child_node_id'] for event in trace]
    if len(traced_ids) != len(set(traced_ids)) or set(traced_ids) & set(imported_ids):
        raise CalibrationError('imported_node_regenerated_or_duplicate')
    for node in nodes:
        if any(step not in by_step or step >= node['step'] for step in node['parents']):
            raise CalibrationError('journal_ancestry_invalid')
        if node['step'] and node['id'] not in set(imported_ids) | set(traced_ids):
            raise CalibrationError('journal_new_node_untraced')
        expected_children = [n['step'] for n in nodes if node['step'] in n['parents']]
        if sorted(node.get('children', [])) != sorted(expected_children):
            raise CalibrationError('journal_child_links_invalid')
        if node['is_buggy'] is False:
            job_id = node.get('metric_info', {}).get('job_id')
            job = job_by_id.get(job_id)
            if (job_id not in rescored or job is None or not node.get('code')
                    or node.get('metric') != rescored[job_id]['trusted_precision_accuracy']
                    or source_sha(node) != job.get('submitted_source_sha256')
                    or node['metric_info'].get('evaluation') != job.get('evaluation')
                    or node['metric_info'].get('score') != job.get('score')):
                raise CalibrationError('journal_trusted_score_or_source_mismatch')
    for event in trace:
        child = by_id[event['child_node_id']]
        if ([by_step[s]['id'] for s in child['parents']] != event['parent_node_ids']
                or child.get('operators_used') != [event['operator']]):
            raise CalibrationError('operator_ancestry_mismatch')


def pool_terminal_ids(nodes, trace):
    """Exactly two initial drafts, each either valid or replaced by one valid debug child."""
    by_id = {node['id']: node for node in nodes}
    drafts = [event for event in trace if event['operator'] == 'draft']
    if len(drafts) != 2 or any(event['operator'] not in ('draft', 'debug') for event in trace):
        raise CalibrationError('pool_draft_shape_invalid')
    terminals, used_debug = [], set()
    for event in drafts:
        node = by_id[event['child_node_id']]
        repairs = [item for item in trace if item['operator'] == 'debug' and item['parent_node_ids'] == [node['id']]]
        if node['is_buggy'] is False:
            if repairs:
                raise CalibrationError('pool_valid_seed_repaired')
        elif node['is_buggy'] is True and len(repairs) == 1:
            node = by_id[repairs[0]['child_node_id']]
            used_debug.add(node['id'])
        else:
            raise CalibrationError('pool_requires_two_valid_terminal_seeds')
        if (node['is_buggy'] is not False or not node.get('code') or
                type(node.get('metric')) not in (int, float) or not math.isfinite(node['metric'])):
            raise CalibrationError('pool_requires_two_valid_terminal_seeds')
        terminals.append(node['id'])
    if used_debug != {event['child_node_id'] for event in trace if event['operator'] == 'debug'}:
        raise CalibrationError('pool_debug_ancestry_invalid')
    if len(set(terminals)) != 2:
        raise CalibrationError('pool_seed_identity_duplicate')
    if set(terminals) != {node['id'] for node in nodes if node['is_buggy'] is False}:
        raise CalibrationError('pool_valid_nodes_not_exact_terminals')
    return terminals


def validate_imported_nodes(nodes, pool_nodes):
    """Imported bytes/metrics/parents remain exact; only descendant child links may grow."""
    if len(nodes) < len(pool_nodes):
        raise CalibrationError('seed_import_journal_missing')
    for current, original in zip(nodes, pool_nodes):
        if ({k:v for k,v in current.items() if k != 'children'} !=
                {k:v for k,v in original.items() if k != 'children'} or
                not set(original['children']) <= set(current['children'])):
            raise CalibrationError('seed_import_journal_mismatch')


def validate_operator_schedule(trace, kind, completed):
    """A fallback draft is a new main call; the imported generation is never rerun."""
    generations = {0} if kind == 'fresh_seed_pool' else {1, 2}
    for event in trace:
        if (type(event.get('generation_id')) is not int or event['generation_id'] not in generations
                or event.get('operator') not in ('draft', 'improve', 'crossover', 'debug')):
            raise CalibrationError('operator_generation_schedule_invalid')
    main_generations = Counter(event['generation_id'] for event in trace if event['operator'] != 'debug')
    if completed and main_generations != {generation: 2 for generation in generations}:
        raise CalibrationError('completed_main_generation_shape_invalid')


def validate_trial(repo_root, manifest, trial, snapshot):
    spec = RunSpec(trial, manifest['identity'])
    directory = repo_root / spec.run_dir
    result = read_json(directory / 'run-result.json')
    expected = {'schema': 'shared-start-inner-result.v1', 'manifest_identity': manifest['identity'],
        'trial_id': trial.trial_id, 'run_key': trial.run_key(), 'cell_key': trial.cell_key(),
        'kind': trial.kind, 'pool_id': trial.pool_id,
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
    if live.get('schema') != 'shared-start-live-jobs.v1' or live.get('run_key') != trial.run_key() or any(live.get(key) != [] for key in ('live_jobs', 'uncertain_submissions', 'submission_intents')):
        raise CalibrationError('trial_live_jobs_unresolved')
    requests = [record for record in snapshot['requests'] if record['run_identity'] == trial.identity()]
    if len({r['request_id'] for r in snapshot['requests']}) != len(snapshot['requests']):
        raise CalibrationError('request_id_duplicate')
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
    validate_operator_schedule(trace, trial.kind, result['status'] == 'completed')
    pool_bundle, pool_result, pool_verified, imported_ids = None, None, None, []
    initial_parent_candidate_ids = None
    if trial.kind == 'continuation':
        matching_pools = [item for item in trials(repo_root, manifest['source_commit'])
                          if item.kind == 'fresh_seed_pool' and item.pool_id == trial.pool_id]
        if len(matching_pools) != 1:
            raise CalibrationError('continuation_pool_identity_invalid')
        pool_trial = matching_pools[0]
        pool_directory = repo_root / RunSpec(pool_trial, manifest['identity']).run_dir
        pool_verified = validate_trial(repo_root, manifest, pool_trial, snapshot)
        pool_bundle = read_json(pool_directory / 'seed-bundle.json')
        bundle_sha = sha256_file(pool_directory / 'seed-bundle.json')
        if snapshot.get('seed_pools', {}).get(trial.pool_id) != {
                'pool_run_identity': pool_trial.identity(), 'bundle_sha256': bundle_sha}:
            raise CalibrationError('seed_bundle_ledger_seal_mismatch')
        if pool_bundle != pool_verified.get('pool_bundle'):
            raise CalibrationError('seed_bundle_independent_verification_mismatch')
        main_events = [event for event in trace if event['operator'] != 'debug']
        if main_events:
            first_main = main_events[0]
            initial_parent_candidate_ids = [candidate.get('node_id') for candidate in
                (first_main.get('selection') or {}).get('candidates', [])]
            if (first_main['operator'] not in ('improve', 'crossover')
                    or len(initial_parent_candidate_ids) != 2
                    or set(initial_parent_candidate_ids) != set(pool_bundle['terminal_node_ids'])):
                raise CalibrationError('initial_shared_population_not_exact_two_seeds')
        pool_result = read_json(pool_directory / 'run-result.json')
        pool_nodes = json_lines(pool_directory / 'journal.jsonl')
        imported_ids = [n['id'] for n in pool_nodes]
        validate_imported_nodes(nodes, pool_nodes)
        imported_jobs = [job for job in pool_result['jobs'] if job.get('job_id') in pool_result['prediction_artifacts']]
        expected_import = {'schema': 'shared-start-seed-import.v1',
            'pool_id': trial.pool_id, 'pool_run_identity': pool_trial.identity(),
            'bundle_sha256': bundle_sha,
            'journal_sha256': pool_bundle['journal_sha256'],
            'terminal_node_ids': pool_bundle['terminal_node_ids'],
            'imported_node_ids': imported_ids,
            'imported_job_ids': [job['job_id'] for job in imported_jobs],
            'prediction_artifacts': pool_result['prediction_artifacts']}
        if result.get('seed_import') != expected_import or result.get('imported_jobs') != imported_jobs:
            raise CalibrationError('seed_import_receipt_mismatch')
        if set(pool_result['job_ids']) & set(result['job_ids']):
            raise CalibrationError('imported_job_duplicate_accounting')
        for job_id, digest in pool_result['prediction_artifacts'].items():
            copied_prediction = directory / 'predictions' / (job_id + '.csv')
            if copied_prediction.is_symlink() or sha256_file(copied_prediction) != digest:
                raise CalibrationError('seed_import_prediction_copy_mismatch')
    elif result.get('seed_import') is not None or result.get('imported_jobs', []) != []:
        raise CalibrationError('pool_cannot_import_seeds')
    behavioral = mechanisms(trace, nodes, trial.decoded)
    if trial.kind == 'continuation':
        behavioral['initial_parent_candidate_ids'] = initial_parent_candidate_ids
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
    expected_artifacts = {job['job_id']: job['evaluation']['prediction_sha256']
                          for job in result['jobs'] if job.get('evidence_verified') is True}
    if artifacts != expected_artifacts:
        raise CalibrationError('new_prediction_artifacts_binding_mismatch')
    all_jobs, all_rescored, all_artifacts = list(result['jobs']), dict(rescored), dict(artifacts)
    if pool_result is not None:
        all_jobs += pool_result['jobs']
        all_rescored.update(pool_verified['rescored_jobs'])
        all_artifacts.update(pool_result['prediction_artifacts'])
    validate_node_bindings(nodes, trace, all_jobs, all_rescored, imported_ids)
    incumbent = result.get('incumbent')
    value = None
    if incumbent is not None:
        matching = [node for node in nodes if node['id'] == incumbent['node_id']]
        if len(matching) != 1 or incumbent['job_id'] not in all_rescored or hashlib.sha256(matching[0]['code'].encode('utf-8', errors='surrogatepass')).hexdigest() != incumbent['code_sha256'] or matching[0]['metric_info'].get('job_id') != incumbent['job_id']:
            raise CalibrationError('incumbent_source_or_job_unproven')
        value = incumbent['validation_score']
        valid_scores = [node['metric'] for node in nodes if not node['is_buggy'] and isinstance(node.get('code'), str) and node['code'] and type(node.get('metric')) in (int, float)]
        if not valid_scores or value != max(valid_scores) or value != all_rescored[incumbent['job_id']]['trusted_precision_accuracy'] or all_artifacts.get(incumbent['job_id']) != incumbent['prediction_sha256']:
            raise CalibrationError('incumbent_score_or_prediction_mismatch')
    completed = result['status'] == 'completed'
    if (costs['main'] > trial.limits['max_main_requests'] or costs['debug'] > trial.limits['max_debug_requests']
            or len(requests) > trial.limits['max_requests'] or costs['charged_tokens'] > trial.limits['max_tokens']):
        raise CalibrationError('trial_budget_overrun')
    if completed and (costs['main'] != trial.main_target or result['main_candidates'] != trial.main_target or result['generations_completed'] != (1 if trial.kind == 'fresh_seed_pool' else 3) or any(record['status'] != 'known' for record in requests) or len(trace) != len(requests)
            or costs['debug'] != behavioral['operator_counts']['debug']):
        raise CalibrationError('completed_trial_shape_invalid')
    status = 'completed' if completed and value is not None else 'terminated' if result['status'] == 'cell_terminated' else 'failed'
    if trial.kind == 'fresh_seed_pool' and status == 'completed':
        terminals = pool_terminal_ids(nodes, trace)
        files = ['run-result.json', 'journal.jsonl', 'operator-trace.jsonl', 'request-trace.jsonl', 'generation-outcomes.jsonl']
        files += ['predictions/' + job_id + '.csv' for job_id in sorted(artifacts)]
        pool_bundle = {'schema': 'shared-start-seed-bundle.v1', 'manifest_identity': manifest['identity'],
            'pool_id': trial.pool_id, 'pool_run_identity': trial.identity(),
            'pool_result_sha256': sha256_file(directory / 'run-result.json'),
            'journal_sha256': result['journal_sha256'], 'terminal_node_ids': terminals,
            'artifact_hashes': {name: sha256_file(directory / name) for name in files}}
    return {'normalized': {'trial_id': trial.trial_id, 'status': status, 'validation_accuracy': value,
                'requests': len(requests), 'charged_tokens': costs['charged_tokens'], 'reason': result['termination']},
            'behavior': behavioral, 'rescored_jobs': rescored, 'job_counts': result['job_counts'],
            'pool_bundle': pool_bundle if trial.kind == 'fresh_seed_pool' else None,
            'kind': trial.kind, 'pool_id': trial.pool_id,
            'seed_incumbent_retained': bool(incumbent and incumbent['node_id'] in imported_ids),
            'result_sha256': sha256_file(directory / 'run-result.json')}
