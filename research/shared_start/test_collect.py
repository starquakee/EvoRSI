"""Independent pool/continuation provenance tests; no model or sandbox calls."""
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.shared_start import collect as c
from research.shared_start.specs import CalibrationError


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def lines(path, values):
    path.write_text(''.join(json.dumps(value, sort_keys=True) + '\n' for value in values))


class FakeTrial:
    source_commit = 'a' * 40
    decoded = {'crossover_prob': .5, 'score_weight': 1., 'delta_weight': .4, 'novelty_weight': .25,
               'max_debug_depth': 1}
    record = {'config_id': 'c04', 'repeat': 0}
    inner_seed = 100
    pool_id = 'pool00'

    def __init__(self, name, kind):
        self.trial_id, self.kind = name, kind
        self.main_target = 2 if kind == 'fresh_seed_pool' else 4
        self.limits = {'max_main_requests': self.main_target, 'max_debug_requests': 2,
                       'max_requests': self.main_target + 2, 'max_tokens': 40000}

    def identity(self):
        return self.trial_id + '-identity'

    def run_key(self):
        return self.trial_id

    def cell_key(self):
        return self.trial_id


def root_node():
    return {'id': 'root', 'step': 0, 'code': '', 'plan': '', 'analysis': '', 'metric': None,
            'metric_info': {}, 'metric_maximize': True, 'is_buggy': True, 'parents': [],
            'children': [], 'operators_used': [], 'creation_time': 0., 'operators_metrics': [],
            'exec_time': 0., 'exit_code': 0, '_term_out': []}


def candidate(node_id, step, score, parent, operator):
    row = root_node()
    row.update(id=node_id, step=step, code='print(' + repr(node_id) + ')', metric=score,
               is_buggy=score is None, parents=[parent], operators_used=[operator])
    return row


def operator_event(node, parent, generation, selection=None):
    return {'child_node_id': node['id'], 'child_code_sha256': c.source_sha(node),
        'parent_node_ids': [parent['id']], 'parent_code_sha256': [c.source_sha(parent)],
        'operator': node['operators_used'][0], 'generation_id': generation,
        'selection': selection, 'config': {'crossover_prob': .5, 'max_debug_depth': 1,
            'experience': {'parent_selection': {'weights': {'score': 1., 'delta': .4, 'novelty': .25}}}}}


def selection(nodes, selected):
    scores = [n['metric'] for n in nodes]
    lo, hi = min(scores), max(scores)
    rows = []
    for node in nodes:
        score = (node['metric'] - lo) / (hi - lo) if hi != lo else .5
        rows.append({'node_id': node['id'], 'fitness': node['metric'], 'official_score': node['metric'],
            'official_status': 'completed', 'official_score_missing_penalty': 0.,
            'component_normalization': {'score': 'minmax', 'delta': 'minmax', 'hybrid_minmax_weight': .5},
            'score_component': score, 'delta_component': 0., 'novelty_component': 1.,
            'utility': score + .25})
    total = sum(math.exp(row['utility']) for row in rows)
    for row in rows:
        row['probability'] = math.exp(row['utility']) / total
    return {'enabled': True, 'weights': {'score': 1., 'delta': .4, 'novelty': .25},
            'selected_node_ids': [selected['id']], 'candidates': rows, 'temperature': 1.}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    pool = FakeTrial('pool00', 'fresh_seed_pool')
    cont = FakeTrial('r00-c04', 'continuation')
    manifest = {'identity': 'b' * 64, 'source_commit': 'a' * 40,
        'policy_version': 'test-policy', 'policy_sha256': 'c' * 64,
        'dataset': {'registry_version': 'test-registry', 'validation_labels_sha256': ''}}
    answers = tmp_path / 'validation-answers.csv'
    answers.write_text('id,label\n0,0\n1,1\n')
    manifest['dataset']['validation_labels_sha256'] = c.sha256_file(answers)
    snapshot = {'requests': []}
    monkeypatch.setattr(c, 'RunSpec', lambda t, m: SimpleNamespace(run_dir='runs/' + t.trial_id))
    monkeypatch.setattr(c, 'comparison_config_identity', lambda spec: 'config')
    monkeypatch.setattr(c, 'protocol_identity', lambda: 'protocol')
    monkeypatch.setattr(c, 'trials', lambda root, commit: (pool, cont))
    monkeypatch.setattr(c.EvaluatorRegistry, 'load', lambda path: SimpleNamespace(
        task=lambda name: SimpleNamespace(answer_path=answers, split='validation')))
    monkeypatch.setattr(c.deploy, 'validate_job_snapshot', lambda path, jid: None)
    monkeypatch.setattr(c.deploy, 'job_snapshot_host_path', lambda path, jid, **kw: Path(path))

    def write_unit(trial, repair=False, pool_data=None, pool_score=1., new_score=.5, fallback=False):
        directory = tmp_path / 'runs' / trial.trial_id
        directory.mkdir(parents=True)
        if pool_data is None:
            nodes = [root_node(), candidate('draft1', 1, None if repair else pool_score, 0, 'draft'),
                     candidate('draft2', 2, .5, 0, 'draft')]
            new_nodes = nodes[1:]
            if repair:
                nodes.append(candidate('repair1', 3, 1., 1, 'debug'))
                new_nodes = nodes[1:]
            trace = [operator_event(n, nodes[n['parents'][0]], 0) for n in new_nodes]
        else:
            nodes = copy.deepcopy(pool_data['nodes'])
            parents = [n for n in nodes if not n['is_buggy']]
            new_nodes = [candidate('new' + str(i), len(nodes) + i, new_score, parents[0]['step'], 'improve') for i in range(4)]
            if fallback:
                new_nodes[2]['parents'] = [0]
                new_nodes[2]['operators_used'] = ['draft']
            nodes += new_nodes
            trace = [operator_event(n, nodes[n['parents'][0]], 1 + i // 2,
                                    None if n['operators_used'] == ['draft'] else selection(parents, parents[0]))
                     for i, n in enumerate(new_nodes)]
        for node in nodes:
            node['children'] = [n['step'] for n in nodes if node['step'] in n['parents']]
        jobs, predictions, request_trace = [], {}, []
        for index, node in enumerate(new_nodes):
            req = {'request_id': trial.trial_id + '-req-' + str(index), 'run_identity': trial.identity(),
                   'kind': 'debug' if node['operators_used'] == ['debug'] else 'main',
                   'charged_tokens': 100, 'reserved_tokens': 200, 'status': 'known'}
            snapshot['requests'].append(req)
            request_trace.append({'request_id': req['request_id'], 'run_key': trial.run_key(),
                'kind': req['kind'], 'reserved_tokens': 200, 'request': c.comparison_request_profile()})
            if node['is_buggy']:
                jobs.append({'job_id': trial.trial_id + '-' + node['id'], 'evidence_verified': False,
                             'score': None, 'status': 'failed'})
                continue
            jid = trial.trial_id + '-' + node['id']
            path = directory / 'predictions' / (jid + '.csv')
            path.parent.mkdir(exist_ok=True)
            path.write_text('id,label\n0,0\n1,' + ('1' if node['metric'] == 1. else '0') + '\n')
            digest = c.sha256_file(path)
            evaluation = {'evaluator': 'external_trusted', 'task_id': 'comparison_spambase',
                'split': 'validation', 'metric': 'accuracy', 'direction': 'maximize',
                'answer_sha256': manifest['dataset']['validation_labels_sha256'],
                'registry_version': 'test-registry', 'score': node['metric'], 'row_count': 2,
                'prediction_sha256': digest, 'prediction_snapshot': str(path)}
            job = {'job_id': jid, 'evidence_verified': True, 'score': node['metric'],
                'source_identity_verified': True, 'worker_cleanup_verified': True,
                'status': 'completed', 'evaluation': evaluation, 'submitted_source_sha256': c.source_sha(node),
                'source_gate': {'allowed': True, 'entrypoint': 'main.py',
                    'entrypoint_sha256': c.source_sha(node),
                    'source_sha256': c._single_file_tree_sha256('main.py', node['code'].encode()),
                    'policy_version': manifest['policy_version'], 'policy_sha256': manifest['policy_sha256']}}
            node['metric_info'] = {'job_id': jid, 'score': node['metric'], 'evaluation': evaluation}
            jobs.append(job)
            predictions[jid] = digest
        imported_jobs, receipt = [], None
        if pool_data:
            p_result, bundle = pool_data['result'], pool_data['bundle']
            imported_jobs = [j for j in p_result['jobs'] if j.get('job_id') in p_result['prediction_artifacts']]
            receipt = {'schema': 'shared-start-seed-import.v1', 'pool_id': pool.pool_id,
                'pool_run_identity': pool.identity(), 'bundle_sha256': c.sha256_file(pool_data['directory'] / 'seed-bundle.json'),
                'journal_sha256': bundle['journal_sha256'], 'terminal_node_ids': bundle['terminal_node_ids'],
                'imported_node_ids': [n['id'] for n in pool_data['nodes']],
                'imported_job_ids': [j['job_id'] for j in imported_jobs], 'prediction_artifacts': p_result['prediction_artifacts']}
            for jid in p_result['prediction_artifacts']:
                (directory / 'predictions' / (jid + '.csv')).write_bytes((pool_data['directory'] / 'predictions' / (jid + '.csv')).read_bytes())
        incumbent = max((n for n in nodes if not n['is_buggy']), key=lambda n: n['metric'])
        own_requests = [r for r in snapshot['requests'] if r['run_identity'] == trial.identity()]
        result = {'schema': 'shared-start-inner-result.v1', 'manifest_identity': manifest['identity'],
            'trial_id': trial.trial_id, 'kind': trial.kind, 'pool_id': trial.pool_id,
            'run_key': trial.run_key(), 'cell_key': trial.cell_key(), 'run_identity': trial.identity(),
            'cell_identity': trial.identity(), 'decoded': trial.decoded, 'inner_seed': trial.inner_seed,
            'config_id': 'c04', 'repeat': 0, 'runner_commit': trial.source_commit, 'protocol_identity': 'protocol',
            'config_identity': 'config', 'model_profile_identity': c.ModelProfile().identity(),
            'dataset': manifest['dataset'], 'policy_version': manifest['policy_version'],
            'policy_sha256': manifest['policy_sha256'], 'metric': 'accuracy', 'metric_direction': 'maximize',
            'worker_cleanup_verified': True, 'status': 'completed', 'termination': 'completed',
            'requests': {'main': trial.main_target, 'debug': int(repair), 'charged_tokens': 100 * len(own_requests)},
            'main_candidates': trial.main_target, 'generations_completed': 1 if pool_data is None else 3,
            'operator_counts': c.mechanisms(trace, nodes, trial.decoded)['operator_counts'],
            'prediction_artifacts': predictions, 'jobs': jobs, 'job_ids': sorted(j['job_id'] for j in jobs),
            'job_counts': {}, 'seed_import': receipt, 'imported_jobs': imported_jobs,
            'incumbent': {'node_id': incumbent['id'], 'job_id': incumbent['metric_info']['job_id'],
                'code_sha256': c.source_sha(incumbent), 'validation_score': incumbent['metric'],
                'prediction_sha256': incumbent['metric_info']['evaluation']['prediction_sha256']}}
        lines(directory / 'journal.jsonl', nodes)
        lines(directory / 'operator-trace.jsonl', trace)
        lines(directory / 'request-trace.jsonl', request_trace)
        lines(directory / 'generation-outcomes.jsonl', [])
        for name, key in [('journal.jsonl', 'journal_sha256'), ('operator-trace.jsonl', 'operator_trace_sha256'),
                          ('request-trace.jsonl', 'request_trace_sha256'), ('generation-outcomes.jsonl', 'generation_outcomes_sha256')]:
            result[key] = c.sha256_file(directory / name)
        dump(directory / 'live-jobs.json', {'schema': 'shared-start-live-jobs.v1', 'run_key': trial.run_key(),
                                         'live_jobs': [], 'uncertain_submissions': [], 'submission_intents': []})
        dump(directory / 'run-result.json', result)
        data = {'directory': directory, 'nodes': nodes, 'trace': trace, 'result': result}
        if trial is pool:
            verified = c.validate_trial(tmp_path, manifest, pool, snapshot)
            data['bundle'] = verified['pool_bundle']
            dump(directory / 'seed-bundle.json', data['bundle'])
            snapshot.setdefault('seed_pools', {})[trial.pool_id] = {
                'pool_run_identity': trial.identity(),
                'bundle_sha256': c.sha256_file(directory / 'seed-bundle.json')}
        return data
    return SimpleNamespace(root=tmp_path, manifest=manifest, snapshot=snapshot,
                           pool=pool, cont=cont, write=write_unit)


@pytest.mark.parametrize('repair', [False, True])
def test_independent_pool_and_seed_incumbent_no_duplicate_cost(experiment, repair):
    e = experiment
    p = e.write(e.pool, repair=repair)
    d = e.write(e.cont, pool_data=p)
    verified = c.validate_trial(e.root, e.manifest, e.cont, e.snapshot)
    assert verified['normalized']['requests'] == 4
    assert verified['normalized']['charged_tokens'] == 400
    assert verified['normalized']['validation_accuracy'] == 1.
    assert verified['seed_incumbent_retained'] is True
    assert len(verified['rescored_jobs']) == 4
    assert sum(r['charged_tokens'] for r in e.snapshot['requests']) == 600 + int(repair) * 100


def test_new_candidate_incumbent_independently_rescored(experiment):
    e = experiment
    p = e.write(e.pool, pool_score=.5)
    e.write(e.cont, pool_data=p, new_score=1.)
    verified = c.validate_trial(e.root, e.manifest, e.cont, e.snapshot)
    assert verified['normalized']['validation_accuracy'] == 1.
    assert verified['seed_incumbent_retained'] is False
    assert verified['normalized']['requests'] == 4


def test_new_fallback_draft_is_a_continuation_main_not_seed_regeneration(experiment):
    e = experiment
    p = e.write(e.pool)
    e.write(e.cont, pool_data=p, fallback=True)
    verified = c.validate_trial(e.root, e.manifest, e.cont, e.snapshot)
    assert verified['normalized']['status'] == 'completed'
    assert verified['normalized']['requests'] == 4
    assert verified['behavior']['operator_counts'] == {'draft': 1, 'improve': 3, 'crossover': 0, 'debug': 0}
    assert set(verified['behavior']['initial_parent_candidate_ids']) == set(p['bundle']['terminal_node_ids'])


@pytest.mark.parametrize('first_operator', ['improve', 'draft'])
def test_first_continuation_selection_must_observe_both_shared_seeds(experiment, first_operator):
    e = experiment
    p = e.write(e.pool)
    d = e.write(e.cont, pool_data=p)
    first = d['trace'][0]
    first['operator'] = first_operator
    # Recompute all utilities/probabilities consistently for the filtered
    # population: numeric trace checks alone cannot detect the missing seed.
    first['selection'] = selection([p['nodes'][1]], p['nodes'][1]) if first_operator == 'improve' else None
    lines(d['directory'] / 'operator-trace.jsonl', d['trace'])
    d['result']['operator_trace_sha256'] = c.sha256_file(d['directory'] / 'operator-trace.jsonl')
    dump(d['directory'] / 'run-result.json', d['result'])
    with pytest.raises(CalibrationError, match='initial_shared_population_not_exact_two_seeds'):
        c.validate_trial(e.root, e.manifest, e.cont, e.snapshot)


@pytest.mark.parametrize('tamper', ['generation_zero', 'unbalanced_generations', 'imported_node_retraced'])
def test_continuation_cannot_rerun_imported_generation_or_node(experiment, tamper):
    e = experiment
    p = e.write(e.pool)
    d = e.write(e.cont, pool_data=p, fallback=True)
    if tamper == 'generation_zero':
        d['trace'][0]['generation_id'] = 0
    elif tamper == 'unbalanced_generations':
        d['trace'][0]['generation_id'] = 2
    else:
        d['trace'][0]['child_node_id'] = p['nodes'][1]['id']
        d['trace'][0]['child_code_sha256'] = c.source_sha(p['nodes'][1])
    lines(d['directory'] / 'operator-trace.jsonl', d['trace'])
    d['result']['operator_trace_sha256'] = c.sha256_file(d['directory'] / 'operator-trace.jsonl')
    dump(d['directory'] / 'run-result.json', d['result'])
    with pytest.raises(CalibrationError):
        c.validate_trial(e.root, e.manifest, e.cont, e.snapshot)


@pytest.mark.parametrize('kind,generation', [('fresh_seed_pool', 1), ('continuation', 0), ('continuation', 3)])
def test_debug_calls_cannot_escape_unit_generations(kind, generation):
    with pytest.raises(CalibrationError, match='generation_schedule'):
        c.validate_operator_schedule([{'operator': 'debug', 'generation_id': generation}], kind, False)


def test_coordinated_bundle_rewrite_refused_by_ledger_seal(experiment):
    e = experiment
    p = e.write(e.pool)
    e.write(e.cont, pool_data=p)
    bundle_path = p['directory'] / 'seed-bundle.json'
    # Even a semantically identical rewrite changes the durable seal.
    bundle_path.write_text(json.dumps(p['bundle'], indent=2))
    with pytest.raises(CalibrationError, match='ledger_seal'):
        c.validate_trial(e.root, e.manifest, e.cont, e.snapshot)


@pytest.mark.parametrize('tamper', ['imported_metric', 'imported_code', 'imported_cost', 'new_source', 'seed_prediction', 'receipt', 'regenerated', 'probability', 'component'])
def test_coherent_result_rewrite_cannot_hide_provenance_tamper(experiment, tamper):
    e = experiment
    p = e.write(e.pool)
    d = e.write(e.cont, pool_data=p)
    r = d['result']
    if tamper in ('imported_metric', 'imported_code'):
        d['nodes'][1]['metric' if tamper == 'imported_metric' else 'code'] = .1 if tamper == 'imported_metric' else 'different code'
    elif tamper == 'imported_cost':
        r['jobs'].append(r['imported_jobs'][0])
        r['job_ids'] = sorted(j['job_id'] for j in r['jobs'])
    elif tamper == 'new_source':
        r['jobs'][0]['submitted_source_sha256'] = c.source_sha(d['nodes'][2])
    elif tamper == 'seed_prediction':
        jid = next(iter(p['result']['prediction_artifacts']))
        (d['directory'] / 'predictions' / (jid + '.csv')).write_text('tampered')
    elif tamper == 'receipt':
        r['seed_import']['bundle_sha256'] = '0' * 64
    elif tamper == 'regenerated':
        d['trace'][0]['operator'] = 'draft'
    elif tamper == 'probability':
        d['trace'][0]['selection']['candidates'][0]['probability'] = .99
    else:
        d['trace'][0]['selection']['candidates'][0]['score_component'] = .7
    lines(d['directory'] / 'journal.jsonl', d['nodes'])
    lines(d['directory'] / 'operator-trace.jsonl', d['trace'])
    r['journal_sha256'] = c.sha256_file(d['directory'] / 'journal.jsonl')
    r['operator_trace_sha256'] = c.sha256_file(d['directory'] / 'operator-trace.jsonl')
    dump(d['directory'] / 'run-result.json', r)
    with pytest.raises(CalibrationError):
        c.validate_trial(e.root, e.manifest, e.cont, e.snapshot)


def test_invalid_terminal_seed_refused_without_replacement():
    nodes = [root_node(), candidate('one', 1, .5, 0, 'draft'), candidate('two', 2, None, 0, 'draft')]
    trace = [operator_event(node, nodes[0], 0) for node in nodes[1:]]
    with pytest.raises(CalibrationError, match='two_valid_terminal'):
        c.pool_terminal_ids(nodes, trace)


def test_validation_prediction_recompute_exact_and_id_bound(tmp_path):
    truth, prediction = tmp_path / 'truth.csv', tmp_path / 'prediction.csv'
    truth.write_text('id,label\n0,0\n1,1\n2,1\n')
    prediction.write_text('id,label\n2,0\n1,1\n0,0\n')
    assert c.recompute_accuracy(prediction, truth)['trusted_precision_accuracy'] == .666667
    prediction.write_text('id,label\n0,0\n1,1\n3,1\n')
    with pytest.raises(CalibrationError, match='id_set'):
        c.recompute_accuracy(prediction, truth)
