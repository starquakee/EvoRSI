from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from research.shared_start.seeds import validate_seed_nodes, restore_journal
from research.shared_start.specs import CalibrationError


def fixture_nodes():
    def node(i, parents, children, good):
        return {'step': i, 'id': 'node-' + str(i), 'parents': parents, 'children': children,
            'code': 'x=' + str(i) if i else '', 'plan': 'plan' if i else '',
            'analysis': 'analysis' if i else '', 'metric': .8 if good else None,
            'metric_info': {'job_id': 'job-' + str(i), 'status': 'completed' if good else 'failed'},
            'metric_maximize': True, 'is_buggy': not good, 'creation_time': 123 + i,
            'operators_used': ([] if not i else ['draft'] if i in (1, 3) else ['debug']),
            'operators_metrics': [] if not i else [{'usage': {'total_tokens': 100}}],
            '_term_out': ['test'], 'term_out': 'test', 'exec_time': .1, 'exit_code': 0 if good else 1}
    nodes = [node(0, [], [1, 3], False), node(1, [0], [2], False),
             node(2, [1], [], True), node(3, [0], [], True)]
    trace = []
    for row in nodes[1:]:
        parent = nodes[row['parents'][0]]
        digest = lambda code: hashlib.sha256(code.encode()).hexdigest()
        trace.append({'operator': 'debug' if row['step'] == 2 else 'draft', 'generation_id': 0,
            'child_node_id': row['id'], 'parent_node_ids': [parent['id']],
            'parent_code_sha256': [digest(parent['code'])], 'child_code_sha256': digest(row['code'])})
    return nodes, trace


def test_repaired_draft_retains_ancestry_and_exactly_two_terminal_seeds():
    nodes, trace = fixture_nodes()
    assert validate_seed_nodes(nodes, trace) == ['node-2', 'node-3']


@pytest.mark.parametrize('mutation', ['invalid_terminal', 'branch', 'source', 'foreign_parent', 'generation'])
def test_bad_seed_pool_fails_closed(mutation):
    nodes, trace = fixture_nodes()
    if mutation == 'invalid_terminal':
        nodes[2]['is_buggy'] = True
    elif mutation == 'branch':
        nodes[1]['children'].append(3)
    elif mutation == 'source':
        nodes[2]['code'] = 'modified'
    elif mutation == 'foreign_parent':
        trace[1]['parent_node_ids'] = ['foreign']
    else:
        trace[1]['generation_id'] = 1
    with pytest.raises(CalibrationError):
        validate_seed_nodes(nodes, trace)


def test_decoder_preserves_private_metrics_and_canonical_ancestry():
    pytest.importorskip('dojo')
    nodes, _ = fixture_nodes()
    journal = restore_journal(nodes)
    assert journal.nodes[2].parents[0] is journal.nodes[1]
    assert next(iter(journal.nodes[1].children)) is journal.nodes[2]
    assert journal.nodes[1].metric.info == nodes[1]['metric_info']
    assert journal.nodes[2].metric.info == nodes[2]['metric_info']
    assert journal.nodes[2].operators_metrics == nodes[2]['operators_metrics']
    # Upstream stage_name counts the buggy synthetic root as a debug parent.
    assert journal.nodes[2].debug_depth == 2
    assert journal.is_root_node(journal.nodes[0])


def bundle_fixture(tmp_path, monkeypatch):
    from research.shared_start import seeds
    directory = tmp_path / 'pool'
    directory.mkdir()
    nodes, trace = fixture_nodes()
    pool = SimpleNamespace(kind='fresh_seed_pool', pool_id='pool-1', identity=lambda: 'pool-id')
    run = SimpleNamespace(kind='continuation', pool_id='pool-1')
    monkeypatch.setattr(seeds, 'trials', lambda *args: [pool])
    monkeypatch.setattr(seeds, 'RunSpec', lambda *args: SimpleNamespace(run_dir='pool'))
    journal = '\n'.join(json.dumps(n) for n in nodes) + '\n'
    (directory / 'journal.jsonl').write_text(journal)
    (directory / 'operator-trace.jsonl').write_text('\n'.join(json.dumps(e) for e in trace) + '\n')
    for name in ('request-trace.jsonl', 'generation-outcomes.jsonl'):
        (directory / name).write_text('{}\n')
    predictions = directory / 'predictions'
    predictions.mkdir()
    jobs, artifacts = [], {}
    for node in (nodes[2], nodes[3]):
        jid = node['metric_info']['job_id']
        path = predictions / (jid + '.csv')
        path.write_text('id,label\n1,0\n')
        digest = seeds.sha256_file(path)
        jobs.append({'job_id': jid, 'evidence_verified': True,
                     'evaluation': {'prediction_sha256': digest}})
        artifacts[jid] = digest
    result = {'status': 'completed', 'termination': 'completed', 'run_identity': 'pool-id',
        'manifest_identity': 'manifest', 'worker_cleanup_verified': True,
        'main_candidates': 2, 'generations_completed': 1,
        'journal_sha256': seeds.sha256_file(directory / 'journal.jsonl'),
        'prediction_artifacts': artifacts, 'jobs': jobs}
    (directory / 'run-result.json').write_text(json.dumps(result))
    names = list(seeds.ARTIFACT_NAMES) + ['predictions/' + j + '.csv' for j in artifacts]
    hashes = {name: seeds.sha256_file(directory / name) for name in names}
    bundle = {'schema': seeds.BUNDLE_SCHEMA, 'manifest_identity': 'manifest', 'pool_id': 'pool-1',
        'pool_run_identity': 'pool-id', 'pool_result_sha256': hashes['run-result.json'],
        'journal_sha256': hashes['journal.jsonl'], 'terminal_node_ids': ['node-2', 'node-3'],
        'artifact_hashes': hashes}
    (directory / 'seed-bundle.json').write_text(json.dumps(bundle))
    return directory, run, {'source_commit': 'commit', 'identity': 'manifest'}, seeds.sha256_file(directory / 'seed-bundle.json')


def test_load_sealed_pool_imports_only_scored_jobs_once(tmp_path, monkeypatch):
    from research.shared_start.seeds import load_seed_pool
    directory, run, manifest, seal = bundle_fixture(tmp_path, monkeypatch)
    loaded = load_seed_pool(tmp_path, manifest, run, expected_bundle_sha256=seal)
    assert loaded['receipt']['imported_node_ids'] == ['node-0', 'node-1', 'node-2', 'node-3']
    assert loaded['receipt']['imported_job_ids'] == ['job-2', 'job-3']
    assert loaded['receipt']['bundle_sha256'] == seal


@pytest.mark.parametrize('target', ['journal.jsonl', 'predictions/job-2.csv', 'seed-bundle.json'])
def test_seed_tamper_fails_before_import(tmp_path, monkeypatch, target):
    from research.shared_start.seeds import load_seed_pool
    directory, run, manifest, seal = bundle_fixture(tmp_path, monkeypatch)
    path = directory / target
    path.write_text(path.read_text() + ' ')
    with pytest.raises(CalibrationError, match='seed_.*(hash|seal).*mismatch'):
        load_seed_pool(tmp_path, manifest, run, expected_bundle_sha256=seal)
