"""Import a supervisor-verified seed journal without generation or evaluation.

Only the two valid terminal solutions enter Evo's island population. Invalid
draft/repair ancestry remains in the private journal, preserving experience
features and prompts. Provider costs and jobs stay in the original pool unit.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from research.shared_start.specs import CalibrationError, RunSpec, sha256_file, trials

BUNDLE_SCHEMA = 'shared-start-seed-bundle.v1'
IMPORT_SCHEMA = 'shared-start-seed-import.v1'
ARTIFACT_NAMES = ('run-result.json', 'journal.jsonl', 'operator-trace.jsonl',
                  'request-trace.jsonl', 'generation-outcomes.jsonl')


def _json(path):
    if path.is_symlink() or not path.is_file():
        raise CalibrationError('seed_artifact_missing_or_symlink')
    return json.loads(path.read_text(encoding='utf-8'))


def _jsonl(path):
    if path.is_symlink() or not path.is_file():
        raise CalibrationError('seed_artifact_missing_or_symlink')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def validate_seed_nodes(nodes, trace):
    """Validate a complete one-generation journal and return ordered terminals."""
    if not nodes or [n.get('step') for n in nodes] != list(range(len(nodes))):
        raise CalibrationError('seed_journal_steps_invalid')
    by_id = {n.get('id'): n for n in nodes}
    if len(by_id) != len(nodes) or any(not isinstance(i, str) or not i for i in by_id):
        raise CalibrationError('seed_journal_ids_invalid')
    root = nodes[0]
    if (root.get('parents') != [] or root.get('code') != '' or root.get('plan') != ''
            or root.get('analysis') != '' or root.get('metric') is not None
            or root.get('is_buggy') is not True):
        raise CalibrationError('seed_journal_root_invalid')
    events = {e.get('child_node_id'): e for e in trace}
    if len(events) != len(trace) or set(events) != set(by_id) - {root['id']}:
        raise CalibrationError('seed_trace_node_bijection_invalid')
    expected_children = {i: [] for i in range(len(nodes))}
    for node in nodes[1:]:
        parents = node.get('parents')
        if (not isinstance(parents, list) or len(parents) != 1
                or type(parents[0]) is not int or not 0 <= parents[0] < node['step']):
            raise CalibrationError('seed_ancestry_invalid')
        expected_children[parents[0]].append(node['step'])
        event = events[node['id']]
        parent = nodes[parents[0]]
        if (event.get('generation_id') != 0 or event.get('operator') not in ('draft', 'debug')
                or event.get('parent_node_ids') != [parent['id']]
                or event.get('parent_code_sha256') != [_code_sha(parent['code'])]
                or event.get('child_code_sha256') != _code_sha(node['code'])):
            raise CalibrationError('seed_operator_binding_invalid')
        if (event['operator'] == 'draft') != (parent is root):
            raise CalibrationError('seed_operator_ancestry_invalid')
        if event['operator'] == 'debug' and (parent.get('is_buggy') is not True
                or events[parent['id']]['operator'] != 'draft'):
            raise CalibrationError('seed_debug_depth_invalid')
    for node in nodes:
        if sorted(node.get('children', [])) != expected_children[node['step']]:
            raise CalibrationError('seed_children_invalid')
    drafts = [by_id[e['child_node_id']] for e in trace if e['operator'] == 'draft']
    if len(drafts) != 2 or len(trace) > 4:
        raise CalibrationError('seed_main_shape_invalid')
    terminals = []
    for node in drafts:
        children = expected_children[node['step']]
        if children:
            if len(children) != 1:
                raise CalibrationError('seed_debug_branch_invalid')
            node = nodes[children[0]]
        value = node.get('metric')
        if (node.get('is_buggy') is not False or isinstance(value, bool)
                or not isinstance(value, (float, int)) or not math.isfinite(value)
                or not 0 <= value <= 1 or node.get('metric_maximize') is not True):
            raise CalibrationError('seed_pool_invalid_terminal')
        terminals.append(node['id'])
    if {n['id'] for n in nodes[1:] if n.get('is_buggy') is False} != set(terminals):
        raise CalibrationError('seed_good_nodes_not_exact_terminals')
    return terminals


def _code_sha(code):
    return hashlib.sha256(code.encode('utf-8', errors='surrogatepass')).hexdigest()


def restore_journal(nodes):
    """Decode with canonical object references and nested metric provenance.

    The upstream decoder expects flattened metric_info and clones nodes after
    linking them; neither is suitable for faithful shared-start imports.
    """
    from dojo.core.solvers.utils.journal import Journal, Node
    from dojo.core.solvers.utils.metric import MetricValue, WorstMetricValue
    decoded = []
    for row in nodes:
        metric = (WorstMetricValue(maximize=row['metric_maximize'], info=row['metric_info']) if row['metric'] is None else
                  MetricValue(value=row['metric'], info=row['metric_info'],
                              maximize=row['metric_maximize']))
        decoded.append(Node(code=row['code'], plan=row['plan'], step=row['step'],
            id=row['id'], ctime=row['creation_time'], parents=[], children=set(),
            operators_used=row['operators_used'], operators_metrics=row['operators_metrics'],
            _term_out=row['_term_out'], exec_time=row['exec_time'], exit_code=row['exit_code'],
            analysis=row['analysis'], metric=metric, is_buggy=row['is_buggy']))
    for node, row in zip(decoded, nodes):
        node.parents = [decoded[i] for i in row['parents']]
        node.children = {decoded[i] for i in row['children']}
    return Journal(nodes=decoded)


def load_seed_pool(repo_root, manifest, run, *, expected_bundle_sha256):
    """Rehash a collector-approved pool bundle before any continuation call."""
    if run.kind != 'continuation':
        raise CalibrationError('seed_import_requires_continuation')
    matches = [t for t in trials(repo_root, manifest['source_commit'])
               if t.kind == 'fresh_seed_pool' and t.pool_id == run.pool_id]
    if len(matches) != 1:
        raise CalibrationError('seed_pool_identity_missing')
    pool = matches[0]
    directory = repo_root / RunSpec(pool, manifest['identity']).run_dir
    current = directory
    while current != repo_root and current != current.parent:
        if current.is_symlink():
            raise CalibrationError('seed_directory_symlink')
        current = current.parent
    bundle_path = directory / 'seed-bundle.json'
    if (not isinstance(expected_bundle_sha256, str) or len(expected_bundle_sha256) != 64
            or set(expected_bundle_sha256) - set('0123456789abcdef')
            or bundle_path.is_symlink() or not bundle_path.is_file()
            or sha256_file(bundle_path) != expected_bundle_sha256):
        raise CalibrationError('seed_bundle_seal_mismatch')
    bundle = _json(bundle_path)
    expected = {'schema': BUNDLE_SCHEMA, 'manifest_identity': manifest['identity'],
                'pool_id': run.pool_id, 'pool_run_identity': pool.identity()}
    if any(bundle.get(k) != v for k, v in expected.items()):
        raise CalibrationError('seed_bundle_identity_invalid')
    result = _json(directory / 'run-result.json')
    if (result.get('status') != 'completed' or result.get('termination') != 'completed'
            or result.get('run_identity') != pool.identity()
            or result.get('manifest_identity') != manifest['identity']
            or result.get('worker_cleanup_verified') is not True
            or result.get('main_candidates') != 2 or result.get('generations_completed') != 1):
        raise CalibrationError('seed_pool_result_invalid')
    predictions = result['prediction_artifacts']
    names = set(ARTIFACT_NAMES) | {'predictions/' + j + '.csv' for j in predictions}
    if set(bundle.get('artifact_hashes', {})) != names:
        raise CalibrationError('seed_bundle_artifact_set_invalid')
    for name in sorted(names):
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != bundle['artifact_hashes'][name]:
            raise CalibrationError('seed_artifact_hash_mismatch')
    if (bundle.get('pool_result_sha256') != bundle['artifact_hashes']['run-result.json']
            or bundle.get('journal_sha256') != bundle['artifact_hashes']['journal.jsonl']
            or result.get('journal_sha256') != bundle['journal_sha256']):
        raise CalibrationError('seed_bundle_hash_binding_invalid')
    nodes = _jsonl(directory / 'journal.jsonl')
    terminals = validate_seed_nodes(nodes, _jsonl(directory / 'operator-trace.jsonl'))
    if terminals != bundle.get('terminal_node_ids'):
        raise CalibrationError('seed_bundle_terminals_invalid')
    jobs = [j for j in result['jobs'] if j.get('job_id') in predictions]
    if len(jobs) != len(predictions) or len({j['job_id'] for j in jobs}) != len(jobs):
        raise CalibrationError('seed_prediction_job_bijection_invalid')
    for job in jobs:
        jid = job['job_id']
        if (job.get('evidence_verified') is not True
                or job.get('evaluation', {}).get('prediction_sha256') != predictions[jid]
                or bundle['artifact_hashes']['predictions/' + jid + '.csv'] != predictions[jid]):
            raise CalibrationError('seed_prediction_binding_invalid')
    receipt = {'schema': IMPORT_SCHEMA, 'pool_id': run.pool_id,
        'pool_run_identity': pool.identity(), 'bundle_sha256': sha256_file(bundle_path),
        'journal_sha256': bundle['journal_sha256'], 'terminal_node_ids': terminals,
        'imported_node_ids': [n['id'] for n in nodes],
        'imported_job_ids': [j['job_id'] for j in jobs], 'prediction_artifacts': predictions}
    return {'receipt': receipt, 'nodes': nodes, 'jobs': jobs,
            'prediction_artifacts': predictions, 'directory': directory}


def initialize_continuation(solver, pool, run_dir):
    if solver.journal.nodes or solver.state.current_generation != 0 or solver.state.current_step != 0:
        raise CalibrationError('seed_solver_not_fresh')
    solver.journal = restore_journal(pool['nodes'])
    solver.root_node = solver.journal.nodes[0]
    solver.state.current_generation = 1
    solver.state.current_step = len(solver.journal.nodes)
    solver.state.running_time = 0.0
    terminal_ids = pool['receipt']['terminal_node_ids']
    by_id = {node.id: node for node in solver.journal.nodes}
    if (len(terminal_ids) != 2 or len(set(terminal_ids)) != 2
            or set(terminal_ids) != {node.id for node in solver.journal.nodes
                                     if not node.is_buggy}
            or solver.cfg.num_islands != 1 or solver.cfg.max_island_size < 2):
        raise CalibrationError('seed_initial_population_invalid')
    terminals = [by_id[node_id] for node_id in terminal_ids]
    original_seed = solver._seed_solution_database_from_journal

    def seed_shared_initial_population(database):
        # Native resume uses add_nodes_to_islands, whose average-fitness
        # admission rule can drop a weaker second seed. Initialization must
        # use the same unconditional API as native generation zero instead.
        if solver.state.current_generation != 1 or database.num_islands != 1:
            raise CalibrationError('seed_initial_population_wrong_generation')
        database.seed_islands_with_nodes(terminals, [0, 0])
        actual = [node.id for island in database._islands for node in island.nodes]
        if actual != terminal_ids:
            raise CalibrationError('seed_initial_population_incomplete')
        # Only this first initialization is special. New generated candidates
        # retain native add_nodes_to_islands admission and feature semantics.
        solver._seed_solution_database_from_journal = original_seed

    solver._seed_solution_database_from_journal = seed_shared_initial_population
    destination = run_dir / 'predictions'
    destination.mkdir(exist_ok=True)
    for job_id, digest in pool['prediction_artifacts'].items():
        data = (pool['directory'] / 'predictions' / (job_id + '.csv')).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise CalibrationError('seed_copy_prediction_hash_mismatch')
        (destination / (job_id + '.csv')).write_bytes(data)
    solver.save_checkpoint()
