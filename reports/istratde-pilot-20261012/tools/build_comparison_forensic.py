"""Post-stop forensic supplement. Never repairs or overrides the canonical audit."""
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / '.runtime/comparison-istratde'
OUT = ROOT / '.runtime/comparison-reports/forensic'
hashes = {}

def read(path, expected=None):
    assert path.is_file() and not path.is_symlink(), str(path)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert sha == expected, str(path)
    hashes[str(path.relative_to(ROOT))] = sha
    return raw

def obj(path, expected=None):
    return json.loads(read(path, expected))

supervisor = obj(ROOT / '.runtime/comparison-live-supervisor-state.json')
assert supervisor['phase'] == 'finished'
assert not Path('/proc/' + str(supervisor['timeout_pid'])).exists()
report = obj(OUT.parent / 'report.json')
audit = obj(OUT.parent / 'audit.json')
assert audit['verdict'] == 'FAIL'
assert report['ledger_error'] == 'corrupt_ledger:final_phase'
assert report['manifest_verified'] and report['orchestration_state_verified']
manifest = obj(RUN / 'experiment-manifest.v1.json', supervisor['experiment_manifest_sha256'])
assert manifest['source_commit'] == supervisor['source_commit'] == report['source_commit']
data = obj(ROOT / 'research/comparison/data-manifest.v1.json', manifest['data_manifest_sha256'])
ledger = obj(RUN / 'comparison-ledger.json')
assert not ledger['reservations']
assert ledger['stopped']['reason'] == 'final_score_recompute_mismatch:wdbc:11:FIXED'
phase_events = [x for x in ledger['events'] if x['type'] == 'final_phase']
assert len(phase_events) == 1
assert phase_events[0]['barrier_identity'] == ledger['final_phase']['barrier_identity']
timestamp_delta = phase_events[0]['at'] - ledger['final_phase']['at']
assert 0 < timestamp_delta < .001

# This is an independent, explicitly limited reconciliation of recorded requests.
# The original strict ledger validator remains failed.
by_cell = defaultdict(Counter)
by_run = defaultdict(Counter)
requests = ledger['requests']
assert len({r['request_id'] for r in requests}) == len(requests)
reserves = {e['request_id']: e for e in ledger['events'] if e['type'] == 'reserve'}
commits = {e['request_id']: e for e in ledger['events'] if e['type'] == 'commit'}
assert set(reserves) == set(commits) == {r['request_id'] for r in requests}
assert len(reserves) == sum(e['type'] == 'reserve' for e in ledger['events'])
assert len(commits) == sum(e['type'] == 'commit' for e in ledger['events'])
for r in requests:
    run = ledger['runs'][r['run_identity']]
    cell = ledger['cells'][r['cell_identity']]
    assert run['cell_identity'] == r['cell_identity']
    assert run['run_key'].rsplit(':', 1)[0] == cell['cell_key']
    assert r['usage_known'] is True
    assert r['kind'] in ('main', 'debug')
    assert r['charged_tokens'] == r['input_tokens'] + r['output_tokens'] <= r['reserved_tokens']
    assert ledger['started_at'] <= r['reserved_at'] <= r['committed_at'] < ledger['final_phase']['at']
    assert r['committed_at'] <= ledger['started_at'] + ledger['limits']['global']['max_elapsed_seconds']
    reserve, commit = reserves[r['request_id']], commits[r['request_id']]
    assert reserve['run_key'] == run['run_key'] and reserve['kind'] == r['kind']
    assert reserve['tokens'] == r['reserved_tokens']
    for k in ('charged_tokens', 'input_tokens', 'output_tokens', 'kind', 'usage_known', 'reserved_tokens'):
        assert commit[k] == r[k]
    charge = Counter(requests=1, charged_tokens=r['charged_tokens'], input_tokens=r['input_tokens'],
                     output_tokens=r['output_tokens'], **{r['kind'] + '_requests': 1})
    by_cell[cell['cell_key']].update(charge)
    by_run[run['run_key']].update(charge)
totals = sum(by_cell.values(), Counter())
assert totals['requests'] == ledger['global_committed']['requests'] <= 1728
assert totals['charged_tokens'] == ledger['global_committed']['tokens'] <= 5400000
for state in ledger['cells'].values():
    cost = by_cell[state['cell_key']]
    assert cost['requests'] == state['committed']['requests'] <= 64
    assert cost['charged_tokens'] == state['committed']['tokens'] <= 200000
for state in ledger['runs'].values():
    cost = by_run[state['run_key']]
    assert cost['main_requests'] == state['main_requests'] <= 6
    assert cost['debug_requests'] == state['debug_requests'] <= 2

job_keys = ('submitted', 'executed_and_cleaned', 'scored', 'execution_failed', 'scoring_failed',
            'admission_denied', 'timeout', 'cancelled', 'unscored_other')

def count_jobs(jobs):
    c = Counter({k: 0 for k in job_keys})
    for job in jobs:
        status = job.get('status')
        if status == 'admission_denied':
            c['admission_denied'] += 1
            continue
        if not job.get('job_id'):
            continue
        c['submitted'] += 1
        c['executed_and_cleaned'] += (job.get('worker_cleanup_verified') is True
                                    and job.get('execution_never_started') is not True)
        if job.get('evidence_verified') is True:
            bucket = 'scored'
        elif status in ('cancelled', 'timeout'):
            bucket = status
        elif status == 'failed':
            bucket = 'scoring_failed' if job.get('scoring_result') == 'scoring_failed' else 'execution_failed'
        else:
            bucket = 'unscored_other'
        c[bucket] += 1
    return c

def labels(raw):
    rows = list(csv.reader(io.StringIO(raw.decode('utf-8'), newline='')))
    assert rows[0] == ['id', 'label']
    values = {}
    for row in rows[1:]:
        assert len(row) == 2 and row[0] and row[0] not in values and row[1] in ('0', '1')
        values[row[0]] = int(row[1])
    assert values
    return values

rows, final_cases, curves = [], [], []
all_jobs, all_operators, all_outcomes = Counter(), Counter(), Counter()
job_ids = []
for row in report['matrix']['rows']:
    key = row['cell_key']
    slug = key.replace(':', '-')
    state = obj(RUN / 'cells' / slug / 'cell.json')
    assert state['cell_key'] == key and len(state['configs']) == 8
    counts, operators, outcomes = Counter(), Counter(), Counter()
    duration, candidate_attempts = 0.0, 0
    cumulative = Counter()
    best = None
    for idx, config in sorted(state['configs'].items(), key=lambda pair: int(pair[0])):
        assert config['status'] == 'completed'
        run_slug = slug + '-' + idx
        run = obj(RUN / 'runs' / run_slug / 'run-result.json', config['result_sha256'])
        intent = obj(RUN / 'intents' / (run_slug + '.json'), config['intent_sha256'])
        assert run['cell_key'] == key and run['config_index'] == int(idx)
        assert run['run_key'] == intent['run_key'] == key + ':' + idx
        assert run['runner_commit'] == manifest['source_commit'] and run['worker_cleanup_verified'] is True
        cost = by_run[run['run_key']]
        assert run['requests'] == {'main': cost['main_requests'], 'debug': cost['debug_requests'],
                                   'charged_tokens': cost['charged_tokens']}
        recounted = count_jobs(run['jobs'])
        assert dict(recounted) == run['job_counts']
        counts.update(recounted)
        operators.update(run['operator_counts'])
        job_ids.extend(j['job_id'] for j in run['jobs'] if j.get('job_id'))
        raw = read(RUN / 'runs' / run_slug / 'generation-outcomes.jsonl', run['generation_outcomes_sha256'])
        outcomes.update(json.loads(line).get('classification', 'unknown') for line in raw.decode().splitlines() if line.strip())
        elapsed = run['finished_at'] - intent['launched_at']
        assert elapsed >= 0
        duration += elapsed
        candidate_attempts += run['main_candidates']
        cumulative.update(cost)
        if config['raw_score'] is not None:
            best = max(best if best is not None else -1, config['raw_score'])
        curves.append({'cell_key': key, 'task_id': row['task_id'], 'method': row['method'],
                       'search_seed': row['search_seed'], 'config_index': int(idx),
                       'cumulative_requests': cumulative['requests'],
                       'cumulative_charged_tokens': cumulative['charged_tokens'],
                       'validation_incumbent': best})
    assert cumulative == by_cell[key]
    all_jobs.update(counts)
    all_operators.update(operators)
    all_outcomes.update(outcomes)
    final_path = RUN / 'final' / slug / 'final.json'
    final_intent = RUN / 'final' / slug / 'intent.json'
    case = {'cell_key': key, 'canonical_final_verified': False, 'arithmetic_accuracy': None,
            'status': 'not_attempted', 'source_and_cleanup_facts_checked': False}
    if final_path.exists():
        final = obj(final_path)
        assert final['cell_key'] == key and final['status'] == 'completed'
        assert final['barrier_identity'] == ledger['final_phase']['barrier_identity']
        obj(final_intent, final['intent_sha256'])
        answer_sha = data['tasks'][row['task_id']]['task_binding']['final_labels_sha256']
        answers = labels(read(ROOT / 'research/comparison/sealed' / row['task_id'] / 'final_labels.csv', answer_sha))
        predictions = labels(read(final_path.parent / 'prediction.csv', final['prediction_sha256']))
        assert set(predictions) == set(answers)
        correct = sum(predictions[k] == v for k, v in answers.items())
        acc = correct / len(answers)
        assert len(answers) == data['tasks'][row['task_id']]['splits']['final']['rows']
        assert all(abs(acc - final[k]) <= 1e-12 for k in ('raw_score', 'recomputed_accuracy', 'utility'))
        source_sha = state['winner']['source_sha256']
        assert source_sha == final['winner_source_sha256']
        read(RUN / 'cells' / slug / 'winner.py', source_sha)
        ev = final['evidence']
        assert ev['source_identity_verified'] and ev['worker_cleanup_verified']
        assert ev['submitted_source_sha256'] == ev['entrypoint_sha256'] == source_sha
        assert ev['answer_sha256'] == final['answer_sha256'] == answer_sha
        assert ev['prediction_sha256'] == final['prediction_sha256']
        assert ev['task_id'] == 'comparison_' + row['task_id'] + '_final'
        assert ev['split'] == 'final' and ev['evaluator'] == 'external_trusted'
        assert ev['metric'] == 'accuracy' and ev['direction'] == 'maximize'
        assert ev['row_count'] == final['row_count'] == len(answers)
        assert all(ev['cleanup'][k] == [] for k in ('live_jobs', 'uncertain_submissions', 'submission_intents'))
        case.update(status='saved_final_arithmetic_checked_but_canonical_audit_failed',
                    arithmetic_accuracy=acc, correct=correct, rows=len(answers),
                    source_and_cleanup_facts_checked=True, prediction_sha256=final['prediction_sha256'],
                    answer_sha256=answer_sha, source_sha256=source_sha)
    elif final_intent.exists():
        case['status'] = 'attempted_integrity_stop_no_accepted_final'
    final_cases.append(case)
    rows.append({**{k: row[k] for k in ('cell_key', 'task_id', 'method', 'search_seed', 'terminal_status',
                    'validation_incumbent', 'winner_config_index')},
                 'configs_completed': 8, 'main_candidate_operator_attempts': candidate_attempts,
                 'canonical_final_verified': False, 'canonical_raw_accuracy': None, 'canonical_utility': None,
                 'forensic_final_status': case['status'], 'forensic_arithmetic_accuracy': case['arithmetic_accuracy'],
                 **{k: by_cell[key][k] for k in ('requests', 'main_requests', 'debug_requests', 'charged_tokens', 'input_tokens', 'output_tokens')},
                 **{'jobs_' + k: counts[k] for k in job_keys},
                 'summed_configuration_wall_seconds': duration,
                 'operator_counts': dict(operators), 'generation_outcomes': dict(outcomes)})
assert len(rows) == 27 and len(curves) == 216
assert len(job_ids) == len(set(job_ids))

# The rejected WDBC prediction remains diagnostic evidence, never an accepted score.
wdbc_key = 'wdbc:11:FIXED'
wdbc = obj(RUN / 'cells/wdbc-11-FIXED/cell.json')
wdbc_sha = wdbc['winner']['source_sha256']
jobs_root = ROOT / '.runtime/rsi-trustworthy/storage/mlsandbox/jobs'
matching = [p.parent.parent for p in jobs_root.glob('*/final/*/code/main.py')
            if hashlib.sha256(p.read_bytes()).hexdigest() == wdbc_sha]
assert len(matching) == 1
wdbc_dir = matching[0]
read(wdbc_dir / 'code/main.py', wdbc_sha)
pred = labels(read(wdbc_dir / 'prediction_snapshot.csv'))
truth = labels(read(ROOT / 'research/comparison/sealed/wdbc/final_labels.csv',
                    data['tasks']['wdbc']['task_binding']['final_labels_sha256']))
assert set(pred) == set(truth)
correct = sum(pred[k] == v for k, v in truth.items())
exact = correct / len(truth)
rounded = round(exact, 6)
assert abs(exact - rounded) > 1e-9
cleanup = obj(RUN / 'final/executor/wdbc-11-FIXED/live-jobs.json')
assert all(cleanup[k] == [] for k in ('live_jobs', 'uncertain_submissions', 'submission_intents'))
diagnosis = {'cell_key': wdbc_key, 'job_id': wdbc_dir.name, 'correct': correct, 'rows': len(truth),
             'exact_csv_accuracy': exact, 'trusted_metric_replayed_round6': rounded,
             'difference': abs(exact-rounded), 'frozen_final_tolerance': 1e-9,
             'source_sha256': wdbc_sha, 'id_sets_equal': True, 'cleanup_sets_empty': True,
             'accepted_final_result': False,
             'scope': 'Original saved code/prediction plus independently replayed six-decimal arithmetic; no candidate rerun.'}
for directory, expected in {
    'acceptance-us009': '59c236e8261dd3565dabca9f756dd806f44625cb38b8b89452e16adefe88ff65',
    'acceptance-us009-round2': 'd8340c359bddf65b39639fdeaf06ca36e4d2237aa97684bfead172a8b61915b2',
}.items():
    read(ROOT / '.runtime' / directory / 'budget-ledger.json', expected)
for relative, sha in hashes.items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == sha
result = {
    'schema': 'istratde-post-stop-forensic.v1', 'source_commit': manifest['source_commit'],
    'canonical_audit_verdict': audit['verdict'], 'canonical_ledger_error': report['ledger_error'],
    'conclusion': 'insufficient-evidence', 'complete_matrix': False,
    'scope': 'Independent read-only reconciliation of original artifacts. Does not repair the ledger or supersede the canonical FAIL audit. Arithmetic observations are diagnostic, not accepted final outcomes.',
    'global_cost': dict(totals), 'configs_completed': 216, 'cells_search_terminal': 27,
    'canonical_accepted_finals': 0, 'saved_finals_arithmetic_checked': sum(c['arithmetic_accuracy'] is not None for c in final_cases),
    'final_intents': 10, 'not_attempted_finals': 17, 'unknown_usage_requests': 0,
    'job_counts': dict(all_jobs), 'operator_counts': dict(all_operators), 'generation_outcomes': dict(all_outcomes),
    'run_supervisor': supervisor,
    'timing': {'supervisor_utc_wall_seconds': supervisor['finished_at']-supervisor['started_at'],
               'supervisor_monotonic_seconds': supervisor['monotonic_elapsed_seconds'],
               'summed_configuration_wall_seconds': sum(r['summed_configuration_wall_seconds'] for r in rows),
               'note': 'UTC wall and monotonic differ; cause not established. Configuration wall includes startup/model/sandbox, not pure GPU compute.'},
    'timestamp_defect': {'state': ledger['final_phase'], 'event': phase_events[0], 'delta_seconds': timestamp_delta},
    'wdbc_diagnosis': diagnosis, 'cells': rows, 'final_cases': final_cases, 'curves': curves,
    'read_only_artifact_hashes': hashes,
    'limitations': ['Strict ledger validation fails; these arithmetic checks are narrower and do not assert full protocol acceptance.',
                   'No full-matrix effect or bootstrap interval is estimable. Unobserved/unverified finals remain null, never zero-filled.',
                   'Canonical cost plots use zero when ledger loading fails; do not interpret them. Supplement curves use reconciled recorded costs.',
                   'Three search seeds per task; any future valid interval remains exploratory and conditional on the three fixed test sets.']}
OUT.mkdir(exist_ok=True)
with (OUT / 'forensic-summary.json').open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
for name, values in [('results-27-cells.csv', rows), ('validation-cost-curves.csv', curves), ('final-recompute.csv', final_cases)]:
    with (OUT / name).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for v in values for k in v}))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k,v in row.items()} for row in values)
print(json.dumps({k: result[k] for k in ('global_cost', 'job_counts', 'operator_counts', 'generation_outcomes',
                                       'saved_finals_arithmetic_checked', 'timing', 'wdbc_diagnosis')}))
