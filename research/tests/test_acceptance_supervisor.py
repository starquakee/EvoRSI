"""Independent US009 contract regressions; no model or sandbox execution.

Supervisor integration findings beyond these regressions are appended at the
END of progress.txt. Read all of them before preparation commit: real request
tracing, native ask reproducibility, OAuth refresh, original deadline and
emergency job cancellation, artifact-bound cache and inner RNG seeding.
These tests are intentionally narrower than the complete acceptance review.
"""
import json
from types import SimpleNamespace

import pytest

from research.adapters.sandbox_eval_client import SandboxResult
from research.contracts.budget import BudgetError, CancellationToken
from research.contracts.source_gate import check_source_tree, load_policy
from research.search.credentials import CredentialError, ManagedKimiCredentials
from research.search.run_index import RunIdentity, RunIndex, RunIndexError
from research.search.sandbox_task import SandboxAcceptanceTask, VALID_SOLUTION


def actual_shape(tmp_path):
    source = "print('candidate')\n"
    folder = tmp_path / 'source'
    folder.mkdir()
    (folder / 'main.py').write_text(source)
    policy = load_policy()
    gate = check_source_tree(folder, policy).to_dict()
    import hashlib
    gate.update(entrypoint='main.py', entrypoint_sha256=hashlib.sha256(source.encode()).hexdigest())
    raw = {
        'job_id': 'job_supervisor_fixture', 'status': 'completed',
        'created_at': '2026-10-08T14:45:38.750352Z',
        'started_at': '2026-10-08T14:45:38.772777Z',
        'completed_at': '2026-10-08T14:45:40.480165Z',
        'result': {
            'score': 0.55, 'result': 'success', 'source_gate': gate,
            'source_identity_verified': True, 'worker_cleanup_verified': True,
            'evaluation': {
                'score': 0.55, 'split': 'test', 'metric': 'accuracy',
                'task_id': 'hello_synth', 'direction': 'maximize',
                'evaluator': 'external_trusted', 'row_count': 40,
                'answer_sha256': 'f25ab07cfc53a420662e3ef899df728de124136c70277b66236ca06f08638579',
                'registry_version': 'evaluator-registry.v1',
                'prediction_sha256': 'aeff9e843b62f64a93b97a6e1921abab9d82b41de8078bca0a900cfa2845ff5c',
                'prediction_snapshot': '/mnt/rsi_storage/mlsandbox/jobs/2026-10-08/default/job_supervisor_fixture/prediction_snapshot.csv',
            },
        },
    }
    return source, raw


def translate(source, raw):
    response = SandboxResult(raw['job_id'], raw['status'], 0.55, raw)
    client = SimpleNamespace(require_cleanup=True, submit_and_wait=lambda *a, **k: response)
    task = SandboxAcceptanceTask(client, CancellationToken())
    return task.step_task({}, source)[1]


def test_matching_complete_real_api_shape_is_valid(tmp_path):
    source, raw = actual_shape(tmp_path)
    assert translate(source, raw)[VALID_SOLUTION] is True


@pytest.mark.parametrize('path,value', [
    (('result', 'source_identity_verified'), False),
    (('result', 'worker_cleanup_verified'), False),
    (('completed_at',), None),
    (('result', 'source_gate'), None),
    (('result', 'source_gate', 'allowed'), False),
    (('result', 'source_gate', 'entrypoint_sha256'), '0' * 64),
    (('result', 'source_gate', 'source_sha256'), '0' * 64),
    (('result', 'source_gate', 'policy_version'), 'unreviewed-policy'),
    (('result', 'source_gate', 'policy_sha256'), '0' * 64),
    (('result', 'evaluation'), {}),
    (('result', 'evaluation', 'evaluator'), 'candidate_claim'),
    (('result', 'evaluation', 'metric'), 'logloss'),
    (('result', 'evaluation', 'task_id'), 'other-task'),
    (('result', 'evaluation', 'direction'), 'minimize'),
    (('result', 'evaluation', 'score'), 0.9),
    (('result', 'evaluation', 'prediction_sha256'), ''),
    (('result', 'evaluation', 'answer_sha256'), '0' * 64),
    (('result', 'evaluation', 'row_count'), 1),
    (('result', 'evaluation', 'registry_version'), 'other-registry'),
    (('result', 'result'), 'scoring_failed'),
])
def test_unproven_or_mismatched_evidence_never_scores(tmp_path, path, value):
    source, raw = actual_shape(tmp_path)
    parent = raw
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    try:
        translated = translate(source, raw)
    except (BudgetError, ValueError):
        return
    assert translated[VALID_SOLUTION] is False


@pytest.mark.parametrize('expires', [float('nan'), float('inf'), float('-inf')])
def test_credentials_reject_nonfinite_expiry(tmp_path, expires):
    path = tmp_path / 'credentials.json'
    path.write_text(json.dumps({'access_token': 'SYNTHETIC_TEST_SENTINEL', 'expires_at': expires}))
    with pytest.raises(CredentialError):
        ManagedKimiCredentials(path, now_fn=lambda: 100).access_token()


def test_loaded_failed_run_is_not_a_cache_hit(tmp_path):
    identity = RunIdentity(42, 'k3', 'hello_synth', 'accuracy', 'source-gate.v2', 'c' * 64, 'd' * 40)
    record = {
        'identity': identity.to_dict(), 'status': 'failed', 'score': None,
        'job_ids': ['job_fixture'], 'operator_trace_sha256': 'a' * 64,
        'worker_cleanup_verified': True, 'operator_counts': {'draft': 1},
        'termination': 'BudgetExhausted', 'created_at': 1.0,
    }
    path = tmp_path / 'index.json'
    path.write_text(json.dumps({'schema': 'acceptance-run-index.v1', 'records': {identity.key(): record}}))
    with pytest.raises(RunIndexError):
        RunIndex(path).get(identity)


@pytest.mark.parametrize('run_dir,ledger_path', [
    ('.runtime/other/runs/cfg0', '.runtime/acceptance-us009/budget-ledger.json'),
    ('.runtime/acceptance-us009/runs/cfg0', '.runtime/other/budget-ledger.json'),
])
def test_inner_spec_cannot_select_another_acceptance_location(run_dir, ledger_path):
    from research.search.acceptance_inner import AcceptanceError, InnerRunSpec
    from research.search.search_vector import decode_retained
    spec = InnerRunSpec('cfg0', run_dir, decode_retained([0.7, 0.5, 0.3, 0.2, 1]), ledger_path, 'a' * 40)
    with pytest.raises(AcceptanceError):
        spec.validate()


def test_untracked_implementation_is_not_a_reviewed_clean_tree(tmp_path):
    import subprocess
    from research.search.acceptance_config import git_tree_dirty
    subprocess.run(['git', 'init', str(tmp_path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (tmp_path / 'new_acceptance_module.py').write_text('VALUE = 1\n')
    assert git_tree_dirty(tmp_path) is True


def test_later_failed_inner_never_produces_complete_verdict(tmp_path, monkeypatch):
    from research.tests.test_acceptance_controller import _repo, _fake_child, FakeAlgo, VECTORS
    from research.search.acceptance import AcceptanceError, run_acceptance
    repo, runtime = _repo(tmp_path, monkeypatch)
    calls = []
    try:
        summary = run_acceptance(repo, runtime, algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=_fake_child(runtime, calls, behaviors={'cfg2': {'status': 'failed'}}))
    except (AcceptanceError, BudgetError, RunIndexError):
        return
    assert summary['verdict'] != 'complete'


@pytest.mark.parametrize('failure_kind', ['timeout', 'interrupt'])
def test_child_abort_stops_ledger_and_checkpoints(tmp_path, monkeypatch, failure_kind):
    import subprocess
    from research.contracts.budget import BudgetLedger
    from research.tests.test_acceptance_controller import _repo, FakeAlgo, VECTORS
    from research.search.acceptance import AcceptanceError, run_acceptance
    repo, runtime = _repo(tmp_path, monkeypatch)
    def child(*args, **kwargs):
        if failure_kind == 'timeout':
            raise subprocess.TimeoutExpired('offline-test-child', 1)
        raise KeyboardInterrupt()
    with pytest.raises((subprocess.TimeoutExpired, KeyboardInterrupt, AcceptanceError, BudgetError)):
        run_acceptance(repo, runtime, algo_factory=lambda: FakeAlgo(VECTORS), child_runner=child)
    assert (runtime / 'acceptance-checkpoint.json').exists()
    with BudgetLedger.open(runtime / 'budget-ledger.json') as ledger:
        assert ledger.snapshot()['stopped'] is not None


def test_unbound_completed_record_is_not_a_cache_hit(tmp_path):
    identity = RunIdentity(42, 'k3', 'hello_synth', 'accuracy', 'source-gate.v2', 'c' * 64, 'd' * 40)
    record = {
        'identity': identity.to_dict(), 'status': 'completed', 'score': 0.55,
        'job_ids': ['job_fixture'], 'operator_trace_sha256': 'a' * 64,
        'worker_cleanup_verified': True,
        'operator_counts': {'draft': 2, 'improve': 2, 'crossover': 2, 'debug': 0},
        'termination': 'completed', 'created_at': 1.0,
    }
    # Digest-shaped strings alone do not prove that source/trace/results
    # exist or still match. No bound files are supplied here.
    path = tmp_path / 'index.json'
    path.write_text(json.dumps({'schema': 'acceptance-run-index.v1', 'records': {identity.key(): record}}))
    with pytest.raises(RunIndexError):
        RunIndex(path).get(identity)


def test_trace_observer_failure_stops_before_provider_and_prevents_retry(tmp_path):
    from research.contracts.budget import BudgetLedger
    from research.search.budget_transport import RequestGuard, TransportGuardConfig
    calls = []
    def broken_sink(*args):
        raise OSError('synthetic trace writer failure')
    with BudgetLedger.open(tmp_path / 'ledger.json') as ledger:
        guard = RequestGuard(ledger, CancellationToken(), TransportGuardConfig(100, 1, attempt_sink=broken_sink))
        with pytest.raises(OSError):
            guard.run(lambda: calls.append(True), messages=[{'role': 'user', 'content': 'test'}], model_kwargs={})
        assert calls == []
        assert ledger.snapshot()['stopped'] is not None
        with pytest.raises(BudgetError):
            ledger.reserve(1, 1)


def test_request_trace_observes_actual_bounded_timeout(tmp_path):
    from research.contracts.budget import BudgetLedger
    from research.search.budget_transport import RequestGuard, TransportGuardConfig
    seen = {}
    kwargs = {'request_timeout': 999.0}
    def sink(reservation, messages, model_kwargs):
        seen['recorded_timeout'] = model_kwargs['request_timeout']
    def provider():
        seen['sent_timeout'] = kwargs['request_timeout']
        return {'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}
    with BudgetLedger.open(tmp_path / 'ledger.json') as ledger:
        guard = RequestGuard(ledger, CancellationToken(), TransportGuardConfig(100, 1, request_deadline_seconds=5.0, attempt_sink=sink))
        guard.run(provider, messages=[{'role': 'user', 'content': 'test'}], model_kwargs=kwargs)
    assert seen['recorded_timeout'] == seen['sent_timeout']
    assert 0 < seen['sent_timeout'] <= 5.0


@pytest.mark.parametrize('artifact', ['operator-trace.jsonl', 'run-result.json'])
def test_cache_hit_revalidates_bound_artifact_contents(tmp_path, artifact):
    from research.tests.test_acceptance_run_index import _record
    record = _record(tmp_path)
    index = RunIndex(tmp_path / 'index.json')
    index.put(record)
    (tmp_path / record.run_dir / artifact).write_text('tampered after indexing')
    with pytest.raises(RunIndexError):
        index.get(record.identity)


def test_native_preflight_does_not_require_supervisor_temporary_vectors(tmp_path, monkeypatch):
    from research.search import acceptance as ap
    from research.tests.test_acceptance_controller import VECTORS
    monkeypatch.setattr(ap, '_native_istratde_vectors', lambda root: VECTORS)
    result = ap.check_preflight_vectors(tmp_path)
    assert isinstance(result, dict)


def test_outer_can_derive_vectors_without_prior_supervisor_file(tmp_path, monkeypatch):
    from research.tests.test_acceptance_controller import _repo, _fake_child, FakeAlgo, VECTORS
    from research.search import acceptance as ap
    from research.search import acceptance_config as ac
    repo, runtime = _repo(tmp_path, monkeypatch)
    (repo / ac.PREFLIGHT_VECTORS_FILE).unlink()
    summary = ap.run_acceptance(repo, runtime, algo_factory=lambda: FakeAlgo(VECTORS),
        child_runner=_fake_child(runtime, []))
    assert summary['verdict'] == 'complete'


@pytest.mark.parametrize('final_budget_state', ['stopped', 'deadline_exceeded'])
def test_final_verdict_requires_healthy_within_deadline_ledger(tmp_path, monkeypatch, final_budget_state):
    from research.contracts.budget import BudgetLedger
    from research.tests.test_acceptance_controller import _repo, _fake_child, FakeAlgo, VECTORS
    from research.search import acceptance as ap
    from research.search.acceptance_inner import AcceptanceError
    clock = [1000.0]
    original_open = BudgetLedger.open
    def open_with_clock(*args, **kwargs):
        kwargs['now_fn'] = lambda: clock[0]
        return original_open(*args, **kwargs)
    monkeypatch.setattr(BudgetLedger, 'open', staticmethod(open_with_clock))
    repo, runtime = _repo(tmp_path, monkeypatch)
    base_child = _fake_child(runtime, [])
    def child(spec, spec_path, *, repo_root):
        result = base_child(spec, spec_path, repo_root=repo_root)
        if spec.run_id == 'cfg3':
            if final_budget_state == 'stopped':
                with BudgetLedger.open(runtime / 'budget-ledger.json') as ledger:
                    ledger.stop('synthetic post-child stop')
            else:
                clock[0] = 6401.0  # original 1000 + 5400 deadline has passed
        return result
    try:
        summary = ap.run_acceptance(repo, runtime, algo_factory=lambda: FakeAlgo(VECTORS), child_runner=child)
    except (AcceptanceError, BudgetError, RunIndexError):
        return
    assert summary['verdict'] != 'complete'


def test_summary_records_verified_zero_incremental_cache_cost(tmp_path, monkeypatch):
    from research.tests.test_acceptance_controller import _repo, _fake_child, FakeAlgo, VECTORS
    from research.search import acceptance as ap
    repo, runtime = _repo(tmp_path, monkeypatch)
    summary = ap.run_acceptance(repo, runtime, algo_factory=lambda: FakeAlgo(VECTORS),
        child_runner=_fake_child(runtime, []))
    assert summary['inner_evaluations'] == {
        'requested': 5,
        'fresh_child_invocations': 4,
        'model_active_runs': 4,
        'completed_runs': 4,
        'cached': 1,
    }
    assert len(summary['cache_events']) == 1
    hit = summary['cache_events'][0]
    assert hit['outer_position'] == 0
    assert hit['incremental_model_cost'] == {'requests': 0, 'tokens': 0}
    assert hit['verified_committed_before'] == hit['verified_committed_after']
