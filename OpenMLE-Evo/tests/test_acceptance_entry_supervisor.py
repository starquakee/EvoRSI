"""Production run_inner integration; external model/HTTP responses are fake.

No live review file, acceptance ledger, credentials, model request, sandbox
job, candidate execution or Docker action. All generated files use tmp_path.
"""
import hashlib
import json
from pathlib import Path
import time

import httpx
import pytest

from research.contracts.budget import BudgetLedger
from research.contracts.source_gate import check_source_bytes, load_policy
from research.search import acceptance_config as ac
from research.search import acceptance_inner as ai
from research.search.credentials import ManagedKimiCredentials
from test_acceptance_inner import _Stream, DECODED

ROOT = Path(__file__).resolve().parents[2]
SENTINEL = 'ONLY_SYNTHETIC_MODEL_TOKEN_FOR_OFFLINE_ENTRY'
PUBLIC_MARKER = 'OFFLINE_PUBLIC_REQUEST_PROVENANCE_MARKER'


@pytest.mark.parametrize('provider_failure,service_failure', [(False, False), (True, False), (False, True)])
def test_production_inner_entry_preserves_request_job_and_budget_evidence(tmp_path, monkeypatch, provider_failure, service_failure):
    # Read-only source/config link, with all run data under a temporary root.
    (tmp_path / 'OpenMLE-Evo').symlink_to(ROOT / 'OpenMLE-Evo', target_is_directory=True)
    (tmp_path / 'research').symlink_to(ROOT / 'research', target_is_directory=True)
    (tmp_path / 'tasks').symlink_to(ROOT / 'tasks', target_is_directory=True)
    monkeypatch.setenv('LOGGING_DIR', str(tmp_path / 'logs'))
    monkeypatch.setattr(ai, 'check_runner_review', lambda root: None)
    monkeypatch.setattr(ac, 'git_head_commit', lambda root: 'c' * 40)
    monkeypatch.setattr(ai, 'PUBLIC_USER_PROMPT', ai.PUBLIC_USER_PROMPT + ' ' + PUBLIC_MARKER)
    auth = tmp_path / ac.AUTH_FILE
    auth.parent.mkdir(parents=True)
    auth.write_text('SANDBOX_API_KEYS=ONLY_SYNTHETIC_SANDBOX_KEY\n')
    auth.chmod(0o600)
    credentials_path = tmp_path / 'fake-credentials.json'
    credentials_path.write_text(json.dumps({'access_token': SENTINEL, 'expires_at': time.time() + 3600}))
    provider = ManagedKimiCredentials(credentials_path)
    monkeypatch.setattr(ai, 'ManagedKimiCredentials', lambda **kwargs: provider)

    calls = []
    def completion(**kwargs):
        calls.append(kwargs)
        assert kwargs['stream'] is True
        assert kwargs['api_key'] == SENTINEL
        assert any(PUBLIC_MARKER in m.get('content', '') for m in kwargs['messages'])
        if provider_failure:
            raise RuntimeError('offline unknown-usage provider failure')
        content = "A compact plan.\n```python\nprint('offline candidate %d')\n```\n" % len(calls)
        return _Stream(content, {'prompt_tokens': 10, 'completion_tokens': 20}, calls)
    monkeypatch.setattr('dojo.core.solvers.llm_helpers.backends.lite_llm.completion_fn', completion)

    jobs = {}
    prediction = (ROOT / 'tasks/hello_synth/data/public/sample_submission.csv').read_bytes()
    def service(request):
        if request.method == 'POST' and request.url.path == '/api/v1/jobs':
            payload = json.loads(request.content)
            is_failure = service_failure and not jobs
            job_id = 'job_%016x' % (len(jobs) + 1)
            source = payload['code'].encode()
            raw_hash = hashlib.sha256(source).hexdigest()
            tree_hash = hashlib.sha256(b'main.py\0' + raw_hash.encode() + b'\n').hexdigest()
            gate = check_source_bytes(source, load_policy()).to_dict()
            gate.update(source_sha256=tree_hash, entrypoint='main.py', entrypoint_sha256=raw_hash)
            storage_rel = Path('mlsandbox/jobs/2026-10-08/default') / job_id
            host_storage = tmp_path / '.runtime/rsi-trustworthy/storage' / storage_rel
            (host_storage / 'code').mkdir(parents=True)
            (host_storage / 'code/main.py').write_bytes(source)
            if not is_failure:
                (host_storage / 'prediction_snapshot.csv').write_bytes(prediction)
            container_storage = '/mnt/rsi_storage/' + storage_rel.as_posix()
            jobs[job_id] = {
                'job_id': job_id, 'status': 'completed',
                'created_at': '2026-10-08T14:45:38.750352Z',
                'started_at': '2026-10-08T14:45:38.772777Z',
                'completed_at': '2026-10-08T14:45:40.480165Z',
                'result': {
                    'score': 0.55, 'result': 'success', 'source_gate': gate,
                    'source_identity_verified': True, 'worker_cleanup_verified': True,
                    'artifact_policy': 'local_scratch', 'nfs_job_storage_dir': container_storage,
                    'evaluation': {
                        'score': 0.55, 'split': 'test', 'metric': 'accuracy',
                        'task_id': 'hello_synth', 'direction': 'maximize',
                        'evaluator': 'external_trusted', 'row_count': 40,
                        'answer_sha256': 'f25ab07cfc53a420662e3ef899df728de124136c70277b66236ca06f08638579',
                        'registry_version': 'evaluator-registry.v1',
                        'prediction_sha256': hashlib.sha256(prediction).hexdigest(),
                        'prediction_snapshot': container_storage + '/prediction_snapshot.csv',
                    },
                },
            }
            if is_failure:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['result'].update(score=None, result='code_error', evaluation={})
            return httpx.Response(200, json={'job_id': job_id})
        if request.method == 'GET':
            return httpx.Response(200, json=jobs[request.url.path.split('/')[-1]])
        raise AssertionError('Unexpected offline service request: ' + request.method)
    original_http_client = httpx.Client
    def fake_http_client(*args, **kwargs):
        kwargs['transport'] = httpx.MockTransport(service)
        return original_http_client(*args, **kwargs)
    monkeypatch.setattr(httpx, 'Client', fake_http_client)

    spec = ai.InnerRunSpec('cfg0', str(ac.RUNTIME_DIR / 'runs/cfg0'), dict(DECODED),
        str(ac.RUNTIME_DIR / ac.LEDGER_FILENAME), 'c' * 40)
    spec_path = tmp_path / 'spec.json'
    spec_path.write_text(json.dumps(spec.to_dict()))
    ledger_path = tmp_path / spec.ledger_path
    with BudgetLedger.open(ledger_path, limits=ac.acceptance_budget_limits()):
        pass
    exit_code = ai.run_inner(spec_path, repo_root=tmp_path)
    if not provider_failure:
        assert exit_code == 0
    run_dir = tmp_path / spec.run_dir
    result = json.loads((run_dir / 'run-result.json').read_text())
    expected_calls = 1 if provider_failure else (7 if service_failure else 6)
    expected_jobs = 0 if provider_failure else expected_calls
    assert result['status'] == ('failed' if provider_failure else 'completed'), result.get('termination')
    assert result['best_score'] == (None if provider_failure else 0.55)
    assert len(calls) == expected_calls
    assert len(jobs) == len(result['job_ids']) == expected_jobs
    assert result['model_usage']['requests_delta'] == expected_calls
    if not provider_failure:
        assert result['model_usage']['tokens_delta'] == 30 * expected_calls
    raw_ledger = json.loads(ledger_path.read_text())
    reservations = [event['reservation_id'] for event in raw_ledger['events'] if event['type'] == 'reserve']
    assert len(reservations) == expected_calls
    if provider_failure:
        assert raw_ledger['stopped'] is not None
    artifacts = [p.read_text(errors='replace') for p in run_dir.rglob('*') if p.is_file()]
    combined = '\n'.join(artifacts)
    assert SENTINEL not in combined
    assert 'ONLY_SYNTHETIC_SANDBOX_KEY' not in combined
    # A real request trace must preserve its public request and connect each
    # actual call to the corresponding persistent budget reservation.
    assert PUBLIC_MARKER in combined, 'Production entry drops the real public request trace'
    for reservation_id in reservations:
        assert reservation_id in combined, 'Provider attempt is not linked to the ledger reservation'

    if not provider_failure:
        # Exercise the actual outer record/cache bridge, not a fabricated
        # run-result fixture. This also covers tied best scores in real Evo.
        from research.search import acceptance as ap
        from research.search.run_index import RunIndex, RunIndexError
        result_bytes = (run_dir / 'run-result.json').read_bytes()
        record = ap._record_from_result(result, spec,
            run_result_sha256=hashlib.sha256(result_bytes).hexdigest(), repo_root=tmp_path)
        index = RunIndex(tmp_path / ac.RUNTIME_DIR / ac.RUN_INDEX_FILENAME)
        index.put(record)
        assert index.get(record.identity).score == 0.55
        # Tamper with ONLY synthetic prediction artifacts in this test's
        # run/storage roots. Either immutable copies or original snapshots
        # may be bound, but at least one actual byte artifact must be checked.
        changed = []
        for artifact_root in (run_dir, tmp_path / '.runtime/rsi-trustworthy/storage'):
            for artifact in artifact_root.rglob('*'):
                if artifact.is_file() and artifact.read_bytes() == prediction:
                    assert artifact.resolve().is_relative_to(tmp_path.resolve())
                    artifact.write_bytes(b'synthetic prediction tampering')
                    changed.append(artifact)
        assert changed
        with pytest.raises(RunIndexError):
            index.get(record.identity)
        assert len(calls) == expected_calls
