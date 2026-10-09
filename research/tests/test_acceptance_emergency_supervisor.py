"""Emergency cleanup proof and parent interruption; no real process/job."""
import json
import subprocess
import time

import pytest

from research.contracts.budget import BudgetLedger, BudgetLimits
from research.search import acceptance as ap
from research.search import acceptance_config as ac
from research.search.acceptance_inner import AcceptanceError, InnerRunSpec
from research.search.search_vector import decode_retained

JOB_ID = 'job_0123456789abcdef'


def registry():
    return {'schema': 'acceptance-live-jobs.v1', 'run_id': 'cfg0',
        'live_jobs': [], 'uncertain_submissions': [], 'submission_intents': [],
        'updated_at': time.time()}


@pytest.mark.parametrize('field', ['live_jobs', 'uncertain_submissions', 'submission_intents'])
def test_missing_registry_lists_are_unproven_not_empty(tmp_path, field):
    run_dir = tmp_path / 'cfg0'
    run_dir.mkdir()
    payload = registry()
    payload.pop(field)
    (run_dir / 'live-jobs.json').write_text(json.dumps(payload))
    def no_client():
        raise AssertionError('Malformed registry must not issue cancellation calls')
    evidence = ap.cleanup_child_jobs(run_dir, repo_root=tmp_path, client_factory=no_client)
    assert evidence['cleanup_verified'] is False


@pytest.mark.parametrize('failure', ['timeout', 'interrupt'])
def test_default_child_abort_kills_only_child_and_cancels_recorded_job(tmp_path, monkeypatch, failure):
    run_dir = tmp_path / ac.RUNTIME_DIR / 'runs/cfg0'
    run_dir.mkdir(parents=True)
    payload = registry()
    payload['live_jobs'] = [JOB_ID]
    (run_dir / 'live-jobs.json').write_text(json.dumps(payload))
    interpreter = tmp_path / ap.EVO_VENV_PYTHON
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    ledger_path = tmp_path / ac.RUNTIME_DIR / ac.LEDGER_FILENAME
    with BudgetLedger.open(ledger_path, limits=BudgetLimits(max_elapsed_seconds=60)):
        pass
    spec = InnerRunSpec('cfg0', str(ac.RUNTIME_DIR / 'runs/cfg0'),
        decode_retained([0.7, 0.5, 0.3, 0.2, 1]), str(ac.RUNTIME_DIR / ac.LEDGER_FILENAME), 'a' * 40)
    class FakeProcess:
        killed = False
        returncode = None
        def communicate(self, *args, **kwargs):
            if self.killed:
                return '', ''
            if failure == 'timeout':
                raise subprocess.TimeoutExpired('offline fake child', 1)
            raise KeyboardInterrupt()
        def kill(self):
            self.killed = True
            self.returncode = -9
        def poll(self):
            return self.returncode
    process = FakeProcess()
    monkeypatch.setattr(ap.subprocess, 'Popen', lambda *args, **kwargs: process)
    cancelled = []
    class Client:
        def cancel_job(self, job_id):
            cancelled.append(job_id)
            return {'job_id': job_id, 'cleanup_verified': True}
    with pytest.raises((KeyboardInterrupt, subprocess.TimeoutExpired, AcceptanceError)):
        ap._default_child_runner(spec, run_dir / 'spec.json', repo_root=tmp_path,
            cleanup_client_factory=Client)
    assert process.killed is True
    assert cancelled == [JOB_ID]
    evidence = json.loads((run_dir / 'emergency-cleanup.json').read_text())
    assert evidence['cleanup_verified'] is True
