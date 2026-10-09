"""Pure lifecycle mocks; no candidate execution or process sweep on the host."""
import importlib.util
from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('rsi_review_worker', repo / 'deploy/rsi-trustworthy/worker/minimal_worker.py')
assert spec and spec.loader
worker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worker
spec.loader.exec_module(worker)

class FinishedProcess:
    def wait(self):
        return 0
    def poll(self):
        return 0

def test_unproven_cleanup_never_reports_completed():
    config = worker.WorkerConfig(control_key=b'synthetic-only', sweeper=lambda: False)
    registry = worker.SessionRegistry(config)
    session = worker.Session('test', '', None, config, registry)
    session._process = FinishedProcess()
    session._reap()
    assert registry.poisoned
    assert session.snapshot()['status'] not in ('completed', 'terminated'), 'unproven cleanup is reported as terminal success'

def test_delete_timeout_never_frees_slot():
    class PendingSession:
        def kill(self):
            pass
        def poll_done(self, timeout):
            return False
    registry = worker.SessionRegistry(worker.WorkerConfig(control_key=b'synthetic-only'))
    registry._sessions['pending'] = PendingSession()
    registry.remove('pending')
    assert registry.poisoned or 'pending' in registry._sessions, 'DELETE timeout removed the only active-slot record'


def test_reaping_targets_only_candidate_children(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "iter_uid_processes", lambda uid, **kwargs: [101, 102])
    called = []
    worker.reap_candidate_children(65432, proc_root=tmp_path,
                                   waitpid_fn=lambda pid, flags: (called.append(pid) or (0, 0)))
    assert called == [101, 102]
    assert -1 not in called

def test_sweep_reaps_after_last_kill_round(monkeypatch, tmp_path):
    remaining = [101]
    killed = []
    monkeypatch.setattr(worker, "iter_uid_processes", lambda uid, **kwargs: list(remaining))
    def reap():
        if killed:
            remaining.clear()
    clean = worker.sweep_candidate_processes(
        65432, proc_root=tmp_path, kill_fn=lambda pid, sig: killed.append(pid),
        sleep_fn=lambda seconds: None, max_rounds=1, reap_fn=reap,
    )
    assert killed == [101]
    assert clean
