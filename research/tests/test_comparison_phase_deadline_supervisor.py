"""Original deadline and durable failure sink cover phase work, not just entry."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research.comparison import phase


@pytest.mark.parametrize('initial_phase', ('search', 'final'))
def test_phase_deadline_expires_during_activation(
    initial_phase: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = [10.0]
    calls: list[str] = []
    stopped: list[str] = []
    phase_reads = [initial_phase, 'final']
    monkeypatch.setattr(phase, 'verify_idle', lambda *a, **k: None)
    monkeypatch.setattr(phase, 'current_phase', lambda *a, **k: phase_reads.pop(0))
    monkeypatch.setattr(phase, '_wait_dispatcher_running', lambda *a, **k: None)

    def recreate(*args: object, **kwargs: object) -> None:
        calls.append('recreate')
        clock[0] = 30.0

    def registry_hash(*args: object, **kwargs: object) -> str:
        calls.append('registry_hash')
        clock[0] = 30.0
        return 'a' * 64

    monkeypatch.setattr(phase, '_compose_recreate_dispatcher', recreate)
    monkeypatch.setattr(phase, '_container_registry_sha256', registry_hash)
    with pytest.raises(phase.PhaseActivationError):
        phase.activate_final_phase(
            tmp_path, expected_registry_sha256='a' * 64,
            deadline_epoch=20.0, now_fn=lambda: clock[0], on_failure=stopped.append,
        )
    assert calls, 'the activation must actually reach the slow phase operation'
    assert stopped, 'expiry during activation must stop the original ledger'


def test_phase_subprocess_timeout_reaches_failure_sink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reached: list[bool] = []
    stopped: list[str] = []
    monkeypatch.setattr(phase, 'verify_idle', lambda *a, **k: None)
    monkeypatch.setattr(phase, 'current_phase', lambda *a, **k: 'search')

    def timeout(*args: object, **kwargs: object) -> None:
        reached.append(True)
        raise subprocess.TimeoutExpired('docker compose', 1.0)

    monkeypatch.setattr(phase, '_compose_recreate_dispatcher', timeout)
    with pytest.raises((phase.PhaseActivationError, subprocess.TimeoutExpired)):
        phase.activate_final_phase(
            tmp_path, expected_registry_sha256='a' * 64,
            deadline_epoch=20.0, now_fn=lambda: 10.0, on_failure=stopped.append,
        )
    assert reached == [True]
    assert stopped, 'a real subprocess timeout must durably stop the original ledger'


def test_real_compose_wait_is_capped_by_original_remaining_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    phases = ['search', 'final']
    waits: list[float] = []
    monkeypatch.setattr(phase, 'verify_idle', lambda *a, **k: None)
    monkeypatch.setattr(phase, 'current_phase', lambda *a, **k: phases.pop(0))
    monkeypatch.setattr(phase, '_wait_dispatcher_running', lambda *a, **k: None)
    monkeypatch.setattr(phase, '_container_registry_sha256', lambda *a, **k: 'a' * 64)

    def fake_subprocess(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert argv[:2] == ['docker', 'compose']
        timeout = kwargs['timeout']
        assert isinstance(timeout, (float, int))
        waits.append(float(timeout))
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(phase.subprocess, 'run', fake_subprocess)
    phase.activate_final_phase(
        tmp_path, expected_registry_sha256='a' * 64,
        deadline_epoch=1001.5, now_fn=lambda: 1000.0,
        on_failure=lambda reason: pytest.fail(reason),
    )
    assert len(waits) == 1, 'the REAL compose wrapper must reach its subprocess call'
    assert 0.0 < waits[0] <= 1.5, 'the original deadline must bound the actual blocking wait'


def test_real_registry_hash_wait_is_capped_in_already_final_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    waits: list[float] = []
    monkeypatch.setattr(phase, 'verify_idle', lambda *a, **k: None)
    monkeypatch.setattr(phase, 'current_phase', lambda *a, **k: 'final')

    def fake_subprocess(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert argv[:4] == ['docker', 'exec', 'rsi_dispatcher', 'sha256sum']
        timeout = kwargs['timeout']
        assert isinstance(timeout, (float, int))
        waits.append(float(timeout))
        return subprocess.CompletedProcess(argv, 0, 'a' * 64 + '  registry.final.v1.json\n', '')

    monkeypatch.setattr(phase.subprocess, 'run', fake_subprocess)
    phase.activate_final_phase(
        tmp_path, expected_registry_sha256='a' * 64,
        deadline_epoch=1001.5, now_fn=lambda: 1000.0,
        on_failure=lambda reason: pytest.fail(reason),
    )
    assert len(waits) == 1, 'the REAL registry hash wrapper must reach its subprocess call'
    assert 0.0 < waits[0] <= 1.5, 'already-final verification must obey the original deadline'
