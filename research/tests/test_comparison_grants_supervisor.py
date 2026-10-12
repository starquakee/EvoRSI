"""Supervisor regressions for exact task read grants (EXP-003)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "comparison_supervisor_trampoline",
    ROOT / "deploy/rsi-trustworthy/worker/exec_trampoline.py",
)
assert spec is not None and spec.loader is not None
tramp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tramp)


@pytest.mark.parametrize("path", [
    "/mnt/rsi_data",
    "/mnt/rsi_data/comparison",
    "/mnt/rsi_data/comparison/unregistered/data/public",
    "/mnt/rsi_data/comparison/banknote/data/public/unexpected-child",
])
def test_canonical_path_under_data_root_is_not_authority(path, monkeypatch):
    # Pretend these directories really exist. Rejection must derive from
    # the trusted exact-leaf policy, not a missing-directory accident.
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    with pytest.raises(tramp.TrampolineError):
        tramp.validate_read_grant(path, Path("/mnt/local_sandbox_workdir"))


@pytest.mark.parametrize("grants", [
    ["/mnt/rsi_data"],
    ["/mnt/rsi_data/comparison"],
    ["/mnt/rsi_data/comparison/unregistered/data/public"],
    ["/mnt/rsi_data/comparison/banknote/data/public",
     "/mnt/rsi_data/comparison/wdbc/data/public"],
])
def test_worker_rejects_broad_or_multiple_task_grants(grants, monkeypatch):
    import sys
    worker_spec = importlib.util.spec_from_file_location(
        "comparison_supervisor_worker",
        ROOT / "deploy/rsi-trustworthy/worker/minimal_worker.py",
    )
    assert worker_spec is not None and worker_spec.loader is not None
    worker = importlib.util.module_from_spec(worker_spec)
    sys.modules[worker_spec.name] = worker
    worker_spec.loader.exec_module(worker)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    config = worker.WorkerConfig(control_key=b"supervisor-test-only-control-key")
    accepted, reason = worker.validate_read_grants(grants, config)
    assert accepted is None
    assert reason
