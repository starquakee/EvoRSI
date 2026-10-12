"""Durable-write failure must stop the original final-phase ledger."""
import hashlib
from pathlib import Path
from typing import Any

import pytest

from research.comparison import data, final
from research.comparison.budget import ComparisonLedger
from research.comparison.protocol import identity_of
from research.tests.test_comparison_final import SyntheticExecutor, _make_repo_orch, _publish_kwargs, _synthetic_repo
from research.tests.test_comparison_final_supervisor import fake_barrier


@pytest.mark.parametrize("boundary", ["publication_payload", "publication_record", "prediction_bytes"])
def test_final_write_failure_stops_original_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str) -> None:
    repo = _synthetic_repo(tmp_path)
    orch = _make_repo_orch(tmp_path, monkeypatch)
    barrier = fake_barrier(orch)
    key = orch.cells[0].cell_key()
    source = b"print('synthetic final write failure')\n"
    sha = hashlib.sha256(source).hexdigest()
    barrier["cells"][key] = {"terminal_status": "frozen", "winner": {
        "source_sha256": sha, "code_sha256": sha, "config_index": 0,
        "validation_score": 0.5, "job_id": "synthetic-validation-job", "prediction_sha256": "d" * 64,
    }}
    barrier.pop("barrier_identity")
    barrier["barrier_identity"] = identity_of(barrier)
    monkeypatch.setattr(orch, "verify_final_barrier", lambda ledger: barrier)
    (orch.runtime_dir / "cells" / key.replace(":", "-") / "winner.py").write_bytes(source)
    reached: list[str] = []
    def fail_write(*args: Any, **kwargs: Any) -> None:
        reached.append(boundary)
        raise OSError("synthetic durable write failure")
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        if boundary.startswith("publication"):
            if boundary == "publication_payload":
                monkeypatch.setattr(data, "_write_frozen", fail_write)
            else:
                monkeypatch.setattr(final, "_write_durable", fail_write)
            with pytest.raises((OSError, final.FinalPhaseError)):
                final.publish_final_data(orch, ledger, barrier, now_fn=lambda: 1000.0, **_publish_kwargs(repo))
        else:
            publication = final.publish_final_data(orch, ledger, barrier, now_fn=lambda: 1000.0, **_publish_kwargs(repo))
            executor = SyntheticExecutor(repo["sealed"], registry_version=publication["registry_version"])
            monkeypatch.setattr(final, "_write_durable_bytes", fail_write)
            with pytest.raises((OSError, final.FinalPhaseError)):
                final.run_final_phase(orch, ledger, barrier, executor, now_fn=lambda: 1000.0, sealed_root=repo["sealed"])
            assert executor.calls == [key]
        assert reached == [boundary], "must reach the intended write boundary"
        assert ledger.stopped, "write failure must durably stop the actual original ledger"
