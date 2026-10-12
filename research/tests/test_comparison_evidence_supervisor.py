"""Independent evidence-binding regressions; no runtime imports or network."""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pytest
from research.comparison.budget import ComparisonLedger
from research.comparison.runner import ComparisonRunnerError, bind_comparison_prediction_snapshots, _bind_incumbent
from research.search import acceptance_config as ac
from research.contracts.source_gate import check_source_bytes, load_policy
from research.tests.test_comparison_budget_supervisor import roster


@pytest.mark.parametrize("case", ["missing_file", "other_job_file", "destination_not_directory"])
def test_snapshot_failure_stops_ledger_and_never_borrows_another_job(tmp_path: Path, case: str) -> None:
    job_id = "job_0000000000000001"
    other = "job_0000000000000002" if case == "other_job_file" else job_id
    payload = b"id,label\n0,0\n"
    host = tmp_path/ac.DISPATCHER_STORAGE_ROOT/"mlsandbox"/"jobs"/other/"prediction_snapshot.csv"
    if case != "missing_file":
        host.parent.mkdir(parents=True)
        host.write_bytes(payload)
    jobs = [{"job_id": job_id, "evidence_verified": True,
             "evaluation": {"prediction_snapshot": f"{ac.CONTAINER_STORAGE_PREFIX}mlsandbox/jobs/{other}/prediction_snapshot.csv",
                            "prediction_sha256": hashlib.sha256(payload).hexdigest()}}]
    run_dir = tmp_path/"private-run"
    run_dir.mkdir()
    if case == "destination_not_directory":
        (run_dir/"predictions").write_text("invalid destination component")
    with ComparisonLedger.open(tmp_path/"ledger.json", cells=roster()) as ledger:
        ledger.start_clock()
        with pytest.raises(ComparisonRunnerError):
            bind_comparison_prediction_snapshots(jobs, run_dir, tmp_path, ledger)
        assert ledger.snapshot()["stopped"] is not None
        assert not list(run_dir.rglob("*.csv"))


def winner_fixture() -> tuple[Any, list[dict[str, Any]], dict[str, str]]:
    code = "print('offline fixture')\n"
    digest = hashlib.sha256(code.encode()).hexdigest()
    tree = hashlib.sha256(b"main.py\0"+digest.encode()+b"\n").hexdigest()
    gate = check_source_bytes(code.encode(), load_policy()).to_dict()
    gate.update(entrypoint="main.py", entrypoint_sha256=digest, source_sha256=tree)
    node = SimpleNamespace(id="node-1", code=code, operators_used=["draft"],
                           metric=SimpleNamespace(value=.4, info={"job_id": "job_0000000000000001"}))
    job = {"job_id": "job_0000000000000001", "status": "completed", "score": .4,
           "evidence_verified": True, "source_identity_verified": True,
           "worker_cleanup_verified": True, "source_gate": gate,
           "submitted_source_sha256": digest,
           "evaluation": {"score": .4, "prediction_sha256": "d"*64}}
    return node, [job], {str(job["job_id"]): "d"*64}


def test_winner_valid_binding_remains_eligible(tmp_path: Path) -> None:
    node, jobs, predictions = winner_fixture()
    with ComparisonLedger.open(tmp_path/"ledger.json", cells=roster()) as ledger:
        ledger.start_clock()
        bound = _bind_incumbent(node, jobs, predictions, ledger)
        assert bound is not None and bound["validation_score"] == .4
        assert ledger.snapshot()["stopped"] is None


@pytest.mark.parametrize("case", ["score_mismatch", "executed_hash_mismatch"])
def test_winner_cannot_detach_from_its_evaluated_score_or_executed_hash(tmp_path: Path, case: str) -> None:
    node, jobs, predictions = winner_fixture()
    if case == "score_mismatch":
        node.metric.value = .9
    else:
        jobs[0]["source_gate"]["entrypoint_sha256"] = "f"*64
    with ComparisonLedger.open(tmp_path/"ledger.json", cells=roster()) as ledger:
        ledger.start_clock()
        assert _bind_incumbent(node, jobs, predictions, ledger) is None
        assert ledger.snapshot()["stopped"] is not None


def test_actual_dispatcher_snapshot_path_is_accepted_for_its_own_job(tmp_path: Path) -> None:
    # compose sets STORAGE_PATH=/mnt/rsi_storage/mlsandbox and the actual
    # dispatcher uses JOB_BASE_PATH=STORAGE_PATH/jobs, then job_id.
    job_id = "job_0000000000000001"
    payload = b"id,label\n0,0\n"
    digest = hashlib.sha256(payload).hexdigest()
    host = tmp_path / ac.DISPATCHER_STORAGE_ROOT / "mlsandbox" / "jobs" / job_id / "prediction_snapshot.csv"
    host.parent.mkdir(parents=True)
    host.write_bytes(payload)
    jobs = [{"job_id": job_id, "evidence_verified": True,
             "evaluation": {"prediction_snapshot": f"/mnt/rsi_storage/mlsandbox/jobs/{job_id}/prediction_snapshot.csv",
                            "prediction_sha256": digest}}]
    run_dir = tmp_path / "private-run"
    run_dir.mkdir()
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=roster()) as ledger:
        ledger.start_clock()
        assert bind_comparison_prediction_snapshots(jobs, run_dir, tmp_path, ledger) == {job_id: digest}
        assert (run_dir / "predictions" / f"{job_id}.csv").read_bytes() == payload
        assert not ledger.stopped
