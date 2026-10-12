"""Final evidence contradictions must refuse before accepting a score.

Uses the coding agent's positive synthetic executor, then changes ONE
evidence claim. No real sealed data, model or sandbox is used.
"""
from dataclasses import replace
import hashlib
from pathlib import Path
import pytest
from research.comparison import final
from research.comparison.budget import ComparisonLedger
from research.tests.test_comparison_final import SyntheticExecutor
from research.tests.test_comparison_orchestration_supervisor import make_controller


@pytest.mark.parametrize("mutation", ["none", "live_job", "wrong_entrypoint_hash", "untrusted_evaluator", "failed_live_job"])
def test_final_acceptance_rejects_contradictory_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    orch = make_controller(tmp_path, monkeypatch)
    cell = orch.cells[0]
    source = b"print('synthetic final evidence')\n"
    source_sha = hashlib.sha256(source).hexdigest()
    answer = b"id,label\nx,0\n"
    sealed = tmp_path / "sealed"
    (sealed / cell.task_id).mkdir(parents=True)
    (sealed / cell.task_id / "final_labels.csv").write_bytes(answer)
    final_dir = orch.runtime_dir / "final" / cell.cell_key().replace(":", "-")
    final_dir.mkdir(parents=True)
    execution = SyntheticExecutor(sealed).execute(task_id=final.final_task_id(cell.task_id), source_bytes=source,
        cell_key=cell.cell_key(), timeout_seconds=100.0)
    assert execution.evidence is not None
    evidence = dict(execution.evidence)
    if mutation in ("live_job", "failed_live_job"):
        evidence["cleanup"] = {"live_jobs": [execution.job_id], "uncertain_submissions": [], "submission_intents": []}
    elif mutation == "wrong_entrypoint_hash":
        evidence["entrypoint_sha256"] = "0" * 64
    elif mutation == "untrusted_evaluator":
        evidence["evaluator"] = "candidate_reported"
    execution = replace(execution, evidence=evidence)
    if mutation == "failed_live_job":
        execution = replace(execution, status="failed", score=None, prediction_bytes=None, prediction_sha256=None)
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        def invoke():
            return final._translate_execution(orch, ledger, cell, {"source_sha256": source_sha}, execution,
                lambda: 1000.0, sealed, intent_sha256="a" * 64, barrier_identity="b" * 64,
                answer_sha256=hashlib.sha256(answer).hexdigest(),
                registry_version="synthetic-final-registry", winner_bytes=source)
        if mutation == "none":
            result = invoke()
            assert result["status"] == "completed" and result["raw_score"] == 1.0
            assert not ledger.stopped
        else:
            with pytest.raises(final.FinalPhaseError):
                invoke()
            assert ledger.stopped, "contradictory final provenance must durably stop the ORIGINAL ledger"
