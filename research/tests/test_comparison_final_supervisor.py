"""Final scoring must prove its inputs and refuse late success."""
import hashlib
import json
from pathlib import Path
import pytest
from research.comparison.budget import ComparisonLedger
from research.comparison import final
from research.comparison.protocol import identity_of
from research.tests.test_comparison_orchestration_supervisor import make_controller


def fake_barrier(orch):
    value = {
        "schema": "istratde-comparison-barrier.v1", "source_commit": orch.source_commit,
        "experiment_identity": "b" * 64, "schedule_identity": "a" * 64,
        "cells": {cell.cell_key(): {"terminal_status": "failed_no_valid_program", "winner": None}
                  for cell in orch.cells},
    }
    value["barrier_identity"] = identity_of(value)
    return value


def fake_publication(orch, barrier) -> None:
    """Positive fixture for the STRONGER publication contract: a durable
    record whose every listed file actually exists with the pinned hash
    (the final-phase validator re-hashes all of them), genuinely anchored
    to the frozen bindings — train.csv is the EXACT refit recomputed from
    the binding-pinned train/validation inputs."""
    from research.comparison import data
    from research.tests.test_comparison_specs import (
        SYNTHETIC_FINAL_FEATURES_CSV,
        SYNTHETIC_FINAL_LABELS_CSV,
        SYNTHETIC_TRAIN_CSV,
        SYNTHETIC_VALIDATION_FEATURES_CSV,
        SYNTHETIC_VALIDATION_LABELS_CSV,
    )

    def _place(base: Path, rel: str, content: bytes) -> str:
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return hashlib.sha256(content).hexdigest()

    root = orch.runtime_dir / "final"
    root.mkdir(exist_ok=True)
    publication_root = orch.runtime_dir / "final-publication"
    trainval = data._refit_trainval_bytes(
        SYNTHETIC_TRAIN_CSV,
        SYNTHETIC_VALIDATION_FEATURES_CSV,
        SYNTHETIC_VALIDATION_LABELS_CSV,
    )
    files = {}
    for cell_task in ("banknote", "wdbc", "spambase"):
        # Binding-pinned ORIGINAL inputs under the orch repo root.
        _place(
            orch.repo_root,
            f"tasks/comparison/{cell_task}/data/public/train.csv",
            SYNTHETIC_TRAIN_CSV,
        )
        _place(
            orch.repo_root,
            f"tasks/comparison/{cell_task}/data/public/test.csv",
            SYNTHETIC_VALIDATION_FEATURES_CSV,
        )
        _place(
            orch.repo_root,
            f"research/evaluator/tasks/comparison_{cell_task}/validation_answer.csv",
            SYNTHETIC_VALIDATION_LABELS_CSV,
        )
        files[f"{cell_task}/data/final/train.csv"] = _place(
            publication_root, f"{cell_task}/data/final/train.csv", trainval
        )
        files[f"{cell_task}/data/final/test.csv"] = _place(
            publication_root,
            f"{cell_task}/data/final/test.csv",
            SYNTHETIC_FINAL_FEATURES_CSV,
        )
        files[f"answers/comparison_{cell_task}/final_answer.csv"] = _place(
            publication_root,
            f"answers/comparison_{cell_task}/final_answer.csv",
            SYNTHETIC_FINAL_LABELS_CSV,
        )
    (root / "publication.json").write_text(json.dumps({
        "schema": "istratde-comparison-final-publication.v1",
        "barrier_identity": barrier["barrier_identity"],
        "registry_version": "synthetic-final-registry",
        "publication_root": str(publication_root),
        "registry_file": "registry.final.v1.json",
        "files": files,
    }))
    for cell in orch.cells:
        (root / cell.cell_key().replace(":", "-")).mkdir(exist_ok=True)


def test_final_execution_rechecks_actual_search_barrier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orch = make_controller(tmp_path, monkeypatch)
    barrier = fake_barrier(orch)
    fake_publication(orch, barrier)
    class NeverExecute:
        def execute(self, **kwargs):
            raise AssertionError("no verified search has occurred")
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        with pytest.raises(final.FinalPhaseError):
            final.run_final_phase(orch, ledger, barrier, NeverExecute(), now_fn=lambda: 1000.0,
                                  sealed_root=tmp_path / "sealed")


def test_final_resume_rejects_score_without_bound_prediction_or_intent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orch = make_controller(tmp_path, monkeypatch)
    key = orch.cells[0].cell_key()
    path = tmp_path / "final.json"
    path.write_text(json.dumps({"schema": final.FINAL_RESULT_SCHEMA, "cell_key": key,
                                "status": "completed", "raw_score": 0.9, "utility": 0.9}))
    with pytest.raises(final.FinalPhaseError):
        final._validate_final_result(orch, key, path)


def test_final_execution_finishing_after_original_deadline_is_not_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orch = make_controller(tmp_path, monkeypatch)
    barrier = fake_barrier(orch)
    first = orch.cells[0]
    key = first.cell_key()
    source = b"print('synthetic final test')\n"
    sha = hashlib.sha256(source).hexdigest()
    barrier["cells"][key] = {"terminal_status": "frozen", "winner": {
        "source_sha256": sha, "code_sha256": sha, "config_index": 0,
        "validation_score": 0.5, "job_id": "synthetic-validation-job", "prediction_sha256": "d" * 64,
    }}
    barrier.pop("barrier_identity")
    barrier["barrier_identity"] = identity_of(barrier)
    fake_publication(orch, barrier)
    # Synthetic positive barrier isolates the post-execution deadline boundary.
    # No real sealed data or model/sandbox call is involved.
    monkeypatch.setattr(orch, "verify_final_barrier", lambda ledger: barrier)
    (orch.runtime_dir / "cells" / key.replace(":", "-") / "winner.py").write_bytes(source)
    sealed = tmp_path / "sealed" / first.task_id
    sealed.mkdir(parents=True)
    prediction = b"id,label\nx,1\n"
    (sealed / "final_labels.csv").write_bytes(prediction)
    clock = [1000.0]
    late_execution_calls = []
    class LateExecutor:
        def execute(self, **kwargs):
            late_execution_calls.append(kwargs["cell_key"])
            clock[0] += 86401.0
            return final.FinalExecution("completed", "synthetic-final-job", 1.0, prediction,
                                        hashlib.sha256(prediction).hexdigest(), True, None)
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: clock[0]) as ledger:
        ledger.start_clock()
        # The verified publication flow enters the irreversible final phase
        # BEFORE final execution; mirror that ordering in the fixture.
        ledger.enter_final_phase(barrier["barrier_identity"])
        with pytest.raises(final.FinalPhaseError):
            final.run_final_phase(orch, ledger, barrier, LateExecutor(), now_fn=lambda: clock[0],
                                  sealed_root=tmp_path / "sealed")
        assert ledger.stopped
    assert late_execution_calls == [key], "positive fixture must reach the actual late execution boundary"
    result_path = orch.runtime_dir / "final" / key.replace(":", "-") / "final.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result.get("status") != "completed", "a late job cannot be accepted before the next cell checks time"
