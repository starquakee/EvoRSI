"""Offline final-phase tests (EXP-006).

Synthetic roots/executor only — no real final data is released, no model
calls, no sandbox. Proves barrier-gated publication into the IGNORED
runtime publication root (tracked search registry/data byte-identical),
the irreversible final-phase model-request freeze, one-shot execution with
durable intents, prediction-byte persistence, independent score
recomputation, truthful null/zero failures and fail-closed resume.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from research.comparison import deploy
from research.comparison.budget import ComparisonLedger
from research.comparison.data import (
    EXPECTED_BARRIER_SCHEMA,
    ComparisonDataError,
    publish_final_refit,
)
from research.comparison.final import (
    FinalExecution,
    FinalPhaseError,
    publish_final_data,
    recompute_accuracy,
    run_final_phase,
)
from research.comparison.orchestrate import BARRIER_SCHEMA, Orchestrator
from research.comparison.protocol import TASKS
from research.comparison.specs import TaskBinding, iter_cells
from research.tests.test_comparison_orchestrate import (
    ScriptedNative,
    default_scores,
    fake_child,
)
from research.tests.test_comparison_specs import COMMIT, make_binding


@pytest.fixture(autouse=True)
def _pin_head(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deploy, "git_head_commit", lambda root: COMMIT)


def test_barrier_schema_constant_mirrors_orchestrate():
    assert EXPECTED_BARRIER_SCHEMA == BARRIER_SCHEMA
    # data.py duplicates RUNTIME_DIR (import cycle); pin the equality.
    from research.comparison.runner import RUNTIME_DIR

    assert RUNTIME_DIR == Path(".runtime") / "comparison-istratde"


# Synthetic task content, GENUINELY consistent: trainval is the exact
# train+validation refit (validation ids offset past the train ids) and all
# five binding hashes are the real content hashes of these bytes.
_TRAIN_CSV = b"id,f1,label\n0,0.1,0\n1,0.2,1\n"
_VALIDATION_FEATURES_CSV = b"id,f1\n0,0.3\n"
_VALIDATION_ANSWER_CSV = b"id,label\n0,0\n"
_TRAINVAL_CSV = b"id,f1,label\n0,0.1,0\n1,0.2,1\n2,0.3,0\n"
_FINAL_FEATURES_CSV = b"id,f1\n10,0.5\n11,0.6\n12,0.7\n13,0.8\n"
_FINAL_LABELS_CSV = b"id,label\n10,1\n11,0\n12,1\n13,1\n"


def _repo_bindings() -> dict[str, TaskBinding]:
    """Frozen bindings matching the synthetic repo content EXACTLY."""
    return {
        task.task_id: make_binding(
            task.task_id,
            train_csv_sha256=hashlib.sha256(_TRAIN_CSV).hexdigest(),
            validation_features_sha256=hashlib.sha256(
                _VALIDATION_FEATURES_CSV
            ).hexdigest(),
            validation_labels_sha256=hashlib.sha256(
                _VALIDATION_ANSWER_CSV
            ).hexdigest(),
            final_features_sha256=hashlib.sha256(_FINAL_FEATURES_CSV).hexdigest(),
            final_labels_sha256=hashlib.sha256(_FINAL_LABELS_CSV).hexdigest(),
        )
        for task in TASKS
    }


def _make_repo_orch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Orchestrator:
    monkeypatch.setattr(deploy, "git_head_commit", lambda root: COMMIT)
    return Orchestrator.initialize(
        runtime_dir=tmp_path / "state",
        repo_root=tmp_path,
        cells=iter_cells(_repo_bindings(), COMMIT),
        provider=ScriptedNative(),
        now_fn=lambda: 1000.0,
    )


def _synthetic_repo(root: Path) -> dict[str, Any]:
    """Synthetic sealed/public/evaluator tree + manifest + tracked search
    registry, genuinely matching _repo_bindings: every bound file exists
    with the pinned hash and trainval is the bound refit."""
    bindings = _repo_bindings()
    sealed = root / "research" / "comparison" / "sealed"
    public_root = root / "tasks" / "comparison"
    evaluator_root = root / "research" / "evaluator"
    manifest_tasks: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        task_id = task.task_id
        (sealed / task_id).mkdir(parents=True)
        (public_root / task_id / "data" / "public").mkdir(parents=True)
        (evaluator_root / "tasks" / f"comparison_{task_id}").mkdir(parents=True)
        (sealed / task_id / "trainval.csv").write_bytes(_TRAINVAL_CSV)
        (sealed / task_id / "final_features.csv").write_bytes(_FINAL_FEATURES_CSV)
        (sealed / task_id / "final_labels.csv").write_bytes(_FINAL_LABELS_CSV)
        (public_root / task_id / "data" / "public" / "train.csv").write_bytes(
            _TRAIN_CSV
        )
        (public_root / task_id / "data" / "public" / "test.csv").write_bytes(
            _VALIDATION_FEATURES_CSV
        )
        (
            evaluator_root
            / "tasks"
            / f"comparison_{task_id}"
            / "validation_answer.csv"
        ).write_bytes(_VALIDATION_ANSWER_CSV)
        binding = bindings[task_id]
        manifest_tasks[task_id] = {
            "files": {
                f"tasks/comparison/{task_id}/data/public/train.csv": binding.train_csv_sha256,
                f"tasks/comparison/{task_id}/data/public/test.csv": binding.validation_features_sha256,
                f"research/evaluator/tasks/comparison_{task_id}/validation_answer.csv": binding.validation_labels_sha256,
                f"research/comparison/sealed/{task_id}/trainval.csv": hashlib.sha256(
                    _TRAINVAL_CSV
                ).hexdigest(),
                f"research/comparison/sealed/{task_id}/final_features.csv": binding.final_features_sha256,
                f"research/comparison/sealed/{task_id}/final_labels.csv": binding.final_labels_sha256,
            },
            "task_binding": binding.to_dict(),
            "task_binding_identity": binding.identity(),
        }
    manifest = {
        "schema": "istratde-comparison-data.v1",
        "tasks": manifest_tasks,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # Tracked SEARCH registry: frozen search-phase content that publication
    # must never modify. Contains one search-phase entry so the phase
    # registry (search entries + final entries) loads fail-closed-valid.
    registry_path = root / "research" / "evaluator" / "registry.v1.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_payload = {
        "registry_version": "evaluator-registry.v1",
        "schema_version": 1,
        "tasks": {
            "comparison_banknote": {
                "metric": "accuracy",
                "split": "validation",
                "answer_file": "tasks/comparison_banknote/validation_answer.csv",
                "answer_sha256": hashlib.sha256(_VALIDATION_ANSWER_CSV).hexdigest(),
                "id_column": "id",
                "target_column": "label",
                "label_values": [0, 1],
                "max_prediction_bytes": 1048576,
                "max_rows": 100000,
            }
        },
    }
    registry_path.write_text(json.dumps(registry_payload, indent=2) + "\n")
    return {
        "manifest_path": manifest_path,
        "registry_path": registry_path,
        "registry_bytes": registry_path.read_bytes(),
        "sealed": sealed,
        "root": root,
    }


def _publish_kwargs(repo: dict) -> dict:
    return {
        "sealed_root": repo["sealed"],
        "publication_root": repo["root"] / "final-publication",
        "registry_path": repo["registry_path"],
        "manifest_path": repo["manifest_path"],
        "public_root": repo["root"] / "tasks" / "comparison",
        "evaluator_root": repo["root"] / "research" / "evaluator",
    }


def _run_final(orch, ledger, barrier, executor, *, repo: dict, **kwargs):
    """run_final_phase with the synthetic repo's genuine bound-input roots."""
    return run_final_phase(
        orch,
        ledger,
        barrier,
        executor,
        sealed_root=repo["sealed"],
        public_root=repo["root"] / "tasks" / "comparison",
        evaluator_root=repo["root"] / "research" / "evaluator",
        **kwargs,
    )


class SyntheticExecutor:
    """Truthful fake: predicts all label 0, scores against sealed labels,
    and returns the full bound provenance payload (real pinned policy
    identity, real whole-artifact tree hash, actual trusted facts)."""

    def __init__(
        self,
        sealed_root: Path,
        cleanup: bool = True,
        registry_version: str = "synthetic-final-registry",
    ) -> None:
        self.sealed_root = sealed_root
        self.cleanup = cleanup
        self.registry_version = registry_version
        self.calls: list[str] = []

    def execute(
        self,
        *,
        task_id: str,
        source_bytes: bytes,
        cell_key: str,
        timeout_seconds: float,
    ) -> FinalExecution:
        from research.contracts.source_gate import load_policy
        from research.search.sandbox_task import _single_file_tree_sha256

        self.calls.append(cell_key)
        task = cell_key.split(":")[0]
        answers = (
            self.sealed_root / task / "final_labels.csv"
        ).read_bytes()
        lines = answers.decode().splitlines()
        ids = [line.split(",")[0] for line in lines[1:]]
        prediction = ("id,label\n" + "\n".join(f"{i},0" for i in ids) + "\n").encode()
        score, rows = recompute_accuracy(prediction, answers)
        policy = load_policy()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        job_id = hashlib.sha256(f"final-job-{cell_key}".encode("utf-8")).hexdigest()[:16]
        evidence = {
            "job_id": job_id,
            "status": "completed",
            "task_id": task_id,
            "split": "final",
            "metric": "accuracy",
            "direction": "maximize",
            "evaluator": "external_trusted",
            "answer_sha256": hashlib.sha256(answers).hexdigest(),
            "registry_version": self.registry_version,
            "row_count": rows,
            "entrypoint": "main.py",
            "entrypoint_sha256": source_sha,
            "source_sha256": _single_file_tree_sha256("main.py", source_bytes),
            "policy_version": policy.policy_version,
            "policy_sha256": policy.policy_sha256,
            "submitted_source_sha256": source_sha,
            "source_identity_verified": True,
            "worker_cleanup_verified": True,
            "prediction_snapshot": (
                "/mnt/rsi_storage/mlsandbox/jobs/2026-10-11/final/"
                f"{job_id}/prediction_snapshot.csv"
            ),
            "prediction_sha256": hashlib.sha256(prediction).hexdigest(),
            "cleanup": {
                "live_jobs": [],
                "uncertain_submissions": [],
                "submission_intents": [],
            },
        }
        return FinalExecution(
            status="completed",
            job_id=job_id,
            score=score,
            prediction_bytes=prediction,
            prediction_sha256=hashlib.sha256(prediction).hexdigest(),
            cleanup_verified=self.cleanup,
            reason=None,
            evidence=evidence,
        )


@pytest.fixture(scope="module")
def frozen(tmp_path_factory) -> dict:
    """ONE fully evidenced completed 216-config experiment (module-shared)."""
    base = tmp_path_factory.mktemp("frozen")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(deploy, "git_head_commit", lambda root: COMMIT)
        provider = ScriptedNative()
        orch = Orchestrator.initialize(
            runtime_dir=base / "state",
            repo_root=base,
            cells=iter_cells(_repo_bindings(), COMMIT),
            provider=provider,
            now_fn=lambda: 1000.0,
        )
        ledger_path = base / "ledger.json"
        with ComparisonLedger.open(
            ledger_path, cells=orch.cells, now_fn=lambda: 1000.0
        ) as ledger:
            ledger.start_clock()
            orch.run_schedule(
                ledger, fake_child(orch, default_scores(orch), [], ledger_path)
            )
    return {"base": base, "ledger_path": ledger_path, "runtime_dir": base / "state"}


def _frozen_copy(frozen: dict, tmp_path: Path):
    """Per-test copy of the frozen experiment + reopened handles/barrier."""
    base = tmp_path / "work"
    shutil.copytree(frozen["base"], base)
    orch = Orchestrator.load(
        runtime_dir=base / "state",
        repo_root=base,
        bindings=_repo_bindings(),
        provider=ScriptedNative(),
        now_fn=lambda: 1000.0,
    )
    ledger = ComparisonLedger.open(
        base / "ledger.json", cells=orch.cells, now_fn=lambda: 1000.0
    )
    barrier = orch.verify_final_barrier(ledger)
    return orch, ledger, barrier


def test_publish_and_final_phase_happy_path(frozen, tmp_path):
    orch, ledger, barrier = _frozen_copy(frozen, tmp_path)
    repo = _synthetic_repo(tmp_path / "repo")
    publication_root = repo["root"] / "final-publication"
    try:
        publication = publish_final_data(
            orch, ledger, barrier, now_fn=lambda: 2000.0, **_publish_kwargs(repo)
        )
        assert publication["barrier_identity"] == barrier["barrier_identity"]
        # The irreversible phase transition is bound to the barrier.
        assert ledger.final_phase == {
            "barrier_identity": barrier["barrier_identity"],
            "at": 1000.0,
        }
        assert not ledger.stopped
        # Publication lands ONLY in the ignored runtime publication root.
        for task in TASKS:
            final_dir = publication_root / task.task_id / "data" / "final"
            assert (final_dir / "train.csv").read_bytes() == (
                repo["sealed"] / task.task_id / "trainval.csv"
            ).read_bytes()
            assert (final_dir / "test.csv").read_bytes() == (
                repo["sealed"] / task.task_id / "final_features.csv"
            ).read_bytes()
            sample = (final_dir / "sample_submission.csv").read_text()
            assert set(sample.splitlines()[1:]) <= {f"{i},0" for i in range(10, 14)}
            assert not (final_dir / "final_labels.csv").exists()
            answer = (
                publication_root
                / "answers"
                / f"comparison_{task.task_id}"
                / "final_answer.csv"
            )
            assert answer.read_bytes() == (
                repo["sealed"] / task.task_id / "final_labels.csv"
            ).read_bytes()
        # The tracked search registry is byte-identical; the phase registry
        # holds EXACTLY the three final entries (search entries are not
        # republished; their answers stay in the tracked evaluator tree).
        assert repo["registry_path"].read_bytes() == repo["registry_bytes"]
        phase_registry = json.loads(
            (publication_root / "registry.final.v1.json").read_text()
        )
        assert set(phase_registry["tasks"]) == {
            f"comparison_{t.task_id}_final" for t in TASKS
        }
        assert all(
            phase_registry["tasks"][f"comparison_{t.task_id}_final"]["split"] == "final"
            for t in TASKS
        )
        assert barrier["barrier_identity"][:16] in phase_registry["registry_version"]
        executor = SyntheticExecutor(
            repo["sealed"], registry_version=publication["registry_version"]
        )
        results = _run_final(
            orch, ledger, barrier, executor, now_fn=lambda: 2000.0,
            repo=repo,
        )
        assert len(results) == 27
        assert len(executor.calls) == 27
        for cell_key, result in results.items():
            assert result["status"] == "completed"
            assert result["raw_score"] == pytest.approx(0.25)  # 1 of 4 label 0
            assert result["utility"] == result["raw_score"]
            assert result["recomputed_accuracy"] == pytest.approx(0.25)
            assert result["barrier_identity"] == barrier["barrier_identity"]
            # Prediction BYTES persisted and bound to the result hash.
            prediction_path = (
                orch.runtime_dir
                / "final"
                / cell_key.replace(":", "-")
                / "prediction.csv"
            )
            assert prediction_path.exists()
            assert (
                hashlib.sha256(prediction_path.read_bytes()).hexdigest()
                == result["prediction_sha256"]
            )
        # Resume: every cell skipped after FULL re-validation, executor
        # never re-invoked.
        again = _run_final(
            orch, ledger, barrier, executor, now_fn=lambda: 2000.0,
            repo=repo,
        )
        assert len(executor.calls) == 27
        assert again == results
    finally:
        ledger.close()


def test_publish_idempotent_and_conflict_refused(frozen, tmp_path):
    orch, ledger, barrier = _frozen_copy(frozen, tmp_path)
    repo = _synthetic_repo(tmp_path / "repo")
    publication_root = repo["root"] / "final-publication"
    try:
        publish_final_data(orch, ledger, barrier, now_fn=lambda: 2000.0, **_publish_kwargs(repo))
        # Identical bytes: idempotent (phase re-entry with the same barrier
        # is a no-op).
        publish_final_data(orch, ledger, barrier, now_fn=lambda: 2000.0, **_publish_kwargs(repo))
        # Different bytes at a publication target: refused closed (and the
        # ledger is durably stopped by the publication failure).
        target = publication_root / "banknote" / "data" / "final" / "train.csv"
        target.write_bytes(b"different")
        with pytest.raises(FinalPhaseError, match="artifact_mismatch"):
            publish_final_data(
                orch, ledger, barrier, now_fn=lambda: 2000.0, **_publish_kwargs(repo)
            )
        assert ledger.stopped
    finally:
        ledger.close()


def test_publish_rejects_stale_or_tampered_barrier(frozen, tmp_path):
    orch, ledger, barrier = _frozen_copy(frozen, tmp_path)
    repo = _synthetic_repo(tmp_path / "repo")
    try:
        stale = dict(barrier)
        stale["cells"] = dict(barrier["cells"])
        stale["cells"]["banknote:11:FIXED"] = {
            "terminal_status": "failed_no_valid_program",
            "winner": None,
        }
        # Fix the identity so structural validation passes but the fresh
        # recomputation disagrees.
        from research.comparison.protocol import identity_of

        stale["barrier_identity"] = identity_of(
            {k: v for k, v in stale.items() if k != "barrier_identity"}
        )
        with pytest.raises(FinalPhaseError, match="final_barrier_stale"):
            publish_final_data(
                orch, ledger, stale, now_fn=lambda: 2000.0, **_publish_kwargs(repo)
            )
        # Nothing was published.
        assert not (repo["root"] / "final-publication").exists()
    finally:
        ledger.close()


def test_publish_requires_orchestrator_and_ledger_context(frozen, tmp_path):
    """Direct publication calls fail closed without the real context —
    BEFORE any sealed data or manifest is read."""
    orch, ledger, barrier = _frozen_copy(frozen, tmp_path)
    repo = _synthetic_repo(tmp_path / "repo")
    try:
        with pytest.raises(ComparisonDataError, match="final_barrier_unproven"):
            publish_final_refit(barrier_evidence=barrier, **_publish_kwargs(repo))
        # A foreign (other-experiment) ledger with matching shape but wrong
        # experiment identity is refused by the full shared verification.
        from research.comparison.specs import iter_cells

        other_cells = iter_cells(_repo_bindings(), "f" * 40)
        foreign = ComparisonLedger.open(
            tmp_path / "foreign.json", cells=other_cells, now_fn=lambda: 1000.0
        )
        try:
            foreign.start_clock()
            with pytest.raises(ComparisonDataError, match="final_barrier_unproven"):
                publish_final_refit(
                    orchestrator=orch, ledger=foreign, **_publish_kwargs(repo)
                )
        finally:
            foreign.close()
        assert not (repo["root"] / "final-publication").exists()
    finally:
        ledger.close()


def test_publish_requires_entered_final_phase(frozen, tmp_path):
    """The durable irreversible phase transition must be bound to the
    barrier BEFORE sealed bytes are copied; calling publish_final_refit
    directly without it refuses."""
    orch, ledger, barrier = _frozen_copy(frozen, tmp_path)
    repo = _synthetic_repo(tmp_path / "repo")
    try:
        with pytest.raises(ComparisonDataError, match="final_phase_not_entered"):
            publish_final_refit(
                orchestrator=orch,
                ledger=ledger,
                barrier_evidence=barrier,
                **_publish_kwargs(repo),
            )
        assert not (repo["root"] / "final-publication").exists()
    finally:
        ledger.close()


def test_final_phase_requires_publication(frozen, tmp_path):
    orch, ledger, barrier = _frozen_copy(frozen, tmp_path)
    repo = _synthetic_repo(tmp_path / "repo")
    try:
        # Phase never entered: refused (and the ledger durably stops).
        with pytest.raises(FinalPhaseError, match="final_phase_not_entered"):
            _run_final(
                orch, ledger, barrier, SyntheticExecutor(repo["sealed"]),
                now_fn=lambda: 2000.0, repo=repo,
            )
        assert ledger.stopped
    finally:
        ledger.close()
    # Published but the durable record is missing: refused.
    orch, ledger, barrier = _frozen_copy(frozen, tmp_path / "second")
    try:
        publish_final_data(orch, ledger, barrier, now_fn=lambda: 2000.0, **_publish_kwargs(repo))
        (orch.runtime_dir / "final" / "publication.json").unlink()
        with pytest.raises(FinalPhaseError, match="final_data_not_published"):
            _run_final(
                orch, ledger, barrier, SyntheticExecutor(repo["sealed"]),
                now_fn=lambda: 2000.0, repo=repo,
            )
        assert ledger.stopped
    finally:
        ledger.close()


def test_final_duplicate_execution_refused_after_crash(frozen, tmp_path):
    orch, ledger, barrier = _frozen_copy(frozen, tmp_path)
    repo = _synthetic_repo(tmp_path / "repo")
    try:
        publish_final_data(orch, ledger, barrier, now_fn=lambda: 2000.0, **_publish_kwargs(repo))

        class CrashingExecutor(SyntheticExecutor):
            def execute(self, **kwargs):
                super().execute(**kwargs)
                raise RuntimeError("simulated_crash_after_submit")

        with pytest.raises(FinalPhaseError, match="final_infrastructure_failure"):
            _run_final(
                orch, ledger, barrier, CrashingExecutor(repo["sealed"]),
                now_fn=lambda: 2000.0, repo=repo,
            )
        assert ledger.stopped
    finally:
        ledger.close()
    # Resume with a fresh ledger handle: durable intent without result is a
    # permanent refusal — the frozen program is never executed twice.
    resumed = Orchestrator.load(
        runtime_dir=orch.runtime_dir,
        repo_root=tmp_path / "work",
        bindings=_repo_bindings(),
        provider=ScriptedNative(),
        now_fn=lambda: 1000.0,
    )
    with ComparisonLedger.open(
        tmp_path / "work" / "ledger.json", cells=resumed.cells, now_fn=lambda: 1000.0
    ) as ledger2:
        executor = SyntheticExecutor(repo["sealed"])
        with pytest.raises(FinalPhaseError, match="final_interrupted|ledger_stopped"):
            _run_final(
                resumed, ledger2, barrier, executor,
                now_fn=lambda: 2000.0, repo=repo,
            )
        assert executor.calls == []


def test_final_candidate_failure_is_null_and_zero(frozen, tmp_path):
    orch, ledger, barrier = _frozen_copy(frozen, tmp_path)
    repo = _synthetic_repo(tmp_path / "repo")
    try:
        publication = publish_final_data(orch, ledger, barrier, now_fn=lambda: 2000.0, **_publish_kwargs(repo))

        class FailingExecutor(SyntheticExecutor):
            def execute(self, **kwargs):
                if kwargs["cell_key"] == "banknote:11:FIXED":
                    return FinalExecution(
                        status="failed",
                        job_id="job-x",
                        score=None,
                        prediction_bytes=None,
                        prediction_sha256=None,
                        cleanup_verified=True,
                        reason="sandbox_failed",
                        evidence={
                            "job_id": "job-x",
                            "status": "failed",
                            "cleanup": {
                                "live_jobs": [],
                                "uncertain_submissions": [],
                                "submission_intents": [],
                            },
                        },
                    )
                return super().execute(**kwargs)

        results = _run_final(
            orch, ledger, barrier,
            FailingExecutor(repo["sealed"], registry_version=publication["registry_version"]),
            now_fn=lambda: 2000.0, repo=repo,
        )
        failed = results["banknote:11:FIXED"]
        assert failed["status"] == "failed"
        assert failed["raw_score"] is None
        assert failed["utility"] == 0.0
        assert not ledger.stopped  # ordinary candidate failure stays local
        assert sum(1 for r in results.values() if r["status"] == "completed") == 26
    finally:
        ledger.close()


def test_final_cleanup_failure_stops_ledger(frozen, tmp_path):
    orch, ledger, barrier = _frozen_copy(frozen, tmp_path)
    repo = _synthetic_repo(tmp_path / "repo")
    try:
        publish_final_data(orch, ledger, barrier, now_fn=lambda: 2000.0, **_publish_kwargs(repo))
        with pytest.raises(FinalPhaseError, match="final_cleanup_unverified"):
            _run_final(
                orch, ledger, barrier,
                SyntheticExecutor(repo["sealed"], cleanup=False),
                now_fn=lambda: 2000.0, repo=repo,
            )
        assert ledger.stopped
    finally:
        ledger.close()


def test_final_score_recompute_mismatch_stops_ledger(frozen, tmp_path):
    orch, ledger, barrier = _frozen_copy(frozen, tmp_path)
    repo = _synthetic_repo(tmp_path / "repo")
    try:
        publish_final_data(orch, ledger, barrier, now_fn=lambda: 2000.0, **_publish_kwargs(repo))

        class LyingExecutor(SyntheticExecutor):
            def execute(self, **kwargs):
                execution = super().execute(**kwargs)
                return FinalExecution(
                    status="completed",
                    job_id=execution.job_id,
                    score=0.99,  # disagrees with the prediction bytes
                    prediction_bytes=execution.prediction_bytes,
                    prediction_sha256=execution.prediction_sha256,
                    cleanup_verified=True,
                    reason=None,
                    evidence=execution.evidence,
                )

        with pytest.raises(FinalPhaseError, match="final_score_recompute_mismatch"):
            _run_final(
                orch, ledger, barrier, LyingExecutor(repo["sealed"]),
                now_fn=lambda: 2000.0, repo=repo,
            )
        assert ledger.stopped
    finally:
        ledger.close()


def test_final_winner_tamper_stops_ledger(frozen, tmp_path):
    orch, ledger, barrier = _frozen_copy(frozen, tmp_path)
    repo = _synthetic_repo(tmp_path / "repo")
    try:
        publish_final_data(orch, ledger, barrier, now_fn=lambda: 2000, **_publish_kwargs(repo))
        winner_path = (
            orch.runtime_dir / "cells" / "banknote-11-FIXED" / "winner.py"
        )
        winner_path.write_text("# tampered after barrier\n", encoding="utf-8")
        # The barrier re-verification itself detects the tamper (and the
        # phase stops the ledger durably before raising).
        with pytest.raises(FinalPhaseError, match="winner_bytes_tampered"):
            _run_final(
                orch, ledger, barrier, SyntheticExecutor(repo["sealed"]),
                now_fn=lambda: 2000.0, repo=repo,
            )
        assert ledger.stopped
    finally:
        ledger.close()


def test_final_resume_rejects_tampered_prediction_bytes(frozen, tmp_path):
    """Persisted prediction bytes are re-hashed and the score recomputed on
    resume; tampering with them is rejected (not cached-truth accepted)."""
    orch, ledger, barrier = _frozen_copy(frozen, tmp_path)
    repo = _synthetic_repo(tmp_path / "repo")
    try:
        publication = publish_final_data(orch, ledger, barrier, now_fn=lambda: 2000.0, **_publish_kwargs(repo))
        executor = SyntheticExecutor(
            repo["sealed"], registry_version=publication["registry_version"]
        )
        results = _run_final(
            orch, ledger, barrier, executor, now_fn=lambda: 2000.0,
            repo=repo,
        )
        assert len(results) == 27
        prediction_path = (
            orch.runtime_dir / "final" / "banknote-11-FIXED" / "prediction.csv"
        )
        prediction_path.write_bytes(b"id,label\n10,1\n11,1\n12,1\n13,1\n")
        with pytest.raises(FinalPhaseError, match="final_prediction_mismatch"):
            _run_final(
                orch, ledger, barrier, SyntheticExecutor(repo["sealed"]),
                now_fn=lambda: 2000.0, repo=repo,
            )
        assert ledger.stopped
    finally:
        ledger.close()


def test_final_phase_respects_original_deadline(tmp_path, monkeypatch):
    orch = _make_repo_orch(tmp_path, monkeypatch)
    clock = [1000.0]
    ledger_path = tmp_path / "ledger.json"
    with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: clock[0]) as ledger:
        ledger.start_clock()
        orch.run_schedule(
            ledger, fake_child(orch, default_scores(orch), [], ledger_path)
        )
    repo = _synthetic_repo(tmp_path / "repo")
    ledger = ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: clock[0])
    try:
        barrier = orch.verify_final_barrier(ledger)
        publish_final_data(orch, ledger, barrier, now_fn=lambda: 2000.0, **_publish_kwargs(repo))
        clock[0] += 86401.0  # past the ORIGINAL deadline; no new timer
        executor = SyntheticExecutor(repo["sealed"])
        with pytest.raises(FinalPhaseError, match="deadline_expired"):
            _run_final(
                orch, ledger, barrier, executor,
                now_fn=lambda: 2000.0, repo=repo,
            )
        assert executor.calls == []
        assert ledger.stopped
    finally:
        ledger.close()


def test_recompute_accuracy_strict():
    answers = b"id,label\n0,0\n1,1\n2,1\n"
    assert recompute_accuracy(b"id,label\n0,0\n1,1\n2,0\n", answers) == (2 / 3, 3)
    with pytest.raises(FinalPhaseError, match="final_prediction_rows_mismatch"):
        recompute_accuracy(b"id,label\n0,0\n1,1\n", answers)
    with pytest.raises(FinalPhaseError, match="final_prediction_malformed"):
        recompute_accuracy(b"id,label\n0,2\n1,1\n2,0\n", answers)
    with pytest.raises(FinalPhaseError, match="final_prediction_malformed"):
        recompute_accuracy(b"wrong,header\n0,0\n", answers)
