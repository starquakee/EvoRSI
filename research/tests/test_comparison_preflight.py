"""Offline tests for research/comparison/preflight.py (EXP-007).

prepare/preflight/report run against the REAL repo bindings/registry with
a deterministic fake native provider and a throwaway runtime dir inside
the (ignored) .runtime tree. No model calls, no ledger creation, no
review file, no docker contact (the compose mount contract gate is covered
by test_worker_grant_mounts.py and the real preflight run).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import pytest

from research.comparison import deploy, preflight, review
from research.comparison.native import NativeCompletion
from research.comparison.protocol import COMPARISON_DIMS
from research.comparison.review import ComparisonManifestError

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT = "0123456789abcdef0123456789abcdef01234567"


class FakeNative:
    """Deterministic fake provider (bounds-valid, score-independent)."""

    def _batch(self, seed: int, salt: int) -> tuple[tuple[float, ...], ...]:
        rng_seed = int.from_bytes(
            hashlib.sha256(f"fake-native:{seed}:{salt}".encode()).digest()[:4], "big"
        )
        import random

        rng = random.Random(rng_seed)
        return tuple(
            tuple(rng.uniform(dim.lower, dim.upper) for dim in COMPARISON_DIMS)
            for _ in range(4)
        )

    def setup_population(self, seed: int) -> tuple[tuple[float, ...], ...]:
        return self._batch(seed, 0)

    def first_batch(self, seed: int) -> tuple[tuple[float, ...], ...]:
        return self._batch(seed, 1)

    def second_batch(
        self, seed: int, fitness: Sequence[float]
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
        return self._batch(seed, 1), self._batch(seed, 2)

    def complete_run(
        self,
        seed: int,
        first_fitness: Sequence[float],
        second_fitness: Sequence[float],
    ) -> NativeCompletion:
        second = self._batch(seed, 2)
        digest = hashlib.sha256(
            json.dumps([seed, [float(f) for f in first_fitness]]).encode()
        ).hexdigest()
        return NativeCompletion(
            first_batch=self._batch(seed, 1),
            second_batch=second,
            final_population_sha256=digest,
        )


@pytest.fixture()
def clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deploy, "git_head_commit", lambda root: COMMIT)
    monkeypatch.setattr(review, "_git_porcelain", lambda root: "")


@pytest.fixture()
def runtime_dir():
    base = REPO_ROOT / ".runtime" / "test-comparison-preflight"
    base.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=base))
    try:
        yield path
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture()
def prepared(clean_git, runtime_dir: Path):
    evidence = preflight.prepare_experiment(
        REPO_ROOT, provider=FakeNative(), runtime_dir=runtime_dir, now_fn=lambda: 1000.0
    )
    return evidence


class TestPrepare:
    def test_prepare_freezes_offline_state(self, prepared, runtime_dir: Path):
        assert prepared["schema"] == preflight.PREPARE_EVIDENCE_SCHEMA
        assert prepared["source_commit"] == COMMIT
        assert prepared["cells_initialized"] == 27
        assert prepared["ledger_created"] is False
        assert prepared["review_accepted"] is False
        assert (runtime_dir / "schedule.json").is_file()
        assert (runtime_dir / review.MANIFEST_FILENAME).is_file()
        cells = list((runtime_dir / "cells").iterdir())
        assert len(cells) == 27
        for task in ("banknote", "spambase", "wdbc"):
            readme = (
                runtime_dir / "final-publication" / task / "data" / "final" / "README.md"
            )
            assert readme.is_file()
        assert not (runtime_dir / "comparison-ledger.json").exists()
        assert not (runtime_dir / review.REVIEW_FILENAME).exists()

    def test_prepare_refuses_stray_runtime_content(self, clean_git, runtime_dir: Path):
        (runtime_dir / "stray.txt").write_text("x", encoding="utf-8")
        with pytest.raises(ComparisonManifestError, match="comparison_runtime_not_fresh"):
            preflight.prepare_experiment(
                REPO_ROOT,
                provider=FakeNative(),
                runtime_dir=runtime_dir,
                now_fn=lambda: 1000.0,
            )

    def test_prepare_tolerates_placeholder_tree(self, clean_git, runtime_dir: Path):
        preflight.create_final_placeholders(runtime_dir)
        evidence = preflight.prepare_experiment(
            REPO_ROOT,
            provider=FakeNative(),
            runtime_dir=runtime_dir,
            now_fn=lambda: 1000.0,
        )
        assert evidence["cells_initialized"] == 27

    def test_reprepare_archives_prior_preparation(
        self, prepared, runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A new exact HEAD may freeze fresh state BEFORE anything live."""
        other = "fedcba9876543210fedcba9876543210fedcba98"
        monkeypatch.setattr(deploy, "git_head_commit", lambda root: other)
        # preflight evidence from the first preparation must be preserved.
        preflight.run_preflight(
            REPO_ROOT,
            provider=FakeNative(),
            runtime_dir=runtime_dir,
            check_stack=False,
            now_fn=lambda: 1000.0,
        )
        evidence = preflight.prepare_experiment(
            REPO_ROOT,
            provider=FakeNative(),
            runtime_dir=runtime_dir,
            now_fn=lambda: 2000.0,
        )
        archived = evidence["archived_prior_preparation"]
        assert archived is not None
        archive_dir = runtime_dir.parent / archived
        # Prior evidence preserved, not deleted.
        assert (archive_dir / "schedule.json").is_file()
        assert (archive_dir / "preflight-evidence.v1.json").is_file()
        assert (archive_dir / review.MANIFEST_FILENAME).is_file()
        # Fresh state bound to the NEW commit; placeholders untouched.
        assert evidence["source_commit"] == other
        manifest = json.loads(
            (runtime_dir / review.MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert manifest["source_commit"] == other
        assert not (runtime_dir / "comparison-ledger.json").exists()

    def test_reprepare_refuses_live_artifacts(
        self, prepared, runtime_dir: Path
    ):
        (runtime_dir / "comparison-ledger.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ComparisonManifestError, match="comparison_runtime_live"):
            preflight.prepare_experiment(
                REPO_ROOT,
                provider=FakeNative(),
                runtime_dir=runtime_dir,
                now_fn=lambda: 2000.0,
            )

    def test_reprepare_refuses_launched_runs(
        self, prepared, runtime_dir: Path
    ):
        run_dir = runtime_dir / "runs" / "some-run"
        run_dir.mkdir(parents=True)
        (run_dir / "journal.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(
            ComparisonManifestError, match="comparison_runtime_not_fresh:runs"
        ):
            preflight.prepare_experiment(
                REPO_ROOT,
                provider=FakeNative(),
                runtime_dir=runtime_dir,
                now_fn=lambda: 2000.0,
            )


class TestPreflight:
    def test_preflight_requires_prepare(self, clean_git, runtime_dir: Path):
        with pytest.raises(ComparisonManifestError, match="comparison_manifest_missing"):
            preflight.run_preflight(
                REPO_ROOT,
                provider=FakeNative(),
                runtime_dir=runtime_dir,
                check_stack=False,
                now_fn=lambda: 1000.0,
            )
        assert not (runtime_dir / "preflight-evidence.v1.json").exists()

    def test_preflight_passes_prepared_state(self, prepared, runtime_dir: Path):
        evidence = preflight.run_preflight(
            REPO_ROOT,
            provider=FakeNative(),
            runtime_dir=runtime_dir,
            check_stack=False,
            now_fn=lambda: 1000.0,
        )
        assert evidence["verdict"] == "PASS"
        statuses = {gate["name"]: gate["status"] for gate in evidence["gates"]}
        assert set(statuses) == {
            "clean_head_matches_manifest",
            "manifest_rederivation_identical",
            "data_manifest_valid",
            "registry_pins",
            "native_env_resolves",
            "native_runtime_identity",
            "orchestration_state_valid",
            "native_replay_determinism",
            "historic_ledgers_unchanged",
            "historic_artifact_audit",
            "final_publication_placeholders",
            "no_live_artifacts",
        }
        assert all(status == "pass" for status in statuses.values())
        # The live gate stays visibly closed; preflight never opens it.
        assert evidence["live_execution_gate"] == {
            "status": "refused",
            "reason": "comparison_review_absent",
        }
        req = evidence["review_requirements"]
        assert req["schema"] == review.REVIEW_SCHEMA
        assert req["source_commit"] == COMMIT
        assert len(req["experiment_manifest_sha256"]) == 64
        # Evidence persisted deterministically modulo the volatile fields.
        persisted = json.loads(
            (runtime_dir / "preflight-evidence.v1.json").read_text(encoding="utf-8")
        )
        assert persisted["verdict"] == "PASS"

    def test_preflight_fails_visibly_on_tampered_state(self, prepared, runtime_dir: Path):
        cell_dir = next((runtime_dir / "cells").iterdir())
        payload = json.loads((cell_dir / "cell.json").read_text(encoding="utf-8"))
        payload["vectors"]["0"] = [0.9, 1.9, 0.9, 0.9, 1.0]
        (cell_dir / "cell.json").write_text(json.dumps(payload), encoding="utf-8")
        evidence = preflight.run_preflight(
            REPO_ROOT,
            provider=FakeNative(),
            runtime_dir=runtime_dir,
            check_stack=False,
            now_fn=lambda: 1000.0,
        )
        assert evidence["verdict"] == "FAIL"
        statuses = {gate["name"]: gate["status"] for gate in evidence["gates"]}
        assert statuses["orchestration_state_valid"] == "fail"

    def test_preflight_fails_on_premature_final_bytes(self, prepared, runtime_dir: Path):
        final_dir = runtime_dir / "final-publication" / "banknote" / "data" / "final"
        (final_dir / "train.csv").write_text("id,x\n0,1\n", encoding="utf-8")
        evidence = preflight.run_preflight(
            REPO_ROOT,
            provider=FakeNative(),
            runtime_dir=runtime_dir,
            check_stack=False,
            now_fn=lambda: 1000.0,
        )
        statuses = {gate["name"]: gate["status"] for gate in evidence["gates"]}
        assert statuses["final_publication_placeholders"] == "fail"
        assert evidence["verdict"] == "FAIL"

    def test_preflight_fails_on_live_ledger(self, prepared, runtime_dir: Path):
        (runtime_dir / "comparison-ledger.json").write_text("{}", encoding="utf-8")
        evidence = preflight.run_preflight(
            REPO_ROOT,
            provider=FakeNative(),
            runtime_dir=runtime_dir,
            check_stack=False,
            now_fn=lambda: 1000.0,
        )
        statuses = {gate["name"]: gate["status"] for gate in evidence["gates"]}
        assert statuses["no_live_artifacts"] == "fail"
        assert evidence["verdict"] == "FAIL"


class TestReport:
    def test_report_requires_prepare(self, clean_git, runtime_dir: Path):
        with pytest.raises(ComparisonManifestError, match="comparison_manifest_missing"):
            preflight.build_report(
                REPO_ROOT,
                provider=FakeNative(),
                runtime_dir=runtime_dir,
                now_fn=lambda: 1000.0,
            )

    def test_report_matrix_before_start(self, prepared, runtime_dir: Path):
        report = preflight.build_report(
            REPO_ROOT,
            provider=FakeNative(),
            runtime_dir=runtime_dir,
            now_fn=lambda: 1000.0,
        )
        assert report["schema"] == preflight.REPORT_SCHEMA
        assert report["source_commit"] == COMMIT
        assert report["ledger"] is None
        assert len(report["cells"]) == 27
        assert all(row["terminal_status"] is None for row in report["cells"])
        assert all(row["final"] is None for row in report["cells"])
        methods = {row["method"] for row in report["cells"]}
        assert methods == {"FIXED", "RANDOM", "ISTRATDE"}
