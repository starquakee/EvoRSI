"""Offline tests for research/comparison/review.py (EXP-007).

The immutable experiment manifest derives deterministically from the real
repo; the supervisor review gate refuses unless an ignored, untracked,
strictly well-formed review file binds the exact clean HEAD and the exact
frozen manifest bytes, and a full re-derivation reproduces the manifest.
All git state is monkeypatched; tests never write a real review file or
touch the real runtime dir.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.comparison import deploy, review
from research.comparison.protocol import canonical_json, canonical_protocol
from research.comparison.review import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    REVIEW_FILENAME,
    REVIEW_SCHEMA,
    ComparisonManifestError,
    ComparisonReviewRefused,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT = "0123456789abcdef0123456789abcdef01234567"
OTHER_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return review.build_experiment_manifest(REPO_ROOT, commit=COMMIT)


@pytest.fixture()
def runtime(tmp_path, monkeypatch):
    """A prepared-looking runtime dir in the ignored tree, git state faked."""
    monkeypatch.setattr(deploy, "git_head_commit", lambda root: COMMIT)
    monkeypatch.setattr(review, "_git_porcelain", lambda root: "")
    monkeypatch.setattr(review, "_git_ls_files_tracked", lambda root, rel: False)
    import shutil
    import tempfile

    base = REPO_ROOT / ".runtime" / "test-comparison-review"
    base.mkdir(parents=True, exist_ok=True)
    runtime_dir = Path(tempfile.mkdtemp(dir=base))
    try:
        manifest = review.build_experiment_manifest(REPO_ROOT, commit=COMMIT)
        manifest_path = review.freeze_experiment_manifest(
            REPO_ROOT, manifest, runtime_dir=runtime_dir
        )
        yield SimpleNamespace(
            RuntimeDir=runtime_dir, manifest=manifest, manifest_path=manifest_path
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _write_review(runtime_dir: Path, manifest_path: Path, commit: str = COMMIT) -> Path:
    payload = {
        "schema": REVIEW_SCHEMA,
        "source_commit": commit,
        "experiment_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "accepted_at": "2026-10-11T00:00:00+00:00",
        "reviewer": "supervisor",
    }
    path = runtime_dir / REVIEW_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestManifestDerivation:
    def test_deterministic(self):
        first = review.build_experiment_manifest(REPO_ROOT, commit=COMMIT)
        second = review.build_experiment_manifest(REPO_ROOT, commit=COMMIT)
        assert canonical_json(first) == canonical_json(second)

    def test_binds_everything(self, manifest):
        assert manifest["schema"] == MANIFEST_SCHEMA
        assert manifest["source_commit"] == COMMIT
        assert manifest["protocol_identity"] == canonical_protocol().identity()
        assert manifest["num_cells"] == 27
        assert manifest["num_inner_runs"] == 216
        assert len(manifest["cells"]) == 27
        assert len(manifest["schedule_identity"]) == 64
        assert manifest["job_timeout_seconds"] == 600
        assert set(manifest["task_bindings"]) == {"banknote", "spambase", "wdbc"}
        registry = manifest["registry"]
        assert set(registry["tasks"]) == set(review.REGISTRY_TASKS)
        assert registry["tasks"]["comparison_wdbc"]["split"] == "validation"
        for rel, sha in manifest["pinned_file_sha256"].items():
            assert len(sha) == 64, rel
            assert (REPO_ROOT / rel).is_file(), rel
        native = manifest["native"]
        assert native["jax_version"] == "0.4.33"
        assert Path(native["python"]).is_file()

    def test_commit_changes_cell_identities(self):
        other = review.build_experiment_manifest(REPO_ROOT, commit=OTHER_COMMIT)
        assert other["cells"] != review.build_experiment_manifest(
            REPO_ROOT, commit=COMMIT
        )["cells"]


class TestFreeze:
    def test_freeze_once_and_idempotent(self, tmp_path):
        manifest = review.build_experiment_manifest(REPO_ROOT, commit=COMMIT)
        path = review.freeze_experiment_manifest(REPO_ROOT, manifest, runtime_dir=tmp_path)
        assert path.exists()
        again = review.freeze_experiment_manifest(REPO_ROOT, manifest, runtime_dir=tmp_path)
        assert again.read_bytes() == path.read_bytes()

    def test_refuses_different_bytes(self, tmp_path):
        manifest = review.build_experiment_manifest(REPO_ROOT, commit=COMMIT)
        review.freeze_experiment_manifest(REPO_ROOT, manifest, runtime_dir=tmp_path)
        altered = dict(manifest, source_commit=OTHER_COMMIT)
        with pytest.raises(ComparisonManifestError, match="comparison_manifest_exists"):
            review.freeze_experiment_manifest(REPO_ROOT, altered, runtime_dir=tmp_path)

    def test_refuses_symlink(self, tmp_path):
        manifest = review.build_experiment_manifest(REPO_ROOT, commit=COMMIT)
        target = tmp_path / "elsewhere.json"
        target.write_text("{}", encoding="utf-8")
        link = tmp_path / MANIFEST_FILENAME
        link.symlink_to(target)
        with pytest.raises(ComparisonManifestError, match="comparison_manifest_symlinked"):
            review.freeze_experiment_manifest(REPO_ROOT, manifest, runtime_dir=tmp_path)
        assert target.read_text(encoding="utf-8") == "{}"


class TestReviewGate:
    def test_absent_review_refuses(self, runtime):
        with pytest.raises(ComparisonReviewRefused, match="comparison_review_absent"):
            review.check_review_acceptance(
                REPO_ROOT,
                review_path=runtime.RuntimeDir / REVIEW_FILENAME,
                manifest_path=runtime.manifest_path,
            )

    def test_valid_review_accepted(self, runtime):
        review_path = _write_review(runtime.RuntimeDir, runtime.manifest_path)
        payload = review.check_review_acceptance(
            REPO_ROOT, review_path=review_path, manifest_path=runtime.manifest_path
        )
        assert payload["source_commit"] == COMMIT

    def test_commit_mismatch_refuses(self, runtime):
        review_path = _write_review(
            runtime.RuntimeDir, runtime.manifest_path, commit=OTHER_COMMIT
        )
        with pytest.raises(
            ComparisonReviewRefused, match="comparison_review_commit_mismatch"
        ):
            review.check_review_acceptance(
                REPO_ROOT, review_path=review_path, manifest_path=runtime.manifest_path
            )

    def test_dirty_tree_refuses(self, runtime, monkeypatch):
        review_path = _write_review(runtime.RuntimeDir, runtime.manifest_path)
        monkeypatch.setattr(review, "_git_porcelain", lambda root: " M dirty.py\n")
        with pytest.raises(ComparisonReviewRefused, match="comparison_tree_dirty"):
            review.check_review_acceptance(
                REPO_ROOT, review_path=review_path, manifest_path=runtime.manifest_path
            )

    def test_tracked_review_refuses(self, runtime, monkeypatch):
        review_path = _write_review(runtime.RuntimeDir, runtime.manifest_path)
        monkeypatch.setattr(review, "_git_ls_files_tracked", lambda root, rel: True)
        with pytest.raises(ComparisonReviewRefused, match="comparison_review_tracked"):
            review.check_review_acceptance(
                REPO_ROOT, review_path=review_path, manifest_path=runtime.manifest_path
            )

    def test_manifest_bytes_mismatch_refuses(self, runtime):
        review_path = _write_review(runtime.RuntimeDir, runtime.manifest_path)
        runtime.manifest_path.write_text("{}", encoding="utf-8")
        with pytest.raises(
            ComparisonReviewRefused, match="comparison_review_manifest_mismatch"
        ):
            review.check_review_acceptance(
                REPO_ROOT, review_path=review_path, manifest_path=runtime.manifest_path
            )

    def test_manifest_drift_refuses(self, runtime):
        # Frozen manifest is internally consistent and review-hash-matching,
        # but no longer derivable from the current tree.
        altered = json.loads(runtime.manifest_path.read_text(encoding="utf-8"))
        key = sorted(altered["cells"])[0]
        altered["cells"][key] = "0" * 64
        runtime.manifest_path.write_text(canonical_json(altered) + "\n", encoding="utf-8")
        review_path = _write_review(runtime.RuntimeDir, runtime.manifest_path)
        with pytest.raises(ComparisonReviewRefused, match="comparison_manifest_drift"):
            review.check_review_acceptance(
                REPO_ROOT, review_path=review_path, manifest_path=runtime.manifest_path
            )

    def test_malformed_review_refuses(self, runtime):
        review_path = runtime.RuntimeDir / REVIEW_FILENAME
        review_path.write_text(json.dumps({"schema": REVIEW_SCHEMA}), encoding="utf-8")
        with pytest.raises(ComparisonReviewRefused, match="comparison_review_malformed"):
            review.check_review_acceptance(
                REPO_ROOT, review_path=review_path, manifest_path=runtime.manifest_path
            )

    def test_symlinked_review_refuses(self, runtime, tmp_path):
        target = tmp_path / "foreign.json"
        target.write_text("{}", encoding="utf-8")
        link = runtime.RuntimeDir / REVIEW_FILENAME
        link.symlink_to(target)
        with pytest.raises(ComparisonReviewRefused, match="comparison_review_symlinked"):
            review.check_review_acceptance(
                REPO_ROOT, review_path=link, manifest_path=runtime.manifest_path
            )
        assert json.loads(target.read_text(encoding="utf-8")) == {}


class TestReviewRequirements:
    def test_describe_never_writes(self, runtime, monkeypatch):
        monkeypatch.setattr(
            review, "manifest_path_for", lambda root: runtime.manifest_path
        )
        req = review.describe_review_requirements(REPO_ROOT)
        assert req["schema"] == REVIEW_SCHEMA
        assert req["source_commit"] == COMMIT
        assert req["experiment_manifest_sha256"] == hashlib.sha256(
            runtime.manifest_path.read_bytes()
        ).hexdigest()
        assert not (runtime.RuntimeDir / REVIEW_FILENAME).exists()
