"""EXP-003: pin the worker compose mount contract (offline, read-only).

Parses `docker compose config --format json` (no contact with the running
stack) and proves the worker mount plan matches the narrowed-grant design:
exactly the four pinned search-phase leaves plus the three EXP-006
final-phase leaves (sourced from the ignored runtime publication root,
which holds final bytes only after the verified all-cell barrier), all
read-only, both bundled labeled
WDBC assets masked via read-only /dev/null binds, the scratch tree as the
only writable share, and no mount exposing the evaluator/sealed trees.
Also cross-checks the evaluator registry's worker_data_public_path mapping
against the actual mounted targets.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from research.evaluator.service import EvaluatorRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE = REPO_ROOT / "deploy" / "rsi-trustworthy" / "docker-compose.yaml"

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker CLI unavailable"
)


@pytest.fixture(scope="module")
def worker_volumes() -> list[dict]:
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "config", "--format", "json"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[:300]
    config = json.loads(result.stdout)
    return config["services"]["worker"]["volumes"]


def test_exact_task_data_leaves_mounted_read_only(worker_volumes) -> None:
    expected = {
        "/mnt/rsi_data/hello_synth/data/public",
        "/mnt/rsi_data/comparison/banknote/data/public",
        "/mnt/rsi_data/comparison/wdbc/data/public",
        "/mnt/rsi_data/comparison/spambase/data/public",
        # EXP-006 final-phase leaves: empty until the verified all-cell
        # barrier publishes refit/final-feature bytes (labels NEVER here).
        "/mnt/rsi_data/comparison/banknote/data/final",
        "/mnt/rsi_data/comparison/wdbc/data/final",
        "/mnt/rsi_data/comparison/spambase/data/final",
    }
    data_mounts = {
        v["target"]: v for v in worker_volumes if str(v["target"]).startswith("/mnt/rsi_data")
    }
    assert set(data_mounts) == expected
    for target, mount in data_mounts.items():
        assert mount.get("read_only") is True, target
        # Sources are exactly the per-task data/public|final directories,
        # never a parent (which would expose sealed/private siblings).
        source = Path(mount["source"])
        assert source.name in ("public", "final") and source.parent.name == "data", source
        assert "sealed" not in source.parts and "evaluator" not in source.parts
        if source.name == "final":
            # EXP-006: final-phase bytes are published ONLY into the
            # ignored runtime publication root after the verified barrier;
            # tracked search data/registry stay byte-identical. No final
            # bytes may exist before publication: the source holds ONLY the
            # pre-created placeholder README.md (EXP-007 keeps Docker from
            # owning the bind source) or, after a verified publication, the
            # published final CSVs.
            runtime_root = REPO_ROOT / ".runtime" / "comparison-istratde" / "final-publication"
            assert runtime_root in source.parents, source
            assert not str(source).startswith(str(REPO_ROOT / "tasks")), source
            if source.exists():
                assert {p.name for p in source.iterdir()} <= {
                    "README.md",
                    "train.csv",
                    "test.csv",
                    "sample_submission.csv",
                }, source


def test_bundled_wdbc_assets_masked_read_only(worker_volumes) -> None:
    masked = {
        v["target"]: v
        for v in worker_volumes
        if str(v["target"]).endswith("sklearn/datasets/data/breast_cancer.csv")
    }
    assert set(masked) == {
        "/usr/local/lib/python3.10/dist-packages/sklearn/datasets/data/breast_cancer.csv",
        "/opt/python3.11/lib/python3.11/site-packages/sklearn/datasets/data/breast_cancer.csv",
    }
    for target, mount in masked.items():
        assert mount.get("read_only") is True, target
        assert mount["source"] == "/dev/null", target


def test_scratch_is_the_only_writable_share(worker_volumes) -> None:
    writable = [v["target"] for v in worker_volumes if not v.get("read_only")]
    assert writable == ["/mnt/local_sandbox_workdir"]


def test_no_mount_exposes_evaluator_or_sealed_trees(worker_volumes) -> None:
    for mount in worker_volumes:
        source = Path(mount["source"]).resolve()
        parts = source.parts
        assert "evaluator" not in parts, source
        assert "sealed" not in parts, source
        # The repository research/ tree must never enter the worker at all.
        assert REPO_ROOT / "research" not in source.parents, source


def test_registry_grant_mapping_matches_compose_targets(worker_volumes) -> None:
    registry = EvaluatorRegistry.load(REPO_ROOT / "research" / "evaluator" / "registry.v1.json")
    mounted = {v["target"] for v in worker_volumes}
    for task_id in ("hello_synth", "comparison_banknote", "comparison_wdbc", "comparison_spambase"):
        pinned = registry.task(task_id).worker_data_public_path
        assert pinned is not None, task_id
        assert pinned in mounted, f"{task_id}: registry grant {pinned} not mounted"
        mount = next(v for v in worker_volumes if v["target"] == pinned)
        assert mount.get("read_only") is True
