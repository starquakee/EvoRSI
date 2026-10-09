"""US-009 real-acceptance constants and the live-mode supervisor review gate.

Stdlib-only on purpose: this module is imported by the parent controller
(legacy JAX venv), the inner Evo child (OpenMLE-Evo venv) and the offline
research tests (.venv-research).

Live-mode rule (supervisor contract): the real acceptance may touch the
persistent ledger, credentials or the model ONLY when an ignored review file
(``.runtime/us009-runner-review.json``) binds the CURRENT git HEAD with an
explicit approval. Offline/preflight paths never require that file and never
create the ledger. The review file is written by the supervisor after
reviewing the committed runner; the runner never creates it.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from research.contracts.budget import BudgetLimits

# --- accepted review-file contract (live gate) ------------------------------
RUNNER_REVIEW_SCHEMA = "us009-runner-review.v1"
RUNNER_REVIEW_FILENAME = "us009-runner-review.json"

# --- runtime layout (all under ignored .runtime/, never committed) ----------
RUNTIME_DIR = Path(".runtime") / "acceptance-us009"
LEDGER_FILENAME = "budget-ledger.json"
RUN_INDEX_FILENAME = "run-index.json"
SUMMARY_FILENAME = "acceptance-summary.json"
CHECKPOINT_FILENAME = "acceptance-checkpoint.json"
AUTH_FILE = Path(".runtime") / "rsi-trustworthy" / "auth.env"
PREFLIGHT_VECTORS_FILE = Path(".runtime") / "outer-ask-preflight.json"

# --- acceptance pins (PRD US-009 + batch-3 supervisor guidance) -------------
MODEL_ID = "k3"
MODEL_BASE_URL = "https://api.kimi.com/coding/v1"
API_KEY_PLACEHOLDER = "runtime"  # configs never carry a real token
GATEWAY_URL = "http://127.0.0.1:6581"
TASK_ID = "hello_synth"
METRIC = "accuracy"  # maximize (research.contracts.results.metric_direction)
SEED = 42
POP_SIZE = 4
SANDBOX_DATA_DIR = "/mnt/rsi_data/hello_synth"
RESOURCE_TYPE = "cpu"
JOB_TIMEOUT_SECONDS = 600

# Evo config files (repo-relative), pinned by US-007/US-008.
EVO_CONFIG_DIR = Path("OpenMLE-Evo") / "tts_search" / "configs"
VALIDATION_EXPERIMENT_YAML = EVO_CONFIG_DIR / "experiment" / "trustworthy_validation.yaml"
LITELLM_YAML = EVO_CONFIG_DIR / "litellm" / "trustworthy_kimi.yaml"

# Public synthetic inputs provisioned read-only in US-008 (sha256 pinned).
PUBLIC_DATA_HASHES = {
    "tasks/hello_synth/data/public/train.csv":
        "448e69962d82036c0e1b4aff7b9b3397bedbb4537c2bb1e9d2830f9fa3b29976",
    "tasks/hello_synth/data/public/test.csv":
        "b0ac9e8732f6be47ae3d261f88f8a25bfb1083e4fb6bda32c983b195f55e2995",
    "tasks/hello_synth/data/public/sample_submission.csv":
        "aeff9e843b62f64a93b97a6e1921abab9d82b41de8078bca0a900cfa2845ff5c",
}

EVALUATOR_REGISTRY = Path("research") / "evaluator" / "registry.v1.json"

#: Trusted prediction snapshots: container path prefix <-> host runtime root
#: (deploy/rsi-trustworthy compose mounts .runtime/rsi-trustworthy/storage at
#: /mnt/rsi_storage in the api/dispatcher services).
CONTAINER_STORAGE_PREFIX = "/mnt/rsi_storage/"
DISPATCHER_STORAGE_ROOT = Path(".runtime") / "rsi-trustworthy" / "storage"


def public_data_identity() -> str:
    """Content identity of the pinned public-data set (name + sha256 pairs)."""
    import hashlib

    canonical = json.dumps(PUBLIC_DATA_HASHES, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def registry_identity(repo_root: Path) -> str:
    """Content identity (sha256) of the trusted evaluator registry file."""
    import hashlib

    return hashlib.sha256((repo_root / EVALUATOR_REGISTRY).read_bytes()).hexdigest()


def acceptance_budget_limits() -> BudgetLimits:
    """The ONE acceptance ledger's immutable limits (200k/30/90min)."""
    return BudgetLimits(
        max_tokens=200_000,
        max_requests=30,
        max_elapsed_seconds=5_400.0,
    )


class RunnerReviewRefused(RuntimeError):
    """Live mode is refused; the message IS the machine-readable rule."""


def evaluate_runner_review(review: Mapping[str, Any], *, head_commit: str) -> None:
    """Pure check of a parsed review payload against the implementation commit.

    Raises :class:`RunnerReviewRefused` naming the first violated rule.
    """
    if not isinstance(review, Mapping):
        raise RunnerReviewRefused("runner_review_not_an_object")
    if review.get("schema") != RUNNER_REVIEW_SCHEMA:
        raise RunnerReviewRefused("runner_review_schema_mismatch")
    commit = review.get("implementation_commit")
    if not isinstance(commit, str) or not commit or commit != head_commit:
        raise RunnerReviewRefused("runner_review_commit_mismatch")
    if review.get("approve_real_acceptance") is not True:
        raise RunnerReviewRefused("runner_review_not_approved")


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerReviewRefused("runner_review_git_unavailable") from exc


def git_head_commit(repo_root: Path) -> str:
    return _git(repo_root, "rev-parse", "HEAD")


def git_tree_dirty(repo_root: Path) -> bool:
    """Tracked AND untracked changes count: an untracked new module is
    unreviewed implementation, so the reviewed tree must be fully clean
    (ignored runtime files are excluded by git itself)."""
    return bool(_git(repo_root, "status", "--porcelain"))


def check_runner_review(repo_root: Path, review_path: Path | None = None) -> None:
    """IO wrapper: load the ignored review file and bind it to HEAD.

    Live mode calls this BEFORE opening the ledger, reading credentials or
    any model/sandbox activity. Any failure raises RunnerReviewRefused and
    nothing else has been touched.
    """
    path = review_path or (repo_root / ".runtime" / RUNNER_REVIEW_FILENAME)
    try:
        review = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunnerReviewRefused("runner_review_unavailable") from exc
    evaluate_runner_review(review, head_commit=git_head_commit(repo_root))
    if git_tree_dirty(repo_root):
        raise RunnerReviewRefused("runner_review_dirty_tree")
