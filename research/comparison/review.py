"""Immutable experiment manifest and the supervisor review gate (EXP-007).

The FULL immutable experiment manifest binds every input the live
experiment depends on: the frozen protocol, the exact clean source commit,
the data manifest and task bindings, the trusted evaluator registry, the
frozen schedule identity, the pinned native interpreter/JAX identity, the
deployment/mount source files and the approved budgets. It is written ONCE
by ``prepare`` into the ignored runtime directory; any drift (working-tree
mutation, manifest edit, bound-file change) is detected by re-derivation.

The supervisor review is an IGNORED runtime file (never tracked, never
written by the implementation) that binds the exact source commit and the
exact frozen manifest bytes. ``check_review_acceptance`` is the live gate:
it runs BEFORE any auth, ledger creation, worker submission or model
generation, and refuses unless the review is present, untracked, strictly
well-formed, and both bindings re-verify against the current tree.

Review file format (written ONLY by the independent supervisor):

.. code-block:: json

    {
      "schema": "istratde-comparison-supervisor-review.v1",
      "source_commit": "<40-hex git HEAD>",
      "experiment_manifest_sha256": "<64-hex sha256 of manifest file bytes>",
      "accepted_at": "<ISO-8601 string, informational>",
      "reviewer": "<string, informational>"
    }

``describe_review_requirements`` computes the two binding values from the
frozen manifest so the supervisor can construct the file; it never writes.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.comparison import deploy
from research.comparison.data import load_manifest, load_task_bindings, manifest_identity
from research.comparison.native import NATIVE_ENV_FILE, load_native_python
from research.comparison.protocol import (
    TASKS_BY_ID,
    ModelProfile,
    canonical_json,
    canonical_protocol,
)
from research.comparison.schedule import (
    JOB_TIMEOUT_SECONDS,
    build_schedule,
    schedule_identity,
)
from research.comparison.specs import iter_cells, iter_inner_runs
from research.evaluator.service import EvaluatorRegistry

#: Runtime layout (ignored .runtime/, never committed). Single source of
#: truth; runner.py re-exports these names for its existing callers.
RUNTIME_DIR = Path(".runtime") / "comparison-istratde"
LEDGER_FILENAME = "comparison-ledger.json"

MANIFEST_SCHEMA = "istratde-comparison-experiment.v1"
MANIFEST_FILENAME = "experiment-manifest.v1.json"
REVIEW_SCHEMA = "istratde-comparison-supervisor-review.v1"
REVIEW_FILENAME = "supervisor-review.json"

#: Tracked files whose bytes are bound into the manifest in ADDITION to the
#: clean-tree commit pin: the deployment/mount/data/registry surface the
#: live experiment depends on. Drift in any of them invalidates the review.
PINNED_FILES: tuple[str, ...] = (
    "deploy/rsi-trustworthy/docker-compose.yaml",
    "deploy/rsi-trustworthy/config/sandbox_config.json",
    "deploy/rsi-trustworthy/nginx/nginx-main.conf",
    "deploy/rsi-trustworthy/nginx/nginx.conf",
    "deploy/rsi-trustworthy/worker/minimal_worker.py",
    "deploy/rsi-trustworthy/worker/exec_trampoline.py",
    "research/evaluator/registry.v1.json",
    "research/comparison/data-manifest.v1.json",
    "research/comparison/native-env.json",
    "research/comparison/native_driver.py",
    "OpenMLE-Evo/tts_search/configs/litellm/trustworthy_kimi.yaml",
)

#: Trusted registry tasks the experiment depends on (search phase).
REGISTRY_TASKS: tuple[str, ...] = (
    "hello_synth",
    "comparison_banknote",
    "comparison_wdbc",
    "comparison_spambase",
)

_COMMIT_HEX_LEN = 40
_SHA256_HEX_LEN = 64
_HEX_DIGITS = frozenset("0123456789abcdef")

#: Stack image name:tag pins (compose declares the same values; the manifest
#: additionally freezes the ACTUAL image IDs so a same-tag changed image is
#: detected by re-derivation). Keys are compose service names.
STACK_IMAGE_PINS: dict[str, str] = {
    "nginx": "docker.m.daocloud.io/library/nginx:1.25.5",
    "api": "ghcr.io/lifeissosolong/openmle-sandbox-controller-api:1.0.0",
    "dispatcher": "ghcr.io/lifeissosolong/openmle-sandbox-controller-dispatcher:1.0.0",
    "worker": "ghcr.io/lifeissosolong/openmle-sandbox-worker:1.0.0",
    "postgres": "docker.m.daocloud.io/library/postgres:15.14",
    "redis": "docker.m.daocloud.io/library/redis:7.4.6",
}


def _docker_image_digests() -> dict[str, str]:
    """ACTUAL local image IDs for the pinned tags (seam for tests)."""
    digests: dict[str, str] = {}
    for service, tag in STACK_IMAGE_PINS.items():
        try:
            proc = subprocess.run(
                ["docker", "image", "inspect", tag, "--format", "{{.Id}}"],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ComparisonManifestError("docker_cli_unavailable") from exc
        if proc.returncode != 0:
            raise ComparisonManifestError(f"stack_image_missing:{tag}")
        digest = proc.stdout.strip()
        if len(digest) != 71 or not digest.startswith("sha256:") or any(
            ch not in _HEX_DIGITS for ch in digest[7:]
        ):
            raise ComparisonManifestError(f"stack_image_id_malformed:{tag}")
        digests[service] = digest
    return digests


def _native_runtime_identity(python: Path) -> tuple[str, str]:
    """ACTUAL pinned interpreter identity: JAX version + backend (CPU pin)."""
    env = dict(os.environ)
    env["JAX_PLATFORM_NAME"] = "cpu"  # same pin the driver applies
    try:
        proc = subprocess.run(
            [
                str(python),
                "-c",
                "import jax; print(jax.__version__); print(jax.default_backend())",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ComparisonManifestError("native_jax_unavailable") from exc
    if proc.returncode != 0:
        raise ComparisonManifestError("native_jax_unavailable")
    lines = proc.stdout.split()
    if len(lines) < 2:
        raise ComparisonManifestError("native_identity_malformed")
    return lines[0], lines[1]


class ComparisonManifestError(RuntimeError):
    """Manifest build/freeze/verification failure; the message IS the rule."""


class ComparisonReviewRefused(RuntimeError):
    """Live comparison execution is refused; the message IS the rule."""


def _require_hex(value: Any, length: int, rule: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(ch not in _HEX_DIGITS for ch in value)
    ):
        raise ComparisonReviewRefused(rule)
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_porcelain(repo_root: Path) -> str:
    """Raw ``git status --porcelain`` output (seam for tests)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ComparisonManifestError("comparison_git_unavailable") from exc
    if proc.returncode != 0:
        raise ComparisonManifestError("comparison_git_unavailable")
    return proc.stdout


def require_clean_head(repo_root: Path) -> str:
    """Exact clean git HEAD; any tracked or untracked mutation refuses."""
    head = deploy.git_head_commit(repo_root)
    if (
        len(head) != _COMMIT_HEX_LEN
        or any(ch not in _HEX_DIGITS for ch in head)
    ):
        raise ComparisonManifestError("source_commit_invalid")
    if _git_porcelain(repo_root).strip():
        raise ComparisonManifestError("comparison_tree_dirty")
    return head


def _write_durable(path: Path, text: str) -> None:
    """Atomic tmp-file + replace + fsync of the file AND parent directory."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def build_experiment_manifest(
    repo_root: Path, *, commit: str | None = None
) -> dict[str, Any]:
    """Derive the full immutable experiment manifest from current bytes.

    Deterministic: identical repo state yields byte-identical canonical
    JSON (no timestamps or host-volatile fields). With ``commit=None`` the
    exact CLEAN working-tree HEAD is required and bound; tests may pass an
    explicit commit to exercise derivation without git state.
    """
    repo_root = Path(repo_root)
    source_commit = commit if commit is not None else require_clean_head(repo_root)
    if (
        len(source_commit) != _COMMIT_HEX_LEN
        or any(ch not in _HEX_DIGITS for ch in source_commit)
    ):
        raise ComparisonManifestError("source_commit_invalid")
    protocol = canonical_protocol()
    bindings = load_task_bindings()
    cells = iter_cells(bindings, source_commit)
    runs = iter_inner_runs(cells)
    positions = build_schedule(cells)
    profile = ModelProfile()

    registry_rel = "research/evaluator/registry.v1.json"
    registry = EvaluatorRegistry.load(repo_root / registry_rel)
    registry_tasks: dict[str, Any] = {}
    for task_id in REGISTRY_TASKS:
        spec = registry.task(task_id)
        registry_tasks[task_id] = {
            "split": spec.split,
            "metric": spec.metric,
            "direction": spec.direction.value,
            "answer_sha256": spec.answer_sha256,
            "worker_data_public_path": spec.worker_data_public_path,
        }

    data_manifest_rel = "research/comparison/data-manifest.v1.json"
    data_manifest = load_manifest()
    native_env_rel = "research/comparison/native-env.json"
    try:
        native_env = json.loads((repo_root / NATIVE_ENV_FILE).read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise ComparisonManifestError("native_env_malformed") from exc
    native_python = load_native_python(repo_root)
    jax_actual, backend_actual = _native_runtime_identity(native_python)
    if jax_actual != "0.4.33" or backend_actual != "cpu":
        raise ComparisonManifestError(
            f"native_identity_mismatch:{jax_actual}/{backend_actual}"
        )

    pinned: dict[str, str] = {}
    for rel in PINNED_FILES:
        path = repo_root / rel
        if path.is_symlink() or not path.is_file():
            raise ComparisonManifestError(f"manifest_pin_missing:{rel}")
        pinned[rel] = _sha256_file(path)

    return {
        "schema": MANIFEST_SCHEMA,
        "source_commit": source_commit,
        "protocol": protocol.to_dict(),
        "protocol_identity": protocol.identity(),
        "model_profile": profile.to_dict(),
        "model_profile_identity": profile.identity(),
        "data_manifest": data_manifest_rel,
        "data_manifest_sha256": pinned[data_manifest_rel],
        "data_manifest_identity": manifest_identity(data_manifest),
        "task_bindings": {
            task_id: bindings[task_id].identity() for task_id in sorted(TASKS_BY_ID)
        },
        "cells": {cell.cell_key(): cell.identity() for cell in cells},
        "num_cells": len(cells),
        "num_inner_runs": len(runs),
        "schedule_identity": schedule_identity(positions, source_commit),
        "num_positions": len(positions),
        "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
        "registry": {
            "path": registry_rel,
            "sha256": pinned[registry_rel],
            "registry_version": registry.registry_version,
            "tasks": registry_tasks,
        },
        "native": {
            "env_file": native_env_rel,
            "env_file_sha256": pinned[native_env_rel],
            "python": str(native_python),
            "jax_version": native_env.get("jax_version"),
            "jax_platform": native_env.get("jax_platform"),
            "jax_version_actual": jax_actual,
            "backend_actual": backend_actual,
        },
        "stack_images": {
            "pins": dict(STACK_IMAGE_PINS),
            "image_ids": _docker_image_digests(),
        },
        "pinned_file_sha256": pinned,
    }


def freeze_experiment_manifest(
    repo_root: Path, manifest: Mapping[str, Any], *, runtime_dir: Path | None = None
) -> Path:
    """Write the manifest ONCE into the runtime dir (symlink-safe).

    Identical existing bytes are accepted (idempotent re-prepare of an
    unchanged experiment); different bytes or a symlinked target refuse —
    a frozen manifest is never replaced.
    """
    base = Path(runtime_dir) if runtime_dir is not None else Path(repo_root) / RUNTIME_DIR
    path = base / MANIFEST_FILENAME
    text = canonical_json(dict(manifest)) + "\n"
    if path.is_symlink():
        raise ComparisonManifestError("comparison_manifest_symlinked")
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return path
        raise ComparisonManifestError("comparison_manifest_exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_durable(path, text)
    return path


def load_frozen_manifest(
    repo_root: Path, *, runtime_dir: Path | None = None
) -> dict[str, Any]:
    base = Path(runtime_dir) if runtime_dir is not None else Path(repo_root) / RUNTIME_DIR
    path = base / MANIFEST_FILENAME
    if path.is_symlink() or not path.exists():
        raise ComparisonManifestError("comparison_manifest_missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ComparisonManifestError("comparison_manifest_malformed") from exc
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise ComparisonManifestError("comparison_manifest_malformed")
    return payload


def manifest_path_for(repo_root: Path) -> Path:
    return Path(repo_root) / RUNTIME_DIR / MANIFEST_FILENAME


def verify_frozen_manifest_binding(
    manifest: Mapping[str, Any],
    *,
    source_commit: str,
    schedule_identity_value: str,
    cells: Sequence[Any],
    bindings: Mapping[str, Any],
    repo_root: Path,
) -> None:
    """OFFLINE binding verification of the frozen experiment manifest.

    Re-proves every frozen pin against the actual current bytes — protocol
    and budget pins, model profile, data manifest bytes+identity, task
    bindings, roster, registry file and per-task pins, native env file and
    EVERY pinned source file — so a copied/mutually edited manifest never
    redefines the experiment. No Docker/network/model use: the runtime
    image/native-interpreter identity probes remain the preflight gate;
    here stack_images.pins is checked as a static pin only. Missing
    required pins fail closed (never silently optional).
    """
    repo_root = Path(repo_root)
    proto = canonical_protocol()
    profile = ModelProfile()
    if manifest.get("source_commit") != source_commit:
        raise ComparisonManifestError("manifest_source_commit_mismatch")
    if manifest.get("schedule_identity") != schedule_identity_value:
        raise ComparisonManifestError("manifest_schedule_mismatch")
    expected_cells = {cell.cell_key(): cell.identity() for cell in cells}
    if manifest.get("cells") != expected_cells:
        raise ComparisonManifestError("manifest_cells_mismatch")
    if (
        manifest.get("num_cells") != proto.num_cells
        or manifest.get("num_inner_runs") != proto.num_inner_runs
        or manifest.get("num_positions") != proto.num_inner_runs
        or manifest.get("job_timeout_seconds") != JOB_TIMEOUT_SECONDS
    ):
        raise ComparisonManifestError("manifest_roster_mismatch")
    if (
        manifest.get("protocol") != proto.to_dict()
        or manifest.get("protocol_identity") != proto.identity()
    ):
        raise ComparisonManifestError("manifest_protocol_mismatch")
    if (
        manifest.get("model_profile") != profile.to_dict()
        or manifest.get("model_profile_identity") != profile.identity()
    ):
        raise ComparisonManifestError("manifest_model_profile_mismatch")
    if manifest.get("task_bindings") != {
        task_id: bindings[task_id].identity() for task_id in sorted(bindings)
    }:
        raise ComparisonManifestError("manifest_task_bindings_mismatch")

    def _pinned_file(rel: Any, expected: Any, rule: str) -> bytes:
        if not isinstance(rel, str) or not isinstance(expected, str):
            raise ComparisonManifestError(rule)
        path = repo_root / rel
        if path.is_symlink() or not path.is_file():
            raise ComparisonManifestError(f"{rule}:{rel}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ComparisonManifestError(f"{rule}:{rel}")
        return data

    data_bytes = _pinned_file(
        manifest.get("data_manifest"),
        manifest.get("data_manifest_sha256"),
        "manifest_data_manifest_mismatch",
    )
    try:
        data_manifest = json.loads(data_bytes)
    except ValueError as exc:
        raise ComparisonManifestError("manifest_data_manifest_mismatch") from exc
    if manifest.get("data_manifest_identity") != manifest_identity(data_manifest):
        raise ComparisonManifestError("manifest_data_manifest_mismatch")
    registry_pin = manifest.get("registry")
    if not isinstance(registry_pin, dict):
        raise ComparisonManifestError("manifest_registry_mismatch")
    registry_bytes = _pinned_file(
        registry_pin.get("path"),
        registry_pin.get("sha256"),
        "manifest_registry_mismatch",
    )
    try:
        registry_payload = json.loads(registry_bytes)
    except ValueError as exc:
        raise ComparisonManifestError("manifest_registry_mismatch") from exc
    registry_tasks = registry_payload.get("tasks")
    pinned_tasks = registry_pin.get("tasks")
    if not isinstance(registry_tasks, dict) or not isinstance(pinned_tasks, dict):
        raise ComparisonManifestError("manifest_registry_mismatch")
    # The three comparison search tasks are REQUIRED pins; every pinned
    # task must match the actual registry file, and comparison tasks must
    # match the frozen task bindings (split/metric/answer identity).
    required_tasks = {f"comparison_{task_id}" for task_id in bindings}
    if not required_tasks <= set(pinned_tasks):
        raise ComparisonManifestError("manifest_registry_tasks_missing")
    for task_id, pin in pinned_tasks.items():
        actual = registry_tasks.get(task_id)
        if not isinstance(actual, dict) or not isinstance(pin, dict):
            raise ComparisonManifestError(f"manifest_registry_task_mismatch:{task_id}")
        for field in ("split", "metric", "answer_sha256"):
            if actual.get(field) != pin.get(field):
                raise ComparisonManifestError(
                    f"manifest_registry_task_mismatch:{task_id}"
                )
        if task_id in required_tasks:
            binding = bindings[task_id[len("comparison_"):]]
            if (
                pin.get("answer_sha256") != binding.validation_labels_sha256
                or pin.get("split") != "validation"
                or pin.get("metric") != "accuracy"
            ):
                raise ComparisonManifestError(
                    f"manifest_registry_task_mismatch:{task_id}"
                )
    native_pin = manifest.get("native")
    if not isinstance(native_pin, dict):
        raise ComparisonManifestError("manifest_native_mismatch")
    _pinned_file(
        native_pin.get("env_file"),
        native_pin.get("env_file_sha256"),
        "manifest_native_mismatch",
    )
    stack_images = manifest.get("stack_images")
    if not isinstance(stack_images, dict) or stack_images.get("pins") != dict(
        STACK_IMAGE_PINS
    ):
        raise ComparisonManifestError("manifest_stack_pins_mismatch")
    pins = manifest.get("pinned_file_sha256")
    if not isinstance(pins, dict) or not set(PINNED_FILES) <= set(pins):
        raise ComparisonManifestError("manifest_pins_missing")
    for rel, expected in pins.items():
        _pinned_file(rel, expected, "manifest_pinned_file_mismatch")


def review_path_for(repo_root: Path) -> Path:
    return Path(repo_root) / RUNTIME_DIR / REVIEW_FILENAME


def _git_ls_files_tracked(repo_root: Path, relative: Path) -> bool:
    """True when git tracks the path (seam for tests)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", "--", str(relative)],
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ComparisonReviewRefused("comparison_git_unavailable") from exc
    return proc.returncode == 0


def _assert_untracked(repo_root: Path, review_path: Path) -> None:
    try:
        rel = review_path.relative_to(repo_root)
    except ValueError as exc:
        raise ComparisonReviewRefused("comparison_review_path_foreign") from exc
    if _git_ls_files_tracked(repo_root, rel):
        # A TRACKED approval is a forgery vector: the review only ever lives
        # in the ignored runtime tree.
        raise ComparisonReviewRefused("comparison_review_tracked")


def check_review_acceptance(
    repo_root: Path,
    *,
    review_path: Path | None = None,
    manifest_path: Path | None = None,
) -> Mapping[str, Any]:
    """The live gate. Returns the review payload; refuses otherwise.

    Every refusal happens BEFORE any auth, ledger creation, worker
    submission or model generation. Requirements: review file present, not
    symlinked, not git-tracked, strictly well-formed; its commit equals the
    exact clean HEAD; the frozen manifest bytes hash-match the review; the
    manifest's own commit/protocol pins match; and a FULL re-derivation of
    the manifest from the current tree reproduces it exactly (any drift in
    bound data/registry/deployment files refuses).
    """
    repo_root = Path(repo_root)
    review_path = review_path or review_path_for(repo_root)
    manifest_path = manifest_path or manifest_path_for(repo_root)
    if review_path.is_symlink():
        raise ComparisonReviewRefused("comparison_review_symlinked")
    if not review_path.exists():
        raise ComparisonReviewRefused("comparison_review_absent")
    try:
        payload = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ComparisonReviewRefused("comparison_review_malformed") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "source_commit",
        "experiment_manifest_sha256",
        "accepted_at",
        "reviewer",
    }:
        raise ComparisonReviewRefused("comparison_review_malformed")
    if payload["schema"] != REVIEW_SCHEMA:
        raise ComparisonReviewRefused("comparison_review_malformed")
    if not isinstance(payload["accepted_at"], str) or not payload["accepted_at"]:
        raise ComparisonReviewRefused("comparison_review_malformed")
    if not isinstance(payload["reviewer"], str) or not payload["reviewer"]:
        raise ComparisonReviewRefused("comparison_review_malformed")
    review_commit = _require_hex(
        payload["source_commit"], _COMMIT_HEX_LEN, "comparison_review_malformed"
    )
    review_manifest_sha = _require_hex(
        payload["experiment_manifest_sha256"],
        _SHA256_HEX_LEN,
        "comparison_review_malformed",
    )
    _assert_untracked(repo_root, review_path)
    try:
        head = require_clean_head(repo_root)
    except ComparisonManifestError as exc:
        raise ComparisonReviewRefused(str(exc)) from exc
    if head != review_commit:
        raise ComparisonReviewRefused("comparison_review_commit_mismatch")
    if manifest_path.is_symlink():
        raise ComparisonReviewRefused("comparison_manifest_symlinked")
    if not manifest_path.exists():
        raise ComparisonReviewRefused("comparison_manifest_missing")
    if _sha256_file(manifest_path) != review_manifest_sha:
        raise ComparisonReviewRefused("comparison_review_manifest_mismatch")
    try:
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ComparisonReviewRefused("comparison_manifest_malformed") from exc
    if not isinstance(frozen, dict) or frozen.get("schema") != MANIFEST_SCHEMA:
        raise ComparisonReviewRefused("comparison_manifest_malformed")
    if frozen.get("source_commit") != head:
        raise ComparisonReviewRefused("comparison_manifest_commit_mismatch")
    if frozen.get("protocol_identity") != canonical_protocol().identity():
        raise ComparisonReviewRefused("comparison_manifest_protocol_mismatch")
    rebuilt = build_experiment_manifest(repo_root)
    if rebuilt != frozen:
        raise ComparisonReviewRefused("comparison_manifest_drift")
    return payload


def describe_review_requirements(
    repo_root: Path, *, manifest_path: Path | None = None
) -> dict[str, Any]:
    """What the supervisor's review must bind; NEVER writes the review."""
    repo_root = Path(repo_root)
    manifest_path = manifest_path or manifest_path_for(repo_root)
    if manifest_path.is_symlink() or not manifest_path.exists():
        raise ComparisonManifestError("comparison_manifest_missing")
    return {
        "schema": REVIEW_SCHEMA,
        "path": str(review_path_for(repo_root).relative_to(repo_root)),
        "source_commit": deploy.git_head_commit(repo_root),
        "experiment_manifest_sha256": _sha256_file(manifest_path),
        "note": (
            "written only by the independent supervisor after reviewing the "
            "exact commit and the frozen manifest; the implementation never "
            "creates this file"
        ),
    }
