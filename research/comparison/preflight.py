"""Prepare, preflight and report logic for the comparison (EXP-007).

``prepare`` is strictly offline: it derives and freezes the full immutable
experiment manifest at the exact clean HEAD and initializes the frozen
orchestration state (schedule + all knowable theta vectors via the pinned
native provider). It creates NO ledger, reads NO credential and makes NO
model call — those stay gated behind the supervisor review (review.py).

``preflight`` re-verifies every offline gate (manifest re-derivation,
data/registry pins, native replay determinism, orchestration state,
historic US-009 ledger integrity, static mount contract, absence of live
artifacts) and persists evidence. The supervisor review gate is REPORTED
(closed while the ignored review file is absent) — never opened here.

``report`` is read-only: it revalidates the persisted experiment state and
emits the per-cell matrix plus ledger totals. EXP-008 extends it with the
predeclared bootstrap analysis and plots.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from research.comparison import review
from research.comparison.budget import ReadOnlyComparisonLedger
from research.comparison.data import load_task_bindings
from research.comparison.native import NativeProvider, SubprocessNativeProvider
from research.comparison.orchestrate import Orchestrator
from research.comparison.protocol import (
    SEARCH_SEEDS,
    TASKS_BY_ID,
    canonical_json,
)
from research.comparison.review import (
    ComparisonManifestError,
    build_experiment_manifest,
    load_frozen_manifest,
)
from research.comparison.schedule import (
    derive_native_seed,
    load_schedule,
    schedule_identity,
)
from research.comparison.specs import iter_cells

PREPARE_EVIDENCE_SCHEMA = "istratde-comparison-prepare-evidence.v1"
PREFLIGHT_EVIDENCE_SCHEMA = "istratde-comparison-preflight.v1"
REPORT_SCHEMA = "istratde-comparison-report.v1"

#: The one comparison ledger filename (owned by review.py).
LEDGER_FILENAME = review.LEDGER_FILENAME

#: Historic US-009 ledgers (immutable evidence; audit-compared, never written).
HISTORIC_LEDGERS: tuple[str, ...] = (
    ".runtime/acceptance-us009/budget-ledger.json",
    ".runtime/acceptance-us009-round2/budget-ledger.json",
)
HISTORIC_BASELINE_AUDIT = ".runtime/comparison-historic-baseline-audit.json"

#: The ORIGINAL immutable SHA-256 identities of the two historic ledgers,
#: independently pinned by the supervisor (2026-10-11). Byte equality
#: against THESE pins is the unchanged-original-ledger check; totals alone
#: are not sufficient.
HISTORIC_LEDGER_SHA256: dict[str, str] = {
    ".runtime/acceptance-us009/budget-ledger.json": (
        "59c236e8261dd3565dabca9f756dd806f44625cb38b8b89452e16adefe88ff65"
    ),
    ".runtime/acceptance-us009-round2/budget-ledger.json": (
        "d8340c359bddf65b39639fdeaf06ca36e4d2237aa97684bfead172a8b61915b2"
    ),
}

#: Compose file for the static mount-contract gate.
COMPOSE_FILE = Path("deploy") / "rsi-trustworthy" / "docker-compose.yaml"

_EXPECTED_DATA_LEAVES = {
    "/mnt/rsi_data/hello_synth/data/public",
    "/mnt/rsi_data/comparison/banknote/data/public",
    "/mnt/rsi_data/comparison/wdbc/data/public",
    "/mnt/rsi_data/comparison/spambase/data/public",
    "/mnt/rsi_data/comparison/banknote/data/final",
    "/mnt/rsi_data/comparison/wdbc/data/final",
    "/mnt/rsi_data/comparison/spambase/data/final",
}
_EXPECTED_WDBC_MASKS = {
    "/usr/local/lib/python3.10/dist-packages/sklearn/datasets/data/breast_cancer.csv",
    "/opt/python3.11/lib/python3.11/site-packages/sklearn/datasets/data/breast_cancer.csv",
}
_DEFAULT_REGISTRY_PATH = "/opt/rsi_repo/research/evaluator/registry.v1.json"
_FINAL_REGISTRY_CONTAINER_PATH = "/opt/rsi_final/registry.final.v1.json"
_FINAL_REGISTRY_MOUNT = "/opt/rsi_final"

#: Illustrative offline fitness vectors for the native replay-determinism
#: probe (never derived from any real evaluation).
_PROBE_FIRST_FITNESS = (-0.2, -0.4, -0.6, -0.8)
_PROBE_SECOND_FITNESS = (-0.5, -0.3, -0.9, -0.7)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_durable(path: Path, text: str) -> None:
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


# ----------------------------------------------------------------- prepare --
def _verify_placeholder_tree(runtime_dir: Path) -> None:
    """The final-publication tree may hold ONLY placeholder READMEs."""
    for task in sorted(TASKS_BY_ID):
        final_dir = runtime_dir / "final-publication" / task / "data" / "final"
        if not final_dir.is_dir():
            raise ComparisonManifestError(f"final_placeholder_missing:{task}")
        for path in sorted(final_dir.iterdir()):
            if path.is_symlink() or path.name != "README.md" or not path.is_file():
                raise ComparisonManifestError(
                    f"comparison_runtime_not_fresh:{path.relative_to(runtime_dir)}"
                )


#: Recognized PREPARATION entries (archiveable before a live ledger exists).
_PREPARATION_ENTRIES = {
    "schedule.json",
    "cells",
    "specs",
    "intents",
    "runs",
    review.MANIFEST_FILENAME,
    "preflight-evidence.v1.json",
    "prepare-evidence.json",
}


def _archive_prior_preparation(runtime_dir: Path) -> str | None:
    """Archive a previous PRE-LIVE preparation; refuse anything live.

    A fresh exact-HEAD re-preparation is legitimate only before any live
    artifact exists: a ledger, a supervisor review, a controller lock or a
    non-empty runs/ tree makes the runtime immutable and refuses. Earlier
    preparation evidence (manifest/schedule/cells/preflight) is MOVED to a
    sibling archive directory — never deleted. The final-publication
    placeholder tree is infrastructure (possibly live bind mounts) and is
    left in place untouched.
    """
    if runtime_dir.is_symlink():
        raise ComparisonManifestError("comparison_runtime_symlinked")
    for live in (
        LEDGER_FILENAME,
        review.REVIEW_FILENAME,
        "controller.lock",
    ):
        probe = runtime_dir / live
        if probe.exists() or probe.is_symlink():
            raise ComparisonManifestError(f"comparison_runtime_live:{live}")
    entries = [e for e in runtime_dir.iterdir() if e.name != "final-publication"]
    placeholder = runtime_dir / "final-publication"
    if placeholder.exists():
        _verify_placeholder_tree(runtime_dir)
    if not entries:
        return None
    runs = runtime_dir / "runs"
    if runs.exists() and any(runs.iterdir()):
        # A launched config is interrupted-experiment evidence, never
        # silently archived or rerun.
        raise ComparisonManifestError("comparison_runtime_not_fresh:runs")
    for entry in entries:
        if entry.is_symlink():
            raise ComparisonManifestError(
                f"comparison_runtime_symlinked:{entry.name}"
            )
        if entry.name not in _PREPARATION_ENTRIES:
            raise ComparisonManifestError(
                f"comparison_runtime_not_fresh:{entry.name}"
            )
    _verify_placeholder_tree(runtime_dir)
    archive = runtime_dir.parent / f"{runtime_dir.name}.archive-{uuid4().hex[:8]}"
    archive.mkdir()
    for entry in entries:
        os.rename(entry, archive / entry.name)
    return archive.name


def create_final_placeholders(runtime_dir: Path) -> list[str]:
    """Pre-create the final bind-source dirs (placeholder READMEs only)."""
    created: list[str] = []
    for task in sorted(TASKS_BY_ID):
        final_dir = runtime_dir / "final-publication" / task / "data" / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        readme = final_dir / "README.md"
        if readme.is_symlink():
            raise ComparisonManifestError("comparison_runtime_symlinked")
        if not readme.exists():
            _write_durable(
                readme,
                (
                    "Placeholder. Final-phase bytes (train.csv = train+validation"
                    " refit, test.csv = final features, sample_submission.csv)\n"
                    "are published ONLY after the verified all-27-cell barrier;"
                    " final labels NEVER appear here.\n"
                ),
            )
            created.append(str(readme))
    return created


def prepare_experiment(
    repo_root: Path,
    *,
    provider: NativeProvider | None = None,
    runtime_dir: Path | None = None,
    now_fn: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Offline prepare: freeze manifest + schedule + initial theta vectors.

    No ledger, no credentials, no model calls, no sandbox submissions.
    Refuses unless the tree is clean. Before any live artifact exists, a
    previous preparation is ARCHIVED (moved to a sibling directory, never
    deleted) so a new exact HEAD can freeze fresh state; a ledger, review
    file, controller lock or launched run makes the runtime immutable.
    """
    repo_root = Path(repo_root)
    runtime_dir = Path(runtime_dir) if runtime_dir is not None else (
        repo_root / review.RUNTIME_DIR
    )
    manifest = build_experiment_manifest(repo_root)  # binds exact clean HEAD
    archived = None
    if runtime_dir.exists():
        archived = _archive_prior_preparation(runtime_dir)
    commit = manifest["source_commit"]
    bindings = load_task_bindings()
    cells = iter_cells(bindings, commit)
    native = provider if provider is not None else SubprocessNativeProvider(repo_root)
    placeholder = runtime_dir / "final-publication"
    if placeholder.exists():
        # The placeholder leaf dirs may be LIVE Docker bind sources: their
        # directory identities (inodes) must survive preparation, so the
        # orchestrator initializes into a fresh staging sibling and its
        # state is MOVED in — the placeholder tree is never removed,
        # renamed or recreated (verified by the mount-lifetime regression).
        staging = runtime_dir.parent / f"{runtime_dir.name}.init-{uuid4().hex}"
        if staging.exists():
            raise ComparisonManifestError("comparison_runtime_not_fresh:staging")
        try:
            Orchestrator.initialize(
                runtime_dir=staging,
                repo_root=repo_root,
                cells=cells,
                provider=native,
                now_fn=now_fn,
            )
            for entry in staging.iterdir():
                target = runtime_dir / entry.name
                if target.exists():
                    raise ComparisonManifestError(
                        f"comparison_runtime_not_fresh:{entry.name}"
                    )
                os.rename(entry, target)
        except Exception:
            # Fail closed: partial state makes any later prepare refuse.
            raise
        finally:
            if staging.exists():
                try:
                    staging.rmdir()
                except OSError:
                    pass
        # Re-load from the REAL location: full revalidation of the moved
        # schedule/vector state against the deterministic plan.
        orch = Orchestrator.load(
            runtime_dir=runtime_dir,
            repo_root=repo_root,
            bindings=bindings,
            provider=native,
            now_fn=now_fn,
        )
    else:
        orch = Orchestrator.initialize(
            runtime_dir=runtime_dir,
            repo_root=repo_root,
            cells=cells,
            provider=native,
            now_fn=now_fn,
        )
    frozen = load_schedule(runtime_dir / "schedule.json", orch.cells)
    if schedule_identity(frozen, commit) != manifest["schedule_identity"]:
        raise ComparisonManifestError("schedule_identity_mismatch")
    manifest_path = review.freeze_experiment_manifest(
        repo_root, manifest, runtime_dir=runtime_dir
    )
    create_final_placeholders(runtime_dir)
    return {
        "schema": PREPARE_EVIDENCE_SCHEMA,
        "prepared_at": float(now_fn()),
        "source_commit": commit,
        "experiment_manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "schedule_identity": manifest["schedule_identity"],
        "num_cells": manifest["num_cells"],
        "num_inner_runs": manifest["num_inner_runs"],
        "cells_initialized": len(orch.cells),
        "archived_prior_preparation": archived,
        "ledger_created": False,
        "review_accepted": False,
    }


# --------------------------------------------------------------- preflight --
def _gate(
    gates: list[dict[str, Any]], name: str, fn: Callable[[], Any]
) -> bool:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - gate failure IS the evidence
        gates.append(
            {
                "name": name,
                "status": "fail",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        return False
    gates.append({"name": name, "status": "pass", "detail": detail})
    return True


def _compose_config(repo_root: Path) -> Mapping[str, Any]:
    if shutil.which("docker") is None:
        raise ComparisonManifestError("docker_cli_unavailable")
    proc = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(repo_root / COMPOSE_FILE),
            "config",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise ComparisonManifestError("compose_config_invalid")
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict) or not isinstance(payload.get("services"), dict):
        raise ComparisonManifestError("compose_config_invalid")
    return payload


def _check_mount_contract(repo_root: Path) -> Any:
    config = _compose_config(repo_root)
    services = config["services"]
    worker = services["worker"]
    volumes = worker.get("volumes") or []
    data_mounts = {
        v["target"]: v
        for v in volumes
        if str(v.get("target", "")).startswith("/mnt/rsi_data")
    }
    if set(data_mounts) != _EXPECTED_DATA_LEAVES:
        raise ComparisonManifestError(
            f"mount_data_leaves_mismatch:{sorted(data_mounts)}"
        )
    for target, mount in data_mounts.items():
        if mount.get("read_only") is not True:
            raise ComparisonManifestError(f"mount_not_read_only:{target}")
        source = Path(mount["source"])
        if source.name not in ("public", "final") or source.parent.name != "data":
            raise ComparisonManifestError(f"mount_source_not_leaf:{target}")
        if "sealed" in source.parts or "evaluator" in source.parts:
            raise ComparisonManifestError(f"mount_exposes_private:{target}")
        if source.name == "final" and (
            Path(repo_root / review.RUNTIME_DIR / "final-publication")
            not in source.parents
        ):
            raise ComparisonManifestError(f"final_mount_not_runtime:{target}")
    masks = {
        v["target"]: v
        for v in volumes
        if str(v.get("target", "")).endswith("sklearn/datasets/data/breast_cancer.csv")
    }
    if set(masks) != _EXPECTED_WDBC_MASKS or any(
        m.get("read_only") is not True or m.get("source") != "/dev/null"
        for m in masks.values()
    ):
        raise ComparisonManifestError("wdbc_mask_missing")
    writable = [v["target"] for v in volumes if not v.get("read_only")]
    if writable != ["/mnt/local_sandbox_workdir"]:
        raise ComparisonManifestError(f"unexpected_writable_mounts:{writable}")
    for mount in volumes:
        parts = Path(mount["source"]).resolve().parts
        if "evaluator" in parts or "sealed" in parts:
            raise ComparisonManifestError("mount_exposes_private_tree")
    dispatcher = services["dispatcher"]
    env = dispatcher.get("environment") or {}
    if env.get("EVALUATOR_REGISTRY_PATH") != _DEFAULT_REGISTRY_PATH:
        raise ComparisonManifestError("dispatcher_registry_default_mismatch")
    dispatcher_mounts = {
        v["target"]: v for v in (dispatcher.get("volumes") or [])
    }
    final_mount = dispatcher_mounts.get(_FINAL_REGISTRY_MOUNT)
    if (
        final_mount is None
        or final_mount.get("read_only") is not True
        or Path(final_mount["source"])
        != repo_root / review.RUNTIME_DIR / "final-publication"
    ):
        raise ComparisonManifestError("dispatcher_final_mount_missing")
    return {
        "data_leaves": sorted(data_mounts),
        "wdbc_masks": sorted(masks),
        "dispatcher_registry": env.get("EVALUATOR_REGISTRY_PATH"),
    }


def _check_historic_ledgers(repo_root: Path) -> Any:
    audit_path = repo_root / HISTORIC_BASELINE_AUDIT
    if not audit_path.is_file():
        raise ComparisonManifestError("historic_baseline_audit_missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {}
    for entry in audit.get("rounds", []):
        ledger = entry["ledger"]
        expected[ledger["path"]] = (
            ledger["committed_requests"],
            ledger["committed_tokens"],
            bool(ledger["stopped"]),
        )
    detail: dict[str, Any] = {}
    total_requests = 0
    total_tokens = 0
    for rel in HISTORIC_LEDGERS:
        path = repo_root / rel
        if not path.is_file():
            raise ComparisonManifestError(f"historic_ledger_missing:{rel}")
        raw = path.read_bytes()
        actual_sha = _sha256_bytes(raw)
        pinned = HISTORIC_LEDGER_SHA256.get(rel)
        if pinned is not None and actual_sha != pinned:
            # Byte identity against the ORIGINAL supervisor-pinned hashes:
            # accounting totals alone do not prove the ledger is unchanged.
            raise ComparisonManifestError(f"historic_ledger_hash_changed:{rel}")
        payload = json.loads(raw.decode("utf-8"))
        committed = payload.get("committed") or {}
        actual = (
            committed.get("requests"),
            committed.get("tokens"),
            bool(payload.get("stopped")),
        )
        if rel in expected and expected[rel] != actual:
            raise ComparisonManifestError(f"historic_ledger_changed:{rel}")
        total_requests += int(committed.get("requests") or 0)
        total_tokens += int(committed.get("tokens") or 0)
        detail[rel] = {
            "sha256": actual_sha,
            "pinned_sha256": pinned,
            "committed_requests": actual[0],
            "committed_tokens": actual[1],
            "stopped": actual[2],
        }
    aggregate = audit.get("aggregate_committed") or {}
    if (
        aggregate.get("requests") != total_requests
        or aggregate.get("tokens") != total_tokens
    ):
        raise ComparisonManifestError("historic_ledger_aggregate_mismatch")
    return detail


def _check_historic_artifact_audit(repo_root: Path) -> Any:
    """Re-run the read-only US-009 runtime audit (all 164 artifact hashes)."""
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "research" / "tools" / "audit_acceptance_runtime.py"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0 or "AUDIT OK" not in proc.stdout:
        raise ComparisonManifestError(
            f"historic_artifact_audit_failed:{proc.stdout.strip()[-200:]}"
        )
    return proc.stdout.strip().splitlines()[-1]


#: Expected running-stack container images (name:tag pins; owned by
#: review.STACK_IMAGE_PINS, keyed here by container name).
EXPECTED_STACK_IMAGES: dict[str, str] = {
    f"rsi_{service}": tag for service, tag in review.STACK_IMAGE_PINS.items()
}


def _expected_mount_contract(repo_root: Path) -> dict[str, dict[str, tuple[str, bool]]]:
    """The exact (target -> (host source, writable)) contract for the two
    services whose mounts carry security boundaries (worker + dispatcher),
    derived from the tracked compose layout — NOT from a fresh capture of
    the running stack."""
    repo = str(repo_root)
    worker: dict[str, tuple[str, bool]] = {
        "/opt/rsi_worker/minimal_worker.py": (
            f"{repo}/deploy/rsi-trustworthy/worker/minimal_worker.py",
            False,
        ),
        "/opt/rsi_worker/exec_trampoline.py": (
            f"{repo}/deploy/rsi-trustworthy/worker/exec_trampoline.py",
            False,
        ),
        "/run/rsi_control/key": (
            f"{repo}/.runtime/rsi-trustworthy/worker_control_key",
            False,
        ),
        "/mnt/rsi_storage/mlsandbox/jobs": (
            f"{repo}/.runtime/rsi-trustworthy/storage/mlsandbox/jobs",
            False,
        ),
        "/mnt/rsi_storage/mlsandbox/uploads": (
            f"{repo}/.runtime/rsi-trustworthy/storage/mlsandbox/uploads",
            False,
        ),
        "/mnt/rsi_data/hello_synth/data/public": (
            f"{repo}/tasks/hello_synth/data/public",
            False,
        ),
        "/mnt/local_sandbox_workdir": (
            f"{repo}/.runtime/rsi-trustworthy/workdir",
            True,
        ),
    }
    for task in sorted(TASKS_BY_ID):
        worker[f"/mnt/rsi_data/comparison/{task}/data/public"] = (
            f"{repo}/tasks/comparison/{task}/data/public",
            False,
        )
        worker[f"/mnt/rsi_data/comparison/{task}/data/final"] = (
            f"{repo}/.runtime/comparison-istratde/final-publication/{task}/data/final",
            False,
        )
    for mask in _EXPECTED_WDBC_MASKS:
        worker[mask] = ("/dev/null", False)
    dispatcher: dict[str, tuple[str, bool]] = {
        "/app": (f"{repo}/sandbox-controller/task_dispatcher", False),
        "/app/config/sandbox_config.json": (
            f"{repo}/deploy/rsi-trustworthy/config/sandbox_config.json",
            False,
        ),
        "/opt/rsi_repo/research": (f"{repo}/research", False),
        "/run/rsi_control/key": (
            f"{repo}/.runtime/rsi-trustworthy/worker_control_key",
            False,
        ),
        "/mnt/rsi_storage": (f"{repo}/.runtime/rsi-trustworthy/storage", True),
        "/mnt/local_sandbox_workdir": (
            f"{repo}/.runtime/rsi-trustworthy/workdir",
            True,
        ),
        "/opt/rsi_final": (
            f"{repo}/.runtime/comparison-istratde/final-publication",
            False,
        ),
    }
    return {"rsi_worker": worker, "rsi_dispatcher": dispatcher}


def inspect_runtime_stack(
    repo_root: Path, *, expected_images: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """ACTUAL running-stack identity vs the reviewed baseline.

    Verifies, failing closed: every expected container running with the
    pinned image TAG; when ``expected_images`` (the frozen manifest's
    ``stack_images.image_ids``) is given, the ACTUAL image ID must equal the
    frozen digest (a same-tag changed image refuses — no silently refreshed
    baseline); the worker's and dispatcher's ACTUAL mounts must match the
    exact compose-contract triples (target, host source, writability) — a
    same-target wrong-source mount refuses; and the dispatcher's active
    registry is explicitly the search or the final phase registry.
    """
    if shutil.which("docker") is None:
        raise ComparisonManifestError("docker_cli_unavailable")
    mount_contract = _expected_mount_contract(repo_root)
    detail: dict[str, Any] = {"containers": {}}
    for name, image in EXPECTED_STACK_IMAGES.items():
        proc = subprocess.run(
            ["docker", "inspect", name], capture_output=True, text=True, timeout=60
        )
        if proc.returncode != 0:
            raise ComparisonManifestError(f"stack_container_missing:{name}")
        info = json.loads(proc.stdout)[0]
        config = info.get("Config") or {}
        state = info.get("State") or {}
        if config.get("Image") != image:
            raise ComparisonManifestError(f"stack_image_mismatch:{name}")
        actual_image_id = info.get("Image")
        if expected_images is not None:
            service = name.removeprefix("rsi_")
            frozen = expected_images.get(service)
            if not isinstance(frozen, str) or actual_image_id != frozen:
                raise ComparisonManifestError(f"stack_image_id_mismatch:{name}")
        if not state.get("Running"):
            raise ComparisonManifestError(f"stack_container_down:{name}")
        detail["containers"][name] = {
            "id": info.get("Id"),
            "image_id": actual_image_id,
            "started_at": state.get("StartedAt"),
        }
        expected_mounts = mount_contract.get(name)
        if expected_mounts is not None:
            actual_mounts = {
                m.get("Destination"): (m.get("Source"), bool(m.get("RW")))
                for m in (info.get("Mounts") or [])
            }
            if actual_mounts != expected_mounts:
                raise ComparisonManifestError(
                    f"mount_contract_mismatch:{name}:"
                    f"{sorted(set(actual_mounts.items()) ^ set(expected_mounts.items()))}"
                )
        if name == "rsi_dispatcher":
            env = {
                entry.split("=", 1)[0]: entry.split("=", 1)[1]
                for entry in (config.get("Env") or [])
                if "=" in entry
            }
            registry = env.get("EVALUATOR_REGISTRY_PATH")
            if registry == _DEFAULT_REGISTRY_PATH:
                detail["phase"] = "search"
            elif registry == _FINAL_REGISTRY_CONTAINER_PATH:
                detail["phase"] = "final"
            else:
                raise ComparisonManifestError(
                    f"dispatcher_registry_unknown:{registry}"
                )
    return detail


def _check_native_runtime_identity(repo_root: Path, manifest: Mapping[str, Any]) -> Any:
    """ACTUAL pinned native identity, compared to the FROZEN manifest."""
    from research.comparison.native import load_native_python

    python = load_native_python(repo_root)
    version, backend = review._native_runtime_identity(python)
    expected = manifest.get("native") or {}
    if (
        version != "0.4.33"
        or backend != "cpu"
        or version != expected.get("jax_version_actual")
        or backend != expected.get("backend_actual")
    ):
        raise ComparisonManifestError(
            f"native_identity_mismatch:{version}/{backend}"
        )
    vendor = repo_root / "research" / "vendor" / "istratde" / "src"
    if not vendor.is_dir():
        raise ComparisonManifestError("native_vendor_missing")
    return {"python": str(python), "jax_version": version, "backend": backend}


def _check_no_live_artifacts(runtime_dir: Path) -> Any:
    ledger = runtime_dir / LEDGER_FILENAME
    if ledger.exists() or ledger.is_symlink():
        raise ComparisonManifestError("live_ledger_present")
    lock = runtime_dir / "controller.lock"
    if lock.exists() or lock.is_symlink():
        raise ComparisonManifestError("controller_lock_present")
    runs_dir = runtime_dir / "runs"
    if runs_dir.exists() and any(runs_dir.iterdir()):
        raise ComparisonManifestError("run_artifacts_present")
    review_path = runtime_dir / review.REVIEW_FILENAME
    if review_path.exists() or review_path.is_symlink():
        raise ComparisonManifestError("supervisor_review_present")
    return "no ledger, no review, no controller lock, no run artifacts"


def _check_final_placeholders(runtime_dir: Path) -> Any:
    for task in sorted(TASKS_BY_ID):
        final_dir = runtime_dir / "final-publication" / task / "data" / "final"
        if not final_dir.is_dir():
            raise ComparisonManifestError(f"final_placeholder_missing:{task}")
        names = {p.name for p in final_dir.iterdir()}
        if names != {"README.md"}:
            raise ComparisonManifestError(f"final_bytes_premature:{task}:{sorted(names)}")
    return "placeholder READMEs only; no final bytes published"


def run_preflight(
    repo_root: Path,
    *,
    provider: NativeProvider | None = None,
    runtime_dir: Path | None = None,
    check_stack: bool = True,
    now_fn: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Run every offline preflight gate and persist the evidence.

    Refuses (ComparisonManifestError) before writing anything when the
    experiment was never prepared. The supervisor review gate is REPORTED
    (and stays closed while absent); it is never opened by preflight.
    """
    repo_root = Path(repo_root)
    runtime_dir = Path(runtime_dir) if runtime_dir is not None else (
        repo_root / review.RUNTIME_DIR
    )
    manifest = load_frozen_manifest(repo_root, runtime_dir=runtime_dir)  # refuses when unprepared
    gates: list[dict[str, Any]] = []

    def _manifest_rederives() -> Any:
        if build_experiment_manifest(repo_root) != manifest:
            raise ComparisonManifestError("comparison_manifest_drift")
        return "re-derivation identical"

    def _clean_head() -> Any:
        head = review.require_clean_head(repo_root)
        if head != manifest.get("source_commit"):
            raise ComparisonManifestError("manifest_commit_mismatch")
        return head

    _gate(gates, "clean_head_matches_manifest", _clean_head)
    _gate(gates, "manifest_rederivation_identical", _manifest_rederives)

    def _data_valid() -> Any:
        bindings = load_task_bindings()
        return {
            task: {"rows": b.rows, "features": b.features, "identity": b.identity()}
            for task, b in sorted(bindings.items())
        }

    _gate(gates, "data_manifest_valid", _data_valid)

    def _registry_pins() -> Any:
        from research.evaluator.service import EvaluatorRegistry

        registry = EvaluatorRegistry.load(
            repo_root / "research" / "evaluator" / "registry.v1.json"
        )
        expected = manifest.get("registry", {}).get("tasks", {})
        for task_id, pins in expected.items():
            spec = registry.task(task_id)
            if (
                spec.split != pins["split"]
                or spec.metric != pins["metric"]
                or spec.direction.value != pins["direction"]
                or spec.answer_sha256 != pins["answer_sha256"]
                or spec.worker_data_public_path != pins["worker_data_public_path"]
            ):
                raise ComparisonManifestError(f"registry_pin_drift:{task_id}")
        return {"registry_version": registry.registry_version}

    _gate(gates, "registry_pins", _registry_pins)

    def _native_env() -> Any:
        from research.comparison.native import load_native_python

        return str(load_native_python(repo_root))

    _gate(gates, "native_env_resolves", _native_env)
    _gate(
        gates,
        "native_runtime_identity",
        lambda: _check_native_runtime_identity(repo_root, manifest),
    )

    state: dict[str, Any] = {}

    def _load_orch() -> Orchestrator:
        if "orch" not in state:
            bindings = load_task_bindings()
            native = (
                provider
                if provider is not None
                else SubprocessNativeProvider(repo_root)
            )
            orch = Orchestrator.load(
                runtime_dir=runtime_dir,
                repo_root=repo_root,
                bindings=bindings,
                provider=native,
                now_fn=now_fn,
            )
            state["orch"] = orch
        return state["orch"]

    def _orchestration_valid() -> Any:
        orch = _load_orch()
        summary = orch.terminal_summary()
        if summary:
            raise ComparisonManifestError("unexpected_terminal_cells")
        return {"cells": len(orch.cells), "positions": len(orch.positions)}

    _gate(gates, "orchestration_state_valid", _orchestration_valid)

    def _native_replay() -> Any:
        orch = _load_orch()
        provider_obj = orch.provider
        checked = 0
        for task_id in sorted(TASKS_BY_ID):
            for seed in SEARCH_SEEDS:
                r_cell = next(
                    c
                    for c in orch.cells
                    if c.task_id == task_id
                    and c.method == "RANDOM"
                    and c.search_seed == seed
                )
                s_cell = next(
                    c
                    for c in orch.cells
                    if c.task_id == task_id
                    and c.method == "ISTRATDE"
                    and c.search_seed == seed
                )
                native_seed = derive_native_seed(task_id, seed)
                first = provider_obj.complete_run(
                    native_seed, _PROBE_FIRST_FITNESS, _PROBE_SECOND_FITNESS
                )
                second = provider_obj.complete_run(
                    native_seed, _PROBE_FIRST_FITNESS, _PROBE_SECOND_FITNESS
                )
                if first != second:
                    raise ComparisonManifestError(
                        f"native_replay_nondeterministic:{task_id}:{seed}"
                    )
                frozen_r = _frozen_vectors(runtime_dir, r_cell.cell_key())
                frozen_s = _frozen_vectors(runtime_dir, s_cell.cell_key())
                expected_first = [tuple(map(float, first.first_batch[i])) for i in range(4)]
                for index in range(4):
                    if tuple(frozen_r[str(index)]) != expected_first[index]:
                        raise ComparisonManifestError(
                            f"paired_random_first_ask_mismatch:{task_id}:{seed}:{index}"
                        )
                    if tuple(frozen_s[str(index)]) != expected_first[index]:
                        raise ComparisonManifestError(
                            f"istratde_first_ask_mismatch:{task_id}:{seed}:{index}"
                        )
                checked += 1
        return {"pairs_replayed": checked, "tells_per_run": 2}

    _gate(gates, "native_replay_determinism", _native_replay)
    _gate(gates, "historic_ledgers_unchanged", lambda: _check_historic_ledgers(repo_root))
    _gate(
        gates,
        "historic_artifact_audit",
        lambda: _check_historic_artifact_audit(repo_root),
    )
    if check_stack:
        _gate(gates, "compose_mount_contract", lambda: _check_mount_contract(repo_root))
        stack_baseline = (manifest.get("stack_images") or {}).get("image_ids")
        _gate(
            gates,
            "runtime_stack_identity",
            lambda: inspect_runtime_stack(repo_root, expected_images=stack_baseline),
        )
    _gate(gates, "final_publication_placeholders", lambda: _check_final_placeholders(runtime_dir))
    _gate(gates, "no_live_artifacts", lambda: _check_no_live_artifacts(runtime_dir))

    # The supervisor review gate is REPORTED, never opened by preflight.
    review_path = review.review_path_for(repo_root)
    if review_path.exists() and not review_path.is_symlink():
        try:
            accepted = review.check_review_acceptance(repo_root)
            live_gate = {"status": "accepted", "reviewer": accepted.get("reviewer")}
        except review.ComparisonReviewRefused as exc:
            live_gate = {"status": "refused", "reason": str(exc)}
    else:
        live_gate = {"status": "refused", "reason": "comparison_review_absent"}

    requirements = None
    if live_gate["status"] == "refused":
        try:
            requirements = review.describe_review_requirements(
                repo_root, manifest_path=runtime_dir / review.MANIFEST_FILENAME
            )
        except ComparisonManifestError:
            requirements = None

    verdict = (
        "PASS" if all(gate["status"] == "pass" for gate in gates) else "FAIL"
    )
    evidence = {
        "schema": PREFLIGHT_EVIDENCE_SCHEMA,
        "generated_at": float(now_fn()),
        "source_commit": manifest.get("source_commit"),
        "experiment_manifest_sha256": _sha256_bytes(
            (runtime_dir / review.MANIFEST_FILENAME).read_bytes()
        ),
        "gates": gates,
        "live_execution_gate": live_gate,
        "review_requirements": requirements,
        "verdict": verdict,
    }
    _write_durable(
        runtime_dir / "preflight-evidence.v1.json",
        canonical_json(evidence) + "\n",
    )
    return evidence


def _frozen_vectors(runtime_dir: Path, cell_key: str) -> Mapping[str, Any]:
    path = runtime_dir / "cells" / cell_key.replace(":", "-") / "cell.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    vectors = payload.get("vectors")
    if not isinstance(vectors, dict):
        raise ComparisonManifestError(f"cell_vectors_missing:{cell_key}")
    return vectors


# ------------------------------------------------------------------ report --
def build_report(
    repo_root: Path,
    *,
    provider: NativeProvider | None = None,
    runtime_dir: Path | None = None,
    now_fn: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Read-only experiment report: per-cell matrix + ledger totals.

    Revalidates ALL persisted state through Orchestrator.load and the
    ledger's own reopen validation. EXP-008 extends this with the
    predeclared paired bootstrap analysis and plots.
    """
    repo_root = Path(repo_root)
    runtime_dir = Path(runtime_dir) if runtime_dir is not None else (
        repo_root / review.RUNTIME_DIR
    )
    manifest = load_frozen_manifest(repo_root, runtime_dir=runtime_dir)
    bindings = load_task_bindings()
    native = provider if provider is not None else SubprocessNativeProvider(repo_root)
    orch = Orchestrator.load(
        runtime_dir=runtime_dir,
        repo_root=repo_root,
        bindings=bindings,
        provider=native,
        now_fn=now_fn,
    )
    terminal = orch.terminal_summary()
    ledger_summary: dict[str, Any] | None = None
    ledger_path = runtime_dir / LEDGER_FILENAME
    if ledger_path.exists() and not ledger_path.is_symlink():
        # READ-ONLY (EXP-008): full _load validation + roster binding, but
        # never the flock and never a persisted stop on detected problems.
        view = ReadOnlyComparisonLedger.open(
            ledger_path, cells=orch.cells, now_fn=now_fn
        )
        snapshot = view.snapshot()
        ledger_summary = {
            "experiment_identity": snapshot["experiment_identity"],
            "started_at": snapshot["started_at"],
            "deadline_epoch": snapshot["deadline_epoch"],
            "stopped": snapshot["stopped"],
            "global": snapshot["global"],
            "final_phase": ledger_final_phase(snapshot),
        }
    rows: list[dict[str, Any]] = []
    for cell in orch.cells:
        cell_key = cell.cell_key()
        state = orch._cell_state(cell_key)
        record = terminal.get(cell_key)
        winner = state.get("winner")
        final_result = _read_final_result(runtime_dir, cell_key)
        rows.append(
            {
                "cell_key": cell_key,
                "task_id": cell.task_id,
                "method": cell.method,
                "search_seed": cell.search_seed,
                "terminal_status": (
                    record.get("status") if isinstance(record, dict) else None
                ),
                "validation_score": (
                    winner.get("validation_score") if isinstance(winner, dict) else None
                ),
                "final": final_result,
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": float(now_fn()),
        "source_commit": manifest.get("source_commit"),
        "experiment_manifest_sha256": _sha256_bytes(
            (runtime_dir / review.MANIFEST_FILENAME).read_bytes()
        ),
        "schedule_identity": manifest.get("schedule_identity"),
        "ledger": ledger_summary,
        "cells": rows,
    }


def ledger_final_phase(snapshot: Mapping[str, Any]) -> Any:
    return snapshot.get("final_phase")


def _read_final_result(runtime_dir: Path, cell_key: str) -> dict[str, Any] | None:
    path = runtime_dir / "final" / cell_key.replace(":", "-") / "final.json"
    if path.is_symlink() or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "unreadable"}
    if not isinstance(payload, dict):
        return {"status": "unreadable"}
    return {
        "status": payload.get("status"),
        "raw_score": payload.get("raw_score"),
        "utility": payload.get("utility"),
    }
