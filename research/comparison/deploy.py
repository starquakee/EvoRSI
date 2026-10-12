"""Deployment pins for the comparison runner (EXP-005).

The comparison runs against the SAME isolated 6581 trustworthy stack as the
US-009 acceptance (same gateway, same private auth file, same dispatcher
storage mount, same pinned credential-free litellm profile YAML), but the
comparison package keeps its OWN constants so it never imports the US-009
acceptance config module (the EXP-001 package-isolation contract). These
values are deployment topology, not experiment parameters.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from research.adapters.endpoints import DEFAULT_ENDPOINT
from research.comparison.protocol import ComparisonSpecError

#: New-stack loopback gateway (research client default endpoint).
GATEWAY_URL = DEFAULT_ENDPOINT

#: Private sandbox auth file (0600, gitignored; written by init_auth.sh).
AUTH_FILE = Path(".runtime") / "rsi-trustworthy" / "auth.env"

#: Dispatcher storage: container prefix <-> host runtime root (compose mounts
#: .runtime/rsi-trustworthy/storage at /mnt/rsi_storage in api/dispatcher).
#: The ACTUAL per-job layout (api_server.py) is
#: STORAGE_PATH/jobs/<YYYY-MM-DD>/<dataset_label>/<job_id> with
#: STORAGE_PATH=/mnt/rsi_storage/mlsandbox; prediction snapshots therefore
#: live at <jobs prefix><date>/<label>/<job_id>/prediction_snapshot.csv.
CONTAINER_STORAGE_PREFIX = "/mnt/rsi_storage/"
CONTAINER_JOBS_PREFIX = "/mnt/rsi_storage/mlsandbox/jobs/"
DISPATCHER_STORAGE_ROOT = Path(".runtime") / "rsi-trustworthy" / "storage"

_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9._-]+")


def validate_job_snapshot(snapshot: object, job_id: object) -> str:
    """Validate the dispatcher-emitted container snapshot path for ITS OWN
    job; return the storage-root-relative path. Fail closed otherwise.

    The actual dispatcher emission (api_server.py) is
    ``/mnt/rsi_storage/mlsandbox/jobs/<YYYY-MM-DD>/<dataset_label>/<job_id>/
    prediction_snapshot.csv``; the trusted middle segments are
    dispatcher-computed at submit time. The binding rule stays exact:
    jobs-prefix, safe segments only (never ``.``/``..``), the parent
    directory name IS the job id, the filename is exactly
    ``prediction_snapshot.csv``.
    """
    rule = "prediction_snapshot_path_invalid"
    if not isinstance(snapshot, str) or not isinstance(job_id, str) or not job_id:
        raise ComparisonSpecError(rule)
    if not _SAFE_SEGMENT.fullmatch(job_id):
        raise ComparisonSpecError(rule)
    if not snapshot.startswith(CONTAINER_JOBS_PREFIX):
        raise ComparisonSpecError(rule)
    parts = snapshot[len(CONTAINER_JOBS_PREFIX):].split("/")
    if (
        len(parts) < 2
        or parts[-1] != "prediction_snapshot.csv"
        or parts[-2] != job_id
        or any(
            not _SAFE_SEGMENT.fullmatch(part) or part in (".", "..")
            for part in parts
        )
    ):
        raise ComparisonSpecError(rule)
    return snapshot[len(CONTAINER_STORAGE_PREFIX):]


def job_snapshot_host_path(
    snapshot: object, job_id: object, *, storage_root: Path
) -> Path:
    """Validated host path of a job's own prediction snapshot.

    Every untrusted component below the storage root is checked with
    ``is_symlink()`` BEFORE any resolve/read: resolving first would follow a
    file- or job-directory symlink into ANOTHER job inside the same storage
    root and make the later symlink test vacuous. The returned path is the
    nominal (never followed) location.
    """
    rule = "prediction_snapshot_path_invalid"
    relative = validate_job_snapshot(snapshot, job_id)
    root = storage_root.resolve()
    current = root
    for part in relative.split("/"):
        if part in ("", ".", ".."):
            raise ComparisonSpecError(rule)
        current = current / part
        if current.is_symlink():
            raise ComparisonSpecError(rule)
    host = storage_root / relative
    resolved = host.resolve()
    if resolved != current or not resolved.is_relative_to(root):
        raise ComparisonSpecError(rule)
    return host

#: Pinned credential-free managed-K3 litellm profile (stream/low/8192/
#: temperature 1/top_p .95); validated against ModelProfile at load.
LITELLM_YAML = (
    Path("OpenMLE-Evo")
    / "tts_search"
    / "configs"
    / "litellm"
    / "trustworthy_kimi.yaml"
)

#: Configs never carry a real token; the runner injects it in memory only.
API_KEY_PLACEHOLDER = "runtime"


def git_head_commit(repo_root: Path) -> str:
    """Current git HEAD of the implementation under review."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ComparisonSpecError("comparison_git_unavailable") from exc
