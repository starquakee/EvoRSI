"""Trusted independent scoring service (US-005).

The evaluator runs on the controller side, separate from the candidate
worker. It accepts only bounded CSV prediction *bytes* plus an allowlisted
task ID; it never executes submitted Python and never imports
candidate-supplied metric code (trusted metrics live in
``research.evaluator.metrics``). Answers and the registry live in an
evaluator-private tree that must never be mounted into a worker;
``validate_worker_mount_plan`` denies any mount exposing that tree,
including via a parent directory or an alternate sub-path.

Fail closed everywhere: unknown task, oversized/malformed predictions,
symlinks, traversal, answer-integrity mismatch and corrupt registries all
raise ``EvaluatorError`` with a machine-readable ``reason`` and never
produce a score.
"""
from __future__ import annotations

import csv
import errno
import hashlib
import io
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.contracts.results import MetricDirection, metric_direction
from research.evaluator.metrics import METRIC_FUNCTIONS

REGISTRY_SCHEMA_VERSION = 1
REGISTRY_VERSION = "evaluator-registry.v1"
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().with_name("registry.v1.json")

SPLITS = frozenset({"validation", "test"})
HARD_MAX_PREDICTION_BYTES = 64 * 1024 * 1024
HARD_MAX_ROWS = 1_000_000

REASON_TASK_NOT_ALLOWLISTED = "task_not_allowlisted"
REASON_PREDICTION_OVERSIZED = "prediction_oversized"
REASON_PREDICTION_MALFORMED = "prediction_malformed"
REASON_SUBMISSION_SCHEMA_MISMATCH = "submission_schema_mismatch"
REASON_SUBMISSION_ROW_MISMATCH = "submission_row_mismatch"
REASON_SUBMISSION_ID_MISMATCH = "submission_id_mismatch"
REASON_SUBMISSION_LABEL_INVALID = "submission_label_invalid"
REASON_ANSWER_INTEGRITY_FAILED = "answer_integrity_failed"
REASON_ANSWER_MALFORMED = "answer_malformed"
REASON_PATH_TRAVERSAL = "path_traversal"
REASON_SYMLINK_REJECTED = "symlink_rejected"
REASON_NOT_REGULAR_FILE = "not_regular_file"
REASON_REGISTRY_INVALID = "registry_invalid"


class EvaluatorError(Exception):
    """Machine-readable scoring rejection/failure (fail closed)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class TaskSpec:
    """One allowlisted scoring task (trusted, pinned by the registry)."""

    task_id: str
    metric: str
    direction: MetricDirection
    split: str
    answer_path: Path
    answer_sha256: str
    id_column: str
    target_column: str
    label_values: frozenset[int]
    max_prediction_bytes: int
    max_rows: int

    @property
    def prediction_columns(self) -> list[str]:
        return [self.id_column, self.target_column]


@dataclass(frozen=True)
class ScoreOutcome:
    """Trusted scoring evidence; only ever produced on success."""

    task_id: str
    split: str
    metric: str
    direction: MetricDirection
    score: float
    row_count: int
    prediction_sha256: str
    answer_sha256: str
    registry_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "metric": self.metric,
            "direction": self.direction.value,
            "score": self.score,
            "row_count": self.row_count,
            "prediction_sha256": self.prediction_sha256,
            "answer_sha256": self.answer_sha256,
            "registry_version": self.registry_version,
        }


class EvaluatorRegistry:
    """Pinned, allowlisted task registry; loads fail closed."""

    def __init__(self, *, registry_version: str, tasks: Mapping[str, TaskSpec]) -> None:
        if not registry_version:
            raise EvaluatorError(REASON_REGISTRY_INVALID, "missing registry_version")
        if not tasks:
            raise EvaluatorError(REASON_REGISTRY_INVALID, "registry has no tasks")
        self.registry_version = registry_version
        self.tasks = dict(tasks)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "EvaluatorRegistry":
        registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
        try:
            raw = registry_path.read_bytes()
        except OSError as exc:
            raise EvaluatorError(REASON_REGISTRY_INVALID, f"unreadable: {exc}") from exc
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvaluatorError(REASON_REGISTRY_INVALID, f"corrupt json: {exc}") from exc
        if not isinstance(data, dict):
            raise EvaluatorError(REASON_REGISTRY_INVALID, "registry must be a JSON object")
        if data.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise EvaluatorError(REASON_REGISTRY_INVALID, "unsupported schema_version")
        registry_version = data.get("registry_version")
        if not isinstance(registry_version, str) or not registry_version:
            raise EvaluatorError(REASON_REGISTRY_INVALID, "missing registry_version")
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, dict) or not raw_tasks:
            raise EvaluatorError(REASON_REGISTRY_INVALID, "missing tasks")
        root = registry_path.resolve().parent
        tasks = {
            task_id: _load_task_spec(task_id, entry, root)
            for task_id, entry in raw_tasks.items()
        }
        return cls(registry_version=registry_version, tasks=tasks)

    def task(self, task_id: str) -> TaskSpec:
        try:
            return self.tasks[task_id]
        except KeyError:
            raise EvaluatorError(REASON_TASK_NOT_ALLOWLISTED, task_id) from None


def _require_str(entry: Mapping[str, Any], key: str, task_id: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise EvaluatorError(REASON_REGISTRY_INVALID, f"{task_id}: missing {key}")
    return value


def _require_bounded_int(
    entry: Mapping[str, Any], key: str, task_id: str, *, hard_max: int
) -> int:
    value = entry.get(key)
    if type(value) is not int or value <= 0 or value > hard_max:
        raise EvaluatorError(REASON_REGISTRY_INVALID, f"{task_id}: invalid {key}")
    return value


def _load_task_spec(task_id: str, entry: Any, root: Path) -> TaskSpec:
    if not isinstance(task_id, str) or not task_id:
        raise EvaluatorError(REASON_REGISTRY_INVALID, "task id must be a non-empty string")
    if not isinstance(entry, dict):
        raise EvaluatorError(REASON_REGISTRY_INVALID, f"{task_id}: entry must be an object")
    metric = _require_str(entry, "metric", task_id)
    if metric not in METRIC_FUNCTIONS:
        # Candidate- or task-supplied metric code is never imported; only
        # trusted in-repo implementations may be named.
        raise EvaluatorError(REASON_REGISTRY_INVALID, f"{task_id}: untrusted metric {metric}")
    try:
        direction = metric_direction(metric)
    except ValueError as exc:
        raise EvaluatorError(REASON_REGISTRY_INVALID, f"{task_id}: {exc}") from exc
    split = _require_str(entry, "split", task_id)
    if split not in SPLITS:
        raise EvaluatorError(REASON_REGISTRY_INVALID, f"{task_id}: invalid split {split}")
    answer_rel = _require_str(entry, "answer_file", task_id)
    answer_parts = Path(answer_rel).parts
    if Path(answer_rel).is_absolute() or ".." in answer_parts:
        raise EvaluatorError(REASON_REGISTRY_INVALID, f"{task_id}: answer_file escapes root")
    answer_path = (root / answer_rel).resolve()
    if not answer_path.is_relative_to(root):
        raise EvaluatorError(REASON_REGISTRY_INVALID, f"{task_id}: answer_file escapes root")
    answer_sha256 = _require_str(entry, "answer_sha256", task_id)
    label_values_raw = entry.get("label_values")
    if (
        not isinstance(label_values_raw, list)
        or not label_values_raw
        or any(type(v) is not int for v in label_values_raw)
    ):
        raise EvaluatorError(REASON_REGISTRY_INVALID, f"{task_id}: invalid label_values")
    spec = TaskSpec(
        task_id=task_id,
        metric=metric,
        direction=direction,
        split=split,
        answer_path=answer_path,
        answer_sha256=answer_sha256,
        id_column=_require_str(entry, "id_column", task_id),
        target_column=_require_str(entry, "target_column", task_id),
        label_values=frozenset(label_values_raw),
        max_prediction_bytes=_require_bounded_int(
            entry, "max_prediction_bytes", task_id, hard_max=HARD_MAX_PREDICTION_BYTES
        ),
        max_rows=_require_bounded_int(entry, "max_rows", task_id, hard_max=HARD_MAX_ROWS),
    )
    _load_answer_rows(spec)  # integrity + parse check at load, fail closed
    return spec


def _parse_csv_bytes(data: bytes, *, max_rows: int, error_reason: str) -> list[list[str]]:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EvaluatorError(error_reason, f"not utf-8: {exc}") from exc
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise EvaluatorError(error_reason, f"csv parse: {exc}") from exc
    rows = [row for row in rows if row]
    if not rows:
        raise EvaluatorError(error_reason, "empty csv")
    width = len(rows[0])
    if width == 0:
        raise EvaluatorError(error_reason, "empty header")
    for row in rows:
        if len(row) != width:
            raise EvaluatorError(error_reason, "ragged rows")
    if len(rows) - 1 > max_rows:
        raise EvaluatorError(error_reason, "row count above bound")
    return rows


def _parse_int_column(values: Sequence[str], *, error_reason: str) -> list[int]:
    parsed: list[int] = []
    for raw in values:
        try:
            parsed.append(int(raw.strip()))
        except ValueError:
            raise EvaluatorError(error_reason, f"non-integer value: {raw!r}") from None
    return parsed


def _load_answer_rows(spec: TaskSpec) -> list[tuple[int, int]]:
    """Read + integrity-check + parse the pinned answer file (fail closed)."""
    try:
        st = os.lstat(spec.answer_path)
    except OSError as exc:
        raise EvaluatorError(REASON_ANSWER_INTEGRITY_FAILED, f"unreadable: {exc}") from exc
    if not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode):
        raise EvaluatorError(REASON_ANSWER_INTEGRITY_FAILED, "answer not a regular file")
    try:
        data = spec.answer_path.read_bytes()
    except OSError as exc:
        raise EvaluatorError(REASON_ANSWER_INTEGRITY_FAILED, f"unreadable: {exc}") from exc
    if _sha256(data) != spec.answer_sha256:
        raise EvaluatorError(REASON_ANSWER_INTEGRITY_FAILED, spec.task_id)
    rows = _parse_csv_bytes(data, max_rows=spec.max_rows, error_reason=REASON_ANSWER_MALFORMED)
    header, body = rows[0], rows[1:]
    if header != spec.prediction_columns:
        raise EvaluatorError(REASON_ANSWER_MALFORMED, f"answer header {header}")
    ids = _parse_int_column([row[0] for row in body], error_reason=REASON_ANSWER_MALFORMED)
    labels = _parse_int_column([row[1] for row in body], error_reason=REASON_ANSWER_MALFORMED)
    if any(label not in spec.label_values for label in labels):
        raise EvaluatorError(REASON_ANSWER_MALFORMED, "answer label outside label_values")
    return list(zip(ids, labels))


def score_submission(
    registry: EvaluatorRegistry, task_id: str, prediction_bytes: bytes
) -> ScoreOutcome:
    """Score bounded prediction bytes against the pinned answer, fail closed.

    Only trusted in-repo metric functions run here; prediction bytes are
    parsed as data, never executed.
    """
    spec = registry.task(task_id)
    if len(prediction_bytes) > spec.max_prediction_bytes:
        raise EvaluatorError(
            REASON_PREDICTION_OVERSIZED,
            f"{len(prediction_bytes)} > {spec.max_prediction_bytes}",
        )
    answer_rows = _load_answer_rows(spec)  # re-verify integrity at score time
    rows = _parse_csv_bytes(
        prediction_bytes, max_rows=spec.max_rows, error_reason=REASON_PREDICTION_MALFORMED
    )
    header, body = rows[0], rows[1:]
    if header != spec.prediction_columns:
        raise EvaluatorError(
            REASON_SUBMISSION_SCHEMA_MISMATCH,
            f"expected columns {spec.prediction_columns}, got {header}",
        )
    if len(body) != len(answer_rows):
        raise EvaluatorError(
            REASON_SUBMISSION_ROW_MISMATCH,
            f"expected {len(answer_rows)} rows, got {len(body)}",
        )
    pred_ids = _parse_int_column([row[0] for row in body], error_reason=REASON_PREDICTION_MALFORMED)
    pred_labels = _parse_int_column(
        [row[1] for row in body], error_reason=REASON_PREDICTION_MALFORMED
    )
    answer_ids = [row[0] for row in answer_rows]
    answer_labels = [row[1] for row in answer_rows]
    if pred_ids != answer_ids:
        raise EvaluatorError(REASON_SUBMISSION_ID_MISMATCH, "id column mismatch or order")
    if any(label not in spec.label_values for label in pred_labels):
        raise EvaluatorError(
            REASON_SUBMISSION_LABEL_INVALID,
            f"labels must be in {sorted(spec.label_values)}",
        )
    metric_fn = METRIC_FUNCTIONS[spec.metric]  # trusted only; load-time checked
    score = metric_fn(answer_labels, pred_labels)
    return ScoreOutcome(
        task_id=spec.task_id,
        split=spec.split,
        metric=spec.metric,
        direction=spec.direction,
        score=score,
        row_count=len(body),
        prediction_sha256=_sha256(prediction_bytes),
        answer_sha256=spec.answer_sha256,
        registry_version=registry.registry_version,
    )


def read_prediction_file(path: Path | str, *, allowed_root: Path | str, max_bytes: int) -> bytes:
    """Read a bounded regular file using pinned, no-follow descriptors.

    Every directory component is opened relative to the previous descriptor.
    Replacing a checked path cannot redirect a later read outside the root.
    At most max_bytes+1 bytes are read, even if the file grows concurrently.
    """
    if type(max_bytes) is not int or max_bytes <= 0 or max_bytes > HARD_MAX_PREDICTION_BYTES:
        raise EvaluatorError(REASON_PREDICTION_OVERSIZED, "invalid bound")
    root = Path(os.path.abspath(os.fspath(allowed_root)))
    candidate = Path(path)
    if ".." in candidate.parts:
        raise EvaluatorError(REASON_PATH_TRAVERSAL, str(path))
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        raise EvaluatorError(REASON_PATH_TRAVERSAL, "prediction outside allowed root") from None
    if not relative.parts:
        raise EvaluatorError(REASON_NOT_REGULAR_FILE, "prediction is root directory")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    descriptors: list[int] = []
    try:
        current = os.open("/", directory_flags)
        descriptors.append(current)
        for part in (*root.parts[1:], *relative.parts[:-1]):
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
        file_fd = os.open(relative.name, file_flags, dir_fd=current)
        descriptors.append(file_fd)
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise EvaluatorError(REASON_NOT_REGULAR_FILE, "prediction is not a regular file")
        if metadata.st_size > max_bytes:
            raise EvaluatorError(REASON_PREDICTION_OVERSIZED, "file size exceeds bound")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(file_fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise EvaluatorError(REASON_PREDICTION_OVERSIZED, "file grew past bound")
        return data
    except OSError as exc:
        reason = (REASON_SYMLINK_REJECTED if exc.errno in (errno.ELOOP, errno.ENOTDIR)
                  else REASON_NOT_REGULAR_FILE)
        raise EvaluatorError(reason, type(exc).__name__) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


@dataclass(frozen=True)
class MountDenial:
    """Machine-readable worker-mount rejection."""

    reason: str
    source: str


REASON_MOUNT_EVALUATOR_ROOT = "evaluator_root_mounted"
REASON_MOUNT_EVALUATOR_ANCESTOR = "evaluator_ancestor_mounted"
REASON_MOUNT_EVALUATOR_SUBPATH = "evaluator_subpath_mounted"


def validate_worker_mount_plan(
    mounts: Sequence[str], *, evaluator_root: Path | str
) -> list[MountDenial]:
    """Deny any worker mount that would expose the evaluator-private tree.

    ``mounts`` are host-side mount sources planned for the candidate
    worker. A mount is denied when its resolved source IS the evaluator
    root (or a symlink alias of it), is an ANCESTOR of it (exposure via
    parent directory), or lives INSIDE it (alternate path to the same
    answers/registry). Read-only mounts are denied too: the answers must
    not be visible to candidate code at all.
    """
    root_real = Path(os.path.realpath(str(evaluator_root)))
    denials: list[MountDenial] = []
    for raw_source in mounts:
        source_real = Path(os.path.realpath(str(raw_source)))
        if source_real == root_real:
            denials.append(MountDenial(REASON_MOUNT_EVALUATOR_ROOT, str(raw_source)))
        elif root_real.is_relative_to(source_real):
            denials.append(MountDenial(REASON_MOUNT_EVALUATOR_ANCESTOR, str(raw_source)))
        elif source_real.is_relative_to(root_real):
            denials.append(MountDenial(REASON_MOUNT_EVALUATOR_SUBPATH, str(raw_source)))
    return denials
