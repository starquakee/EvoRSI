"""Official UCI comparison datasets: fetch, convert, group, split, seal (EXP-002).

This module prepares the three approved tasks (banknote UCI 267, wdbc
UCI 17, spambase UCI 94) from the official UCI archives exactly once, with
auditable byte identities. It is controller-side tooling: it requires the
pinned scikit-learn (for the approved StratifiedGroupKFold split) and is
therefore deliberately NOT imported by ``research.comparison.__init__`` —
the protocol/spec modules stay stdlib-only.

Pipeline (fully deterministic, no timestamps in hashed artifacts):

1. RAW: the official archive bytes are stored sealed under
   ``research/comparison/sealed/<task>/raw/`` with URL/license/sha256
   provenance. Raw files are never worker-visible.
2. CONVERT: strict parsing removes original identifiers (the WDBC id
   column), maps labels to 0/1 (WDBC B=0, M=1; banknote/spambase labels
   are already 0/1), rejects ragged/non-numeric/non-finite rows and
   out-of-domain labels. All rows are preserved.
3. GROUP: rows with identical numeric feature vectors share one group
   (parsed float tuples, so equivalent zero representations like ``-0``
   and ``0`` group together). Contradictory-label duplicate groups are
   preserved and never deduplicated; a group can never straddle splits.
4. SPLIT: one fixed StratifiedGroupKFold(n_splits=5, shuffle=True,
   random_state=20261010+uci_id) per task; folds 0/1/2 = train, fold 3 =
   validation, fold 4 = final. The scikit-learn version is pinned and
   checked so the frozen fold assignment cannot drift silently.
5. IDS: each split is deterministically shuffled with a stdlib
   ``random.Random`` seeded from sha256(SHUFFLE_DOMAIN|task|split) — a
   function of task and split names ONLY, never of labels or row content —
   then rows get fresh sequential ids 0..n-1 in shuffled order. Ids leak
   neither original row order nor labels, and are not original indices.

Worker-visible search data (``tasks/comparison/<task>/data/public/``)
contains ONLY train.csv (id, features, label), test.csv (validation
features: id, features) and sample_submission.csv. Validation labels live
exclusively in the evaluator-private tree; final features/labels and the
train+validation refit file stay sealed under ``research/comparison`` —
physically outside every worker mount. Publishing any final-phase file
(train+validation refit or final features) additionally requires the
verified all-cell frozen barrier (EXP-006): ``publish_final_refit``
re-runs the FULL shared orchestrator/original-ledger verification itself
(requires the explicit orchestrator + open ledger context and the durable
final-phase transition bound to the barrier) and writes ONLY into the
ignored runtime publication root (``FINAL_PUBLICATION_ROOT``) together
with the phase-specific ``registry.final.v1.json`` — the tracked search
registry and search data stay byte-identical.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.comparison.protocol import (
    METHODS,
    SEARCH_SEEDS,
    SPLIT_SEED_BASE,
    TASKS,
    TASKS_BY_ID,
    ComparisonSpecError,
    TaskPin,
    canonical_protocol,
    identity_of,
    split_seed,
)
from research.comparison.specs import TaskBinding

DATA_SCHEMA = "istratde-comparison-data.v1"

#: Domain separator for the per-split row shuffle; versioned because
#: changing it changes every opaque id assignment.
SHUFFLE_DOMAIN = "istratde-comparison-rowshuffle.v1"

#: The approved split is sklearn's StratifiedGroupKFold; the split is
#: frozen, so the sklearn version is pinned and verified at runtime.
SKLEARN_PIN = "1.6.1"

#: Fold assignment (FR1): train folds 0/1/2, validation fold 3, final 4.
TRAIN_FOLDS: tuple[int, int, int] = (0, 1, 2)
VALIDATION_FOLD = 3
FINAL_FOLD = 4
NUM_FOLDS = 5
SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "final")

REPO_ROOT = Path(__file__).resolve().parents[2]
SEALED_ROOT = Path(__file__).resolve().parent / "sealed"
PUBLIC_ROOT = REPO_ROOT / "tasks" / "comparison"
EVALUATOR_TASKS_ROOT = REPO_ROOT / "research" / "evaluator" / "tasks"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / "data-manifest.v1.json"

#: Final-phase publication root (EXP-006): an IGNORED runtime tree. The
#: tracked search registry and worker-visible search data are NEVER
#: modified by final publication; final inputs, trusted final answers and
#: the phase-specific registry live only here.
FINAL_PUBLICATION_ROOT = (
    REPO_ROOT / ".runtime" / "comparison-istratde" / "final-publication"
)
FINAL_REGISTRY_FILENAME = "registry.final.v1.json"

#: Official UCI license for all three datasets (PRD References).
UCI_LICENSE = "CC BY 4.0"

_FETCH_TIMEOUT_SECONDS = 60


class ComparisonDataError(ValueError):
    """Machine-readable data preparation/validation failure (fail closed)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- official sources ----------------------------------------------------------
@dataclass(frozen=True)
class TaskSource:
    """Pinned official UCI source for one comparison task."""

    task_id: str
    url: str
    archive_name: str
    member_name: str
    converter: str

    def __post_init__(self) -> None:
        pin = TASKS_BY_ID.get(self.task_id)
        if pin is None:
            raise ComparisonSpecError("unknown_task")
        if not self.url.startswith("https://archive.ics.uci.edu/"):
            raise ComparisonDataError("source_url_not_official", self.url)
        if self.converter not in ("banknote", "wdbc", "spambase"):
            raise ComparisonDataError("unknown_converter", self.converter)

    @property
    def pin(self) -> TaskPin:
        return TASKS_BY_ID[self.task_id]


TASK_SOURCES: tuple[TaskSource, ...] = (
    TaskSource(
        task_id="banknote",
        url=(
            "https://archive.ics.uci.edu/static/public/267/"
            "banknote%2Bauthentication.zip"
        ),
        archive_name="banknote+authentication.zip",
        member_name="data_banknote_authentication.txt",
        converter="banknote",
    ),
    TaskSource(
        task_id="wdbc",
        url=(
            "https://archive.ics.uci.edu/static/public/17/"
            "breast%2Bcancer%2Bwisconsin%2Bdiagnostic.zip"
        ),
        archive_name="breast+cancer+wisconsin+diagnostic.zip",
        member_name="wdbc.data",
        converter="wdbc",
    ),
    TaskSource(
        task_id="spambase",
        url="https://archive.ics.uci.edu/static/public/94/spambase.zip",
        archive_name="spambase.zip",
        member_name="spambase.data",
        converter="spambase",
    ),
)
TASK_SOURCES_BY_ID = {source.task_id: source for source in TASK_SOURCES}


# --- strict conversion -----------------------------------------------------------
@dataclass(frozen=True)
class RawRow:
    """One converted row: numeric features plus a 0/1 label. No original id."""

    features: tuple[float, ...]
    label: int


def _parse_rows(
    text: str, *, width: int, task_id: str
) -> list[list[str]]:
    try:
        rows = [row for row in csv.reader(io.StringIO(text, newline=""), strict=True) if row]
    except csv.Error as exc:
        raise ComparisonDataError("malformed_data", f"{task_id}: csv parse: {exc}") from exc
    if not rows:
        raise ComparisonDataError("malformed_data", f"{task_id}: empty data")
    for index, row in enumerate(rows):
        if len(row) != width:
            raise ComparisonDataError(
                "malformed_data",
                f"{task_id}: row {index} has width {len(row)}, expected {width}",
            )
    return rows


def _parse_features(values: Sequence[str], *, task_id: str, row: int) -> tuple[float, ...]:
    features: list[float] = []
    for raw in values:
        token = raw.strip()
        if not token:
            raise ComparisonDataError(
                "malformed_data", f"{task_id}: row {row}: empty feature"
            )
        try:
            value = float(token)
        except ValueError:
            raise ComparisonDataError(
                "malformed_data", f"{task_id}: row {row}: non-numeric {raw!r}"
            ) from None
        if not math.isfinite(value):
            raise ComparisonDataError(
                "malformed_data", f"{task_id}: row {row}: non-finite {raw!r}"
            )
        features.append(value)
    return tuple(features)


def _parse_binary_label(token: str, *, task_id: str, row: int) -> int:
    stripped = token.strip()
    if stripped == "0":
        return 0
    if stripped == "1":
        return 1
    raise ComparisonDataError(
        "malformed_data", f"{task_id}: row {row}: label {token!r} not in {{0,1}}"
    )


def _convert_banknote(text: str) -> list[RawRow]:
    rows = _parse_rows(text, width=5, task_id="banknote")
    return [
        RawRow(
            features=_parse_features(row[:4], task_id="banknote", row=index),
            label=_parse_binary_label(row[4], task_id="banknote", row=index),
        )
        for index, row in enumerate(rows)
    ]


def _convert_wdbc(text: str) -> list[RawRow]:
    rows = _parse_rows(text, width=32, task_id="wdbc")
    converted: list[RawRow] = []
    for index, row in enumerate(rows):
        # Column 0 is the original WDBC id: removed, never propagated.
        diagnosis = row[1].strip()
        if diagnosis == "B":
            label = 0
        elif diagnosis == "M":
            label = 1
        else:
            raise ComparisonDataError(
                "malformed_data",
                f"wdbc: row {index}: diagnosis {row[1]!r} not in {{B,M}}",
            )
        converted.append(
            RawRow(
                features=_parse_features(row[2:], task_id="wdbc", row=index),
                label=label,
            )
        )
    return converted


def _convert_spambase(text: str) -> list[RawRow]:
    rows = _parse_rows(text, width=58, task_id="spambase")
    return [
        RawRow(
            features=_parse_features(row[:57], task_id="spambase", row=index),
            label=_parse_binary_label(row[57], task_id="spambase", row=index),
        )
        for index, row in enumerate(rows)
    ]


_CONVERTERS = {
    "banknote": _convert_banknote,
    "wdbc": _convert_wdbc,
    "spambase": _convert_spambase,
}

LABEL_CONVERSION_DOCS = {
    "banknote": "column 5 label used as-is; verified values in {0,1}",
    "wdbc": "diagnosis column mapped B=0, M=1",
    "spambase": "column 58 label used as-is; verified values in {0,1}",
}
ID_REMOVAL_DOCS = {
    "banknote": "source has no identifier column; original row index discarded",
    "wdbc": "original WDBC id column removed; original row index discarded",
    "spambase": "source has no identifier column; original row index discarded",
}


# --- grouping and split -------------------------------------------------------------
def assign_groups(rows: Sequence[RawRow]) -> list[int]:
    """Group id per row: identical numeric feature vectors share one group.

    Keys are parsed float tuples, so equivalent numeric spellings (``-0``
    vs ``0``, ``1`` vs ``1.0``) land in the same group. Contradictory-label
    duplicates stay in one group and are preserved. Group ids are assigned
    in order of first appearance (canonical raw order).
    """
    group_of: dict[tuple[float, ...], int] = {}
    assignment: list[int] = []
    for row in rows:
        key = row.features
        group_id = group_of.get(key)
        if group_id is None:
            group_id = len(group_of)
            group_of[key] = group_id
        assignment.append(group_id)
    return assignment


def assign_folds(labels: Sequence[int], groups: Sequence[int], seed: int) -> list[int]:
    """One fixed StratifiedGroupKFold(5, shuffle, random_state=seed) split.

    The sklearn version is pinned: a different version could change the
    frozen fold assignment, so drift fails closed instead of silently
    re-splitting.
    """
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold

    if sklearn.__version__ != SKLEARN_PIN:
        raise ComparisonDataError(
            "sklearn_version_mismatch",
            f"pinned {SKLEARN_PIN}, found {sklearn.__version__}",
        )
    n_rows = len(labels)
    if not (n_rows == len(groups) and n_rows >= NUM_FOLDS):
        raise ComparisonDataError("malformed_data", "label/group length mismatch")
    splitter = StratifiedGroupKFold(
        n_splits=NUM_FOLDS, shuffle=True, random_state=seed
    )
    folds = [-1] * n_rows
    dummy_x = [[0.0]] * n_rows
    for fold_index, (_, fold_rows) in enumerate(
        splitter.split(dummy_x, list(labels), list(groups))
    ):
        for row_index in fold_rows:
            if folds[row_index] != -1:
                raise ComparisonDataError("split_overlap", f"row {row_index}")
            folds[row_index] = fold_index
    if any(fold == -1 for fold in folds):
        raise ComparisonDataError("split_incomplete", "unassigned rows remain")
    return folds


def fold_split(fold: int) -> str:
    if fold in TRAIN_FOLDS:
        return "train"
    if fold == VALIDATION_FOLD:
        return "validation"
    if fold == FINAL_FOLD:
        return "final"
    raise ComparisonDataError("split_unknown_fold", str(fold))


# --- opaque ids ----------------------------------------------------------------------
def shuffle_seed(task_id: str, split: str) -> int:
    """Per-split shuffle seed: a function of task and split names ONLY.

    Never derived from labels or row content, so the id assignment cannot
    leak labels; the fresh sequential ids after the shuffle cannot leak the
    original row order either.
    """
    if split not in SPLIT_NAMES:
        raise ComparisonDataError("split_unknown_fold", split)
    digest = hashlib.sha256(
        f"{SHUFFLE_DOMAIN}|{task_id}|{split}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def assign_split_ids(task_id: str, split: str, row_count: int) -> list[int]:
    """Return ``row_of_id``: id k belongs to canonical row ``row_of_id[k]``.

    The permutation is a stdlib ``random.Random`` shuffle (Mersenne
    Twister, stable across CPython versions) of the positions 0..n-1,
    seeded by :func:`shuffle_seed`. Labels never enter this function.
    """
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
        raise ComparisonDataError("malformed_data", f"invalid row_count {row_count!r}")
    order = list(range(row_count))
    random.Random(shuffle_seed(task_id, split)).shuffle(order)
    return order


# --- canonical rendering ---------------------------------------------------------
def _render_float(value: float) -> str:
    # Shortest round-trip rendering; -0.0 is normalized so equivalent zero
    # representations render identically.
    if value == 0.0:
        value = 0.0
    return repr(value)


def _feature_header(n_features: int) -> list[str]:
    return [f"f{i}" for i in range(1, n_features + 1)]


def render_features_csv(ids: Sequence[int], features: Sequence[tuple[float, ...]]) -> bytes:
    n_features = len(features[0])
    header = ["id", *_feature_header(n_features)]
    lines = [",".join(header)]
    for row_id, row_features in zip(ids, features):
        if len(row_features) != n_features:
            raise ComparisonDataError("malformed_data", "ragged feature rows")
        lines.append(
            ",".join([str(row_id), *(_render_float(v) for v in row_features)])
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_labeled_csv(
    ids: Sequence[int], features: Sequence[tuple[float, ...]], labels: Sequence[int]
) -> bytes:
    n_features = len(features[0])
    header = ["id", *_feature_header(n_features), "label"]
    lines = [",".join(header)]
    for row_id, row_features, label in zip(ids, features, labels):
        lines.append(
            ",".join(
                [str(row_id), *(_render_float(v) for v in row_features), str(label)]
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_answer_csv(ids: Sequence[int], labels: Sequence[int]) -> bytes:
    lines = ["id,label"]
    lines.extend(f"{row_id},{label}" for row_id, label in zip(ids, labels))
    return ("\n".join(lines) + "\n").encode("utf-8")


# --- prepared task artifacts -----------------------------------------------------
@dataclass(frozen=True)
class PreparedSplit:
    """One split of one task with its opaque ids (canonical-order rows)."""

    name: str
    ids: tuple[int, ...]  # id per position, positions in shuffled order
    rows: tuple[RawRow, ...]  # rows in shuffled (id) order


@dataclass(frozen=True)
class PreparedTask:
    """Fully converted, grouped, split and id-assigned task (in memory)."""

    source: TaskSource
    archive_bytes: bytes
    member_bytes: bytes
    rows: tuple[RawRow, ...]  # canonical raw order
    groups: tuple[int, ...]
    folds: tuple[int, ...]
    splits: Mapping[str, PreparedSplit]

    @property
    def pin(self) -> TaskPin:
        return self.source.pin


def extract_member(archive_bytes: bytes, source: TaskSource) -> bytes:
    """Extract the pinned member from the official archive, fail closed."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as exc:
        raise ComparisonDataError("malformed_archive", source.task_id) from exc
    names = archive.namelist()
    if source.member_name not in names:
        raise ComparisonDataError(
            "malformed_archive",
            f"{source.task_id}: member {source.member_name} not in {names}",
        )
    return archive.read(source.member_name)


def _build_prepared(source: TaskSource, archive_bytes: bytes) -> PreparedTask:
    member = extract_member(archive_bytes, source)
    try:
        text = member.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ComparisonDataError("malformed_data", f"{source.task_id}: not utf-8") from exc
    rows = _CONVERTERS[source.converter](text)
    pin = source.pin
    if len(rows) != pin.rows or len(rows[0].features) != pin.features:
        raise ComparisonDataError(
            "task_shape_mismatch",
            f"{source.task_id}: {len(rows)}x{len(rows[0].features)} != "
            f"{pin.rows}x{pin.features}",
        )
    labels = [row.label for row in rows]
    groups = assign_groups(rows)
    folds = assign_folds(labels, groups, split_seed(pin))
    split_rows: dict[str, list[RawRow]] = {name: [] for name in SPLIT_NAMES}
    for row, fold in zip(rows, folds):
        split_rows[fold_split(fold)].append(row)
    splits: dict[str, PreparedSplit] = {}
    for name in SPLIT_NAMES:
        canonical = split_rows[name]
        order = assign_split_ids(source.task_id, name, len(canonical))
        shuffled = tuple(canonical[position] for position in order)
        split_labels = {row.label for row in shuffled}
        if split_labels != {0, 1}:
            raise ComparisonDataError(
                "split_missing_class", f"{source.task_id}:{name}: {split_labels}"
            )
        splits[name] = PreparedSplit(
            name=name, ids=tuple(range(len(shuffled))), rows=shuffled
        )
    return PreparedTask(
        source=source,
        archive_bytes=archive_bytes,
        member_bytes=member,
        rows=tuple(rows),
        groups=tuple(groups),
        folds=tuple(folds),
        splits=splits,
    )


def load_raw_archive(source: TaskSource, sealed_root: Path) -> bytes:
    path = sealed_root / source.task_id / "raw" / source.archive_name
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ComparisonDataError(
            "raw_missing", f"{source.task_id}: {path} ({exc})"
        ) from exc


def fetch_raw_archive(source: TaskSource) -> bytes:
    """Download the official archive once; caller persists it sealed."""
    request = urllib.request.Request(
        source.url, headers={"User-Agent": "openrsi-comparison-data/1.0"}
    )
    try:
        with urllib.request.urlopen(
            request, timeout=_FETCH_TIMEOUT_SECONDS
        ) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise ComparisonDataError(
                    "fetch_failed", f"{source.task_id}: HTTP {status}"
                )
            return response.read()
    except OSError as exc:
        raise ComparisonDataError("fetch_failed", f"{source.task_id}: {exc}") from exc


# --- artifact emission ------------------------------------------------------------
def _write_frozen(path: Path, data: bytes) -> None:
    """Write bytes; if the file exists it must be byte-identical (frozen).

    Durable: fsync of the file AND its parent directory. A symlinked target
    is never followed (an alias must not redirect evidence bytes).
    """
    if path.is_symlink():
        raise ComparisonDataError("artifact_mismatch", f"symlink: {path}")
    if path.exists():
        if path.read_bytes() != data:
            raise ComparisonDataError("artifact_mismatch", str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    dir_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _feature_rows(split: PreparedSplit) -> tuple[list[int], list[tuple[float, ...]]]:
    return list(split.ids), [row.features for row in split.rows]


def emit_task_artifacts(
    prepared: PreparedTask,
    *,
    repo_root: Path,
    public_root: Path,
    sealed_root: Path,
    evaluator_root: Path,
) -> dict[str, str]:
    """Write all task artifacts; return {repo-relative path: sha256}.

    Worker-visible public files carry NO validation/final labels and NO
    final features. Sealed files (canonical, final, train+validation refit)
    and the evaluator answer are written outside every worker mount.
    """
    task_id = prepared.source.task_id
    public_dir = public_root / task_id / "data" / "public"
    sealed_dir = sealed_root / task_id
    evaluator_dir = evaluator_root / f"comparison_{task_id}"

    train = prepared.splits["train"]
    validation = prepared.splits["validation"]
    final = prepared.splits["final"]

    train_ids, train_features = _feature_rows(train)
    val_ids, val_features = _feature_rows(validation)
    final_ids, final_features = _feature_rows(final)

    artifacts: dict[Path, bytes] = {}

    # Worker-visible (search phase): train labels + validation features only.
    artifacts[public_dir / "train.csv"] = render_labeled_csv(
        train_ids, train_features, [row.label for row in train.rows]
    )
    artifacts[public_dir / "test.csv"] = render_features_csv(val_ids, val_features)
    artifacts[public_dir / "sample_submission.csv"] = render_answer_csv(
        val_ids, [0] * len(val_ids)
    )

    # Trusted evaluator answer: validation labels, evaluator-private tree.
    artifacts[evaluator_dir / "validation_answer.csv"] = render_answer_csv(
        val_ids, [row.label for row in validation.rows]
    )

    # Sealed (never worker-visible): canonical audit data, final phase,
    # train+validation refit (published only behind the all-cell barrier).
    n_features = prepared.pin.features
    canonical_lines = [",".join(["group_id", *_feature_header(n_features), "label"])]
    for group_id, row in zip(prepared.groups, prepared.rows):
        canonical_lines.append(
            ",".join(
                [str(group_id), *(_render_float(v) for v in row.features), str(row.label)]
            )
        )
    artifacts[sealed_dir / "canonical.csv"] = (
        "\n".join(canonical_lines) + "\n"
    ).encode("utf-8")
    artifacts[sealed_dir / "final_features.csv"] = render_features_csv(
        final_ids, final_features
    )
    artifacts[sealed_dir / "final_labels.csv"] = render_answer_csv(
        final_ids, [row.label for row in final.rows]
    )
    # Refit file: train rows keep their ids; validation rows are offset so
    # ids stay unique in the combined file. Frozen until the barrier.
    refit_ids = train_ids + [len(train_ids) + row_id for row_id in val_ids]
    refit_features = train_features + val_features
    refit_labels = [row.label for row in train.rows] + [
        row.label for row in validation.rows
    ]
    artifacts[sealed_dir / "trainval.csv"] = render_labeled_csv(
        refit_ids, refit_features, refit_labels
    )

    repo_files: dict[str, str] = {}
    for path, data in artifacts.items():
        _write_frozen(path, data)
        repo_files[path.relative_to(repo_root).as_posix()] = _sha256(data)
    return repo_files


def _split_summary(split: PreparedSplit) -> dict[str, Any]:
    labels = [row.label for row in split.rows]
    return {
        "rows": len(split.rows),
        "class_counts": {"0": labels.count(0), "1": labels.count(1)},
    }


def task_manifest_entry(prepared: PreparedTask, files: Mapping[str, str]) -> dict[str, Any]:
    """Hash-bound manifest entry for one task, including its TaskBinding."""
    source = prepared.source
    pin = source.pin
    labels = [row.label for row in prepared.rows]
    group_label_sets: dict[int, set[int]] = {}
    group_sizes: dict[int, int] = {}
    for group_id, label in zip(prepared.groups, labels):
        group_label_sets.setdefault(group_id, set()).add(label)
        group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
    task_id = source.task_id
    binding = TaskBinding(
        task_id=task_id,
        split_seed=split_seed(pin),
        rows=pin.rows,
        features=pin.features,
        train_csv_sha256=files[f"tasks/comparison/{task_id}/data/public/train.csv"],
        validation_features_sha256=files[
            f"tasks/comparison/{task_id}/data/public/test.csv"
        ],
        validation_labels_sha256=files[
            f"research/evaluator/tasks/comparison_{task_id}/validation_answer.csv"
        ],
        final_features_sha256=files[
            f"research/comparison/sealed/{task_id}/final_features.csv"
        ],
        final_labels_sha256=files[
            f"research/comparison/sealed/{task_id}/final_labels.csv"
        ],
    )
    fold_vector = ",".join(str(fold) for fold in prepared.folds)
    group_vector = ",".join(str(group) for group in prepared.groups)
    return {
        "uci_id": pin.uci_id,
        "url": source.url,
        "license": UCI_LICENSE,
        "archive": {
            "name": source.archive_name,
            "sha256": _sha256(prepared.archive_bytes),
            "member": source.member_name,
            "member_sha256": _sha256(prepared.member_bytes),
        },
        "rows": pin.rows,
        "features": pin.features,
        "label_conversion": LABEL_CONVERSION_DOCS[task_id],
        "id_removal": ID_REMOVAL_DOCS[task_id],
        "class_counts": {"0": labels.count(0), "1": labels.count(1)},
        "groups": {
            "count": len(group_label_sets),
            "multi_row_groups": sum(1 for size in group_sizes.values() if size > 1),
            "contradictory_label_groups": sum(
                1 for label_set in group_label_sets.values() if len(label_set) > 1
            ),
        },
        "split_seed": split_seed(pin),
        "split_folds": {
            "train": list(TRAIN_FOLDS),
            "validation": VALIDATION_FOLD,
            "final": FINAL_FOLD,
        },
        "fold_assignment_sha256": _sha256(fold_vector.encode("utf-8")),
        "group_assignment_sha256": _sha256(group_vector.encode("utf-8")),
        "sklearn_version": SKLEARN_PIN,
        "splits": {name: _split_summary(prepared.splits[name]) for name in SPLIT_NAMES},
        "files": dict(files),
        "task_binding": binding.to_dict(),
        "task_binding_identity": binding.identity(),
    }


def build_manifest(entries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": DATA_SCHEMA,
        "split_seed_base": SPLIT_SEED_BASE,
        "shuffle_domain": SHUFFLE_DOMAIN,
        "id_scheme": (
            "per-split deterministic stdlib random.Random shuffle seeded by "
            "sha256(shuffle_domain|task|split) (label-independent), then "
            "sequential ids 0..n-1 in shuffled order; original identifiers "
            "and row indices never propagate"
        ),
        "float_rendering": "python repr() shortest round-trip; -0.0 normalized to 0.0",
        "sklearn_version": SKLEARN_PIN,
        "tasks": {task_id: dict(entry) for task_id, entry in entries.items()},
    }


def manifest_identity(manifest: Mapping[str, Any]) -> str:
    return identity_of(manifest)


def prepare_all(
    *,
    fetch: bool,
    repo_root: Path = REPO_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Prepare all three tasks end-to-end; idempotent and fail closed.

    Missing raw archives are downloaded from the pinned official URLs when
    ``fetch`` is true (never otherwise). Every existing artifact must be
    byte-identical to the freshly computed bytes; any difference or a
    differing existing manifest raises instead of overwriting — the split
    and ids are frozen once written.
    """
    entries: dict[str, Mapping[str, Any]] = {}
    for source in TASK_SOURCES:
        sealed_root = repo_root / "research" / "comparison" / "sealed"
        raw_dir = sealed_root / source.task_id / "raw"
        raw_path = raw_dir / source.archive_name
        if raw_path.exists():
            archive_bytes = raw_path.read_bytes()
        elif fetch:
            archive_bytes = fetch_raw_archive(source)
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(archive_bytes)
        else:
            raise ComparisonDataError("raw_missing", str(raw_path))
        prepared = _build_prepared(source, archive_bytes)
        files = emit_task_artifacts(
            prepared,
            repo_root=repo_root,
            public_root=repo_root / "tasks" / "comparison",
            sealed_root=sealed_root,
            evaluator_root=repo_root / "research" / "evaluator" / "tasks",
        )
        entries[source.task_id] = task_manifest_entry(prepared, files)
        _write_provenance(repo_root, source, archive_bytes)
    manifest = build_manifest(entries)
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ComparisonDataError(
                "manifest_frozen_mismatch", str(manifest_path)
            )
    else:
        manifest_path.write_text(rendered, encoding="utf-8")
    return manifest


def _write_provenance(
    repo_root: Path, source: TaskSource, archive_bytes: bytes
) -> None:
    """Human-facing fetch provenance; NOT part of hashed data identity."""
    path = (
        repo_root
        / "research"
        / "comparison"
        / "sealed"
        / source.task_id
        / "raw"
        / "provenance.json"
    )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("archive_sha256") != _sha256(archive_bytes):
            raise ComparisonDataError("artifact_mismatch", str(path))
        return
    payload = {
        "task_id": source.task_id,
        "uci_id": source.pin.uci_id,
        "url": source.url,
        "license": UCI_LICENSE,
        "archive_name": source.archive_name,
        "archive_sha256": _sha256(archive_bytes),
        "member_name": source.member_name,
        "fetched_on": "2026-10-10",
        "note": (
            "Official UCI archive, fetched from the pinned URL above; raw "
            "bytes and this provenance stay sealed (never worker-mounted)."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --- manifest loading and bindings ---------------------------------------------------
def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonDataError("manifest_invalid", f"{manifest_path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != DATA_SCHEMA:
        raise ComparisonDataError("manifest_invalid", "schema mismatch")
    tasks = data.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != {task.task_id for task in TASKS}:
        raise ComparisonDataError("manifest_invalid", "task set mismatch")
    return data


def load_task_bindings(path: Path | str | None = None) -> dict[str, TaskBinding]:
    """Reconstruct the hash-bound TaskBinding of each task from the manifest.

    This is a trusted loading boundary: beyond TaskBinding's own per-field
    validation (task pin, split seed, shape, hash formats), the manifest
    must be internally CONSISTENT — the declared binding identity must match
    the recomputed one, the binding must sit under its own task id, the
    five binding hashes must equal the manifest's file hashes, and the
    split seed/folds/sklearn version metadata must equal the frozen
    protocol values. Any disagreement fails closed.
    """
    manifest = load_manifest(path)
    if manifest.get("sklearn_version") != SKLEARN_PIN:
        raise ComparisonDataError(
            "manifest_invalid",
            f"sklearn_version {manifest.get('sklearn_version')!r} != pinned {SKLEARN_PIN}",
        )
    expected_folds = {
        "train": list(TRAIN_FOLDS),
        "validation": VALIDATION_FOLD,
        "final": FINAL_FOLD,
    }
    bindings: dict[str, TaskBinding] = {}
    for task_id, entry in manifest["tasks"].items():
        if not isinstance(entry, dict):
            raise ComparisonDataError("manifest_invalid", f"{task_id}: entry not an object")
        pin = TASKS_BY_ID[task_id]
        if entry.get("uci_id") != pin.uci_id:
            raise ComparisonDataError("manifest_invalid", f"{task_id}: uci_id mismatch")
        if entry.get("rows") != pin.rows or entry.get("features") != pin.features:
            raise ComparisonDataError("manifest_invalid", f"{task_id}: shape mismatch")
        if entry.get("split_seed") != split_seed(pin):
            raise ComparisonDataError("manifest_invalid", f"{task_id}: split_seed mismatch")
        if entry.get("split_folds") != expected_folds:
            raise ComparisonDataError("manifest_invalid", f"{task_id}: split_folds mismatch")
        raw_binding = entry.get("task_binding")
        if not isinstance(raw_binding, dict):
            raise ComparisonDataError("manifest_invalid", f"{task_id}: no binding")
        try:
            binding = TaskBinding(**raw_binding)
        except (ComparisonSpecError, TypeError) as exc:
            raise ComparisonDataError(
                "manifest_invalid", f"{task_id}: binding rejected: {exc}"
            ) from exc
        if binding.task_id != task_id:
            raise ComparisonDataError(
                "manifest_invalid", f"{task_id}: foreign binding {binding.task_id}"
            )
        if entry.get("task_binding_identity") != binding.identity():
            raise ComparisonDataError(
                "manifest_invalid", f"{task_id}: stale binding identity"
            )
        files = entry.get("files")
        if not isinstance(files, dict):
            raise ComparisonDataError("manifest_invalid", f"{task_id}: no files map")
        binding_file_hashes = {
            f"tasks/comparison/{task_id}/data/public/train.csv": binding.train_csv_sha256,
            f"tasks/comparison/{task_id}/data/public/test.csv": binding.validation_features_sha256,
            f"research/evaluator/tasks/comparison_{task_id}/validation_answer.csv": binding.validation_labels_sha256,
            f"research/comparison/sealed/{task_id}/final_features.csv": binding.final_features_sha256,
            f"research/comparison/sealed/{task_id}/final_labels.csv": binding.final_labels_sha256,
        }
        for rel_path, expected_hash in binding_file_hashes.items():
            if files.get(rel_path) != expected_hash:
                raise ComparisonDataError(
                    "manifest_invalid",
                    f"{task_id}: binding/file hash disagreement at {rel_path}",
                )
        bindings[task_id] = binding
    return bindings


def _read_bound_bytes(path: Path, expected_sha256: str, rule: str) -> bytes:
    """Read bytes that must exist (never via symlink) and hash exactly so."""
    if path.is_symlink() or not path.exists():
        raise ComparisonDataError("final_source_missing", str(path))
    data = path.read_bytes()
    if _sha256(data) != expected_sha256:
        raise ComparisonDataError(rule, str(path))
    return data


def _refit_trainval_bytes(
    train_csv: bytes, validation_features_csv: bytes, validation_answer_csv: bytes
) -> bytes:
    """Recompute the EXACT sealed train+validation refit from the
    binding-pinned artifacts: train rows keep their ids; validation ids are
    offset past the train ids (mirrors emit_task_artifacts)."""

    def _rows(
        data: bytes, *, labeled: bool
    ) -> tuple[list[int], list[tuple[float, ...]], list[int]]:
        try:
            reader = csv.reader(io.StringIO(data.decode("utf-8")))
            header = next(reader, None)
            if (
                header is None
                or len(header) < 2
                or header[0] != "id"
                or (header[-1] == "label") is not labeled
            ):
                raise ValueError("header")
            ids: list[int] = []
            features: list[tuple[float, ...]] = []
            labels: list[int] = []
            for row in reader:
                ids.append(int(row[0]))
                if labeled:
                    features.append(tuple(float(v) for v in row[1:-1]))
                    labels.append(int(row[-1]))
                else:
                    features.append(tuple(float(v) for v in row[1:]))
            return ids, features, labels
        except (ValueError, IndexError) as exc:
            raise ComparisonDataError(
                "final_source_tampered", "refit inputs unparsable"
            ) from exc

    train_ids, train_features, train_labels = _rows(train_csv, labeled=True)
    val_ids, val_features, _ = _rows(validation_features_csv, labeled=False)
    answer_ids, _, answer_labels = _rows(validation_answer_csv, labeled=True)
    answers = dict(zip(answer_ids, answer_labels))
    try:
        refit_labels = train_labels + [answers[row_id] for row_id in val_ids]
    except KeyError as exc:
        raise ComparisonDataError(
            "final_source_tampered", "validation answer ids mismatch"
        ) from exc
    refit_ids = train_ids + [len(train_ids) + row_id for row_id in val_ids]
    refit_features = train_features + val_features
    if not refit_features:
        raise ComparisonDataError("final_source_tampered", "empty refit")
    return render_labeled_csv(refit_ids, refit_features, refit_labels)


#: Mirrors orchestrate.BARRIER_SCHEMA (data.py cannot import orchestrate:
#: orchestrate -> runner -> data would be a cycle). A test pins equality.
EXPECTED_BARRIER_SCHEMA = "istratde-comparison-barrier.v1"

_FINAL_BARRIER_STATUSES = frozenset({"frozen", "failed_no_valid_program"})


def validate_barrier_evidence(evidence: object) -> dict[str, Any]:
    """Structural fail-closed validation of all-cell barrier evidence.

    The STRENGTH of the barrier comes from orchestrate.verify_final_barrier
    having re-hashed every terminal artifact; this boundary re-checks the
    payload's schema, roster completeness, terminal statuses, winner hash
    formats and self-consistent identity so a caller-created dictionary
    cannot pass publication on its own.
    """
    if not isinstance(evidence, dict):
        raise ComparisonDataError("final_barrier_not_frozen", "evidence not a mapping")
    if evidence.get("schema") != EXPECTED_BARRIER_SCHEMA:
        raise ComparisonDataError("final_barrier_not_frozen", "barrier schema mismatch")
    identity = evidence.get("barrier_identity")
    body = {k: v for k, v in evidence.items() if k != "barrier_identity"}
    if not isinstance(identity, str) or identity != identity_of(body):
        raise ComparisonDataError("final_barrier_not_frozen", "barrier identity mismatch")
    commit = evidence.get("source_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or not set(commit) <= set("0123456789abcdef")
    ):
        raise ComparisonDataError("final_barrier_not_frozen", "barrier commit invalid")
    for key in ("schedule_identity", "experiment_identity"):
        value = evidence.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or not set(value) <= set("0123456789abcdef")
        ):
            raise ComparisonDataError(
                "final_barrier_not_frozen", f"barrier {key} invalid"
            )
    cells = evidence.get("cells")
    canonical_keys = {
        f"{task.task_id}:{seed}:{method}"
        for task in TASKS
        for seed in SEARCH_SEEDS
        for method in METHODS
    }
    if not isinstance(cells, dict) or set(cells) != canonical_keys:
        raise ComparisonDataError("final_barrier_not_frozen", "barrier roster invalid")
    for cell_key, entry in cells.items():
        if not isinstance(entry, dict) or entry.get("terminal_status") not in (
            _FINAL_BARRIER_STATUSES
        ):
            raise ComparisonDataError(
                "final_barrier_not_frozen", f"{cell_key}: terminal status invalid"
            )
        winner = entry.get("winner")
        if entry["terminal_status"] == "failed_no_valid_program":
            if winner is not None:
                raise ComparisonDataError(
                    "final_barrier_not_frozen", f"{cell_key}: unexpected winner"
                )
            continue
        if not isinstance(winner, dict):
            raise ComparisonDataError(
                "final_barrier_not_frozen", f"{cell_key}: winner missing"
            )
        config_index = winner.get("config_index")
        if (
            isinstance(config_index, bool)
            or not isinstance(config_index, int)
            or not 0 <= config_index < 8
        ):
            raise ComparisonDataError(
                "final_barrier_not_frozen", f"{cell_key}: winner config invalid"
            )
        score = winner.get("validation_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ComparisonDataError(
                "final_barrier_not_frozen", f"{cell_key}: winner score invalid"
            )
        for field in ("code_sha256", "source_sha256", "prediction_sha256"):
            value = winner.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or not set(value) <= set("0123456789abcdef")
            ):
                raise ComparisonDataError(
                    "final_barrier_not_frozen", f"{cell_key}: winner {field} invalid"
                )
        if not isinstance(winner.get("job_id"), str) or not winner["job_id"]:
            raise ComparisonDataError(
                "final_barrier_not_frozen", f"{cell_key}: winner job invalid"
            )
    return evidence


def publish_final_refit(
    *,
    orchestrator: Any = None,
    ledger: Any = None,
    barrier_evidence: object | None = None,
    repo_root: Path = REPO_ROOT,
    sealed_root: Path = SEALED_ROOT,
    publication_root: Path | None = None,
    manifest_path: Path | None = None,
    # Binding anchors for the trainval refit proof: the worker-visible
    # public tree root (<task>/data/public/{train,test}.csv) and the
    # evaluator tree root (tasks/comparison_<task>/validation_answer.csv).
    public_root: Path | None = None,
    evaluator_root: Path | None = None,
    registry_path: Path | None = None,
    # Superseded inputs of the removed partial verifier; accepted only so
    # legacy callers keep failing closed with ComparisonDataError.
    evidence_root: Path | None = None,
    ledger_evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Publish the final-phase data behind the verified all-cell barrier.

    Provenance is established ONLY by the FULL shared verification: the
    caller must supply the actual ``Orchestrator`` and the open ORIGINAL
    comparison ledger; this function re-runs
    ``orchestrator.verify_final_barrier(ledger)`` itself (complete roster,
    per-config evidence, original per-run ledger accounting, zero open
    reservations, orchestrator-derived experiment identity) and requires
    the durable final-phase transition to be bound to that exact barrier
    BEFORE any sealed byte is read or written. Absent context, a stale or
    caller-created barrier, an unentered/mismatched phase, an expired
    original deadline, or a HEAD drift all refuse closed.

    Publication NEVER touches tracked search artifacts: the frozen search
    registry (``research/evaluator/registry.v1.json``) and worker-visible
    search data stay byte-identical. Before ANY sealed read or public write
    the supplied manifest is bound to the ORIGINAL frozen cell bindings
    (per-task ``task_binding``/identity entries and all five bound file
    hashes must equal the frozen ``CellSpec`` binding fields, and the sealed
    ``trainval.csv`` must be the exact refit of the binding-pinned
    train/validation artifacts); a self-consistent replacement manifest that
    disagrees is refused. Final inputs (train+validation refit as
    ``train.csv``, final features as ``test.csv``, zero-label
    ``sample_submission.csv``), the trusted ``final_answer.csv`` files and
    the phase-specific registry (``registry.final.v1.json`` with the three
    ``comparison_<task>_final`` entries, split ``final``) are written to
    the ignored runtime publication root ONLY, hash-bound to the frozen
    manifest and the barrier. Existing different bytes refuse closed.

    Returns the publication record (path -> sha256, phase registry path and
    version, barrier/manifest identities).
    """
    if orchestrator is None or ledger is None:
        # No partial metadata substitute exists: without the real
        # orchestrator + original ledger context, refuse BEFORE any sealed
        # or manifest access.
        raise ComparisonDataError(
            "final_barrier_unproven", "orchestrator and ledger context required"
        )
    if barrier_evidence is not None:
        # Structural sanity of the caller hint only; never sufficient.
        hint = validate_barrier_evidence(barrier_evidence)
    else:
        hint = None
    try:
        fresh = orchestrator.verify_final_barrier(ledger)
    except Exception as exc:
        raise ComparisonDataError("final_barrier_unproven", str(exc)) from exc
    if hint is not None and hint != fresh:
        raise ComparisonDataError("final_barrier_stale", "barrier evidence mismatch")
    phase = getattr(ledger, "final_phase", None)
    if not isinstance(phase, dict) or phase.get("barrier_identity") != fresh[
        "barrier_identity"
    ]:
        raise ComparisonDataError(
            "final_phase_not_entered", "irreversible phase transition missing"
        )
    deadline = ledger.deadline_epoch
    if deadline is not None and float(ledger.now_fn()) > float(deadline):
        raise ComparisonDataError("final_deadline_expired", "original 24h deadline")
    from research.comparison import deploy  # noqa: PLC0415

    try:
        head = deploy.git_head_commit(repo_root)
    except Exception as exc:
        raise ComparisonDataError("final_source_unproven", str(exc)) from exc
    if head != fresh["source_commit"]:
        raise ComparisonDataError("final_source_unproven", "HEAD != barrier commit")

    manifest = load_manifest(manifest_path)
    manifest_tasks = manifest.get("tasks")
    if not isinstance(manifest_tasks, dict):
        raise ComparisonDataError("manifest_invalid", "no tasks map")
    publication_root = (
        Path(publication_root) if publication_root is not None else FINAL_PUBLICATION_ROOT
    )

    # Bind the supplied manifest to the ORIGINAL frozen cell bindings BEFORE
    # any sealed byte is read or public byte written. A self-consistent
    # replacement manifest whose task_binding entries or bound file hashes
    # disagree with the frozen CellSpec bindings is refused here (recording
    # its identity afterwards would prove nothing), and the sealed trainval
    # must be the EXACT refit of the binding-pinned train/validation
    # artifacts — the original immutable manifest including trainval.
    bindings_by_task: dict[str, Any] = {}
    for cell in orchestrator.cells:
        existing = bindings_by_task.setdefault(cell.task_id, cell.task_binding)
        if existing != cell.task_binding:
            raise ComparisonDataError(
                "manifest_binding_mismatch",
                f"{cell.task_id}: non-uniform frozen binding",
            )
    public_base = (
        Path(public_root) if public_root is not None else REPO_ROOT / "tasks" / "comparison"
    )
    evaluator_base = (
        Path(evaluator_root)
        if evaluator_root is not None
        else REPO_ROOT / "research" / "evaluator"
    )
    for task in TASKS:
        task_id = task.task_id
        binding = bindings_by_task[task_id]
        entry = manifest_tasks.get(task_id)
        if not isinstance(entry, dict) or not isinstance(entry.get("files"), dict):
            raise ComparisonDataError("manifest_invalid", f"{task_id}: no files map")
        files = entry["files"]
        if (
            entry.get("task_binding") != binding.to_dict()
            or entry.get("task_binding_identity") != binding.identity()
        ):
            raise ComparisonDataError(
                "manifest_binding_mismatch", f"{task_id}: frozen binding replaced"
            )
        bound_files = {
            f"tasks/comparison/{task_id}/data/public/train.csv": binding.train_csv_sha256,
            f"tasks/comparison/{task_id}/data/public/test.csv": binding.validation_features_sha256,
            f"research/evaluator/tasks/comparison_{task_id}/validation_answer.csv": binding.validation_labels_sha256,
            f"research/comparison/sealed/{task_id}/final_features.csv": binding.final_features_sha256,
            f"research/comparison/sealed/{task_id}/final_labels.csv": binding.final_labels_sha256,
        }
        for rel_path, expected_hash in bound_files.items():
            if files.get(rel_path) != expected_hash:
                raise ComparisonDataError(
                    "manifest_binding_mismatch",
                    f"{task_id}: {rel_path} disagrees with the frozen binding",
                )
        train = _read_bound_bytes(
            public_base / task_id / "data" / "public" / "train.csv",
            binding.train_csv_sha256,
            "manifest_binding_mismatch",
        )
        validation_features = _read_bound_bytes(
            public_base / task_id / "data" / "public" / "test.csv",
            binding.validation_features_sha256,
            "manifest_binding_mismatch",
        )
        validation_answer = _read_bound_bytes(
            evaluator_base / "tasks" / f"comparison_{task_id}" / "validation_answer.csv",
            binding.validation_labels_sha256,
            "manifest_binding_mismatch",
        )
        trainval_entry = files.get(f"research/comparison/sealed/{task_id}/trainval.csv")
        if not isinstance(trainval_entry, str):
            raise ComparisonDataError(
                "manifest_invalid", f"{task_id}: no trainval file entry"
            )
        trainval = _read_bound_bytes(
            sealed_root / task_id / "trainval.csv",
            trainval_entry,
            "final_source_tampered",
        )
        if trainval != _refit_trainval_bytes(
            train, validation_features, validation_answer
        ):
            raise ComparisonDataError(
                "manifest_binding_mismatch",
                f"{task_id}: trainval is not the bound train+validation refit",
            )

    written: dict[str, str] = {}
    for task in TASKS:
        task_id = task.task_id
        sealed_dir = sealed_root / task_id
        entry = manifest_tasks.get(task_id)
        files = entry.get("files") if isinstance(entry, dict) else None
        if not isinstance(files, dict):
            raise ComparisonDataError("manifest_invalid", f"{task_id}: no files map")

        def _sealed_bytes(name: str) -> bytes:
            path = sealed_dir / name
            if path.is_symlink() or not path.exists():
                raise ComparisonDataError("final_source_missing", str(path))
            data = path.read_bytes()
            expected = files.get(f"research/comparison/sealed/{task_id}/{name}")
            if _sha256(data) != expected:
                raise ComparisonDataError("final_source_tampered", f"{task_id}/{name}")
            return data

        trainval = _sealed_bytes("trainval.csv")
        final_features = _sealed_bytes("final_features.csv")
        final_labels = _sealed_bytes("final_labels.csv")

        # Zero-label sample submission derived from the final ids ONLY —
        # final labels never enter worker-visible files.
        reader = csv.DictReader(io.StringIO(final_features.decode("utf-8")))
        if reader.fieldnames is None or reader.fieldnames[0] != "id":
            raise ComparisonDataError("final_source_tampered", f"{task_id}: header")
        sample_lines = ["id,label"]
        for row in reader:
            sample_lines.append(f"{row['id']},0")
        sample = ("\n".join(sample_lines) + "\n").encode("utf-8")

        final_dir = publication_root / task_id / "data" / "final"
        outputs = {
            final_dir / "train.csv": trainval,
            final_dir / "test.csv": final_features,
            final_dir / "sample_submission.csv": sample,
            publication_root
            / "answers"
            / f"comparison_{task_id}"
            / "final_answer.csv": final_labels,
        }
        for path, data in outputs.items():
            _write_frozen(path, data)
            written[path.relative_to(publication_root).as_posix()] = _sha256(data)

    # Phase-specific registry: ONLY the three final entries (the tracked
    # search registry is read once for its schema/version but never copied
    # wholesale — search-phase answers live in the tracked evaluator tree
    # and must not be republished). Bound to the barrier via version pin.
    from research.evaluator.service import (  # noqa: PLC0415
        DEFAULT_REGISTRY_PATH,
        EvaluatorRegistry,
    )

    source_registry_file = registry_path or DEFAULT_REGISTRY_PATH
    try:
        registry_bytes = source_registry_file.read_bytes()
        registry_payload = json.loads(registry_bytes.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise ComparisonDataError("registry_unavailable", str(exc)) from exc
    if not isinstance(registry_payload.get("tasks"), dict):
        raise ComparisonDataError("registry_unavailable", "no tasks map")
    base_version = registry_payload.get("registry_version")
    if not isinstance(base_version, str) or not base_version:
        raise ComparisonDataError("registry_unavailable", "no registry_version")
    final_tasks: dict[str, Any] = {}
    for task in TASKS:
        task_id = task.task_id
        answer_rel = f"answers/comparison_{task_id}/final_answer.csv"
        final_tasks[f"comparison_{task_id}_final"] = {
            "metric": "accuracy",
            "split": "final",
            "answer_file": answer_rel,
            "answer_sha256": written[answer_rel],
            "id_column": "id",
            "target_column": "label",
            "label_values": [0, 1],
            "max_prediction_bytes": 1048576,
            "max_rows": 100000,
            "worker_data_public_path": f"/mnt/rsi_data/comparison/{task_id}/data/final",
        }
    phase_registry = {
        "schema_version": registry_payload.get("schema_version"),
        "tasks": final_tasks,
        "registry_version": (
            f"{base_version}+final.{fresh['barrier_identity'][:16]}"
        ),
    }
    phase_registry_bytes = (
        json.dumps(phase_registry, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    phase_registry_path = publication_root / FINAL_REGISTRY_FILENAME
    _write_frozen(phase_registry_path, phase_registry_bytes)
    # Fail closed on ANY registry inconsistency in the written phase file.
    phase_registry_loaded = EvaluatorRegistry.load(phase_registry_path)
    # The tracked search registry must remain byte-identical.
    try:
        if source_registry_file.read_bytes() != registry_bytes:
            raise ComparisonDataError(
                "final_publication_conflict", "search registry mutated"
            )
    except OSError as exc:
        raise ComparisonDataError("registry_unavailable", str(exc)) from exc
    written[FINAL_REGISTRY_FILENAME] = _sha256(phase_registry_bytes)
    return {
        "schema": "istratde-comparison-final-publication.v1",
        "barrier_identity": fresh["barrier_identity"],
        "schedule_identity": fresh["schedule_identity"],
        "experiment_identity": fresh["experiment_identity"],
        "source_commit": fresh["source_commit"],
        "protocol_identity": canonical_protocol().identity(),
        "manifest_identity": manifest_identity(manifest),
        "publication_root": str(publication_root.resolve()),
        "registry_file": FINAL_REGISTRY_FILENAME,
        "registry_version": phase_registry_loaded.registry_version,
        "files": written,
    }


def main() -> int:
    manifest = prepare_all(fetch=True)
    summary = {
        task_id: {
            "rows": entry["rows"],
            "features": entry["features"],
            "groups": entry["groups"],
            "splits": entry["splits"],
            "task_binding_identity": entry["task_binding_identity"],
        }
        for task_id, entry in manifest["tasks"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
