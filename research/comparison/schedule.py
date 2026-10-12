"""Frozen interleaved 216-position execution schedule (EXP-006).

The approved protocol (PRD: "Freeze schedule before generation") requires a
deterministic schedule, serialized before any live start:

- config index outermost (0..7);
- inside one config index, one block per (task, search seed) in canonical
  order (task major, seed minor) — 9 blocks;
- inside one (config index, block), the three methods run in a rotated
  order; the rotation offset is a pure function of the config index and the
  block index: ``offset = (config_index + block_index) % 3`` over the
  canonical method order FIXED, RANDOM, ISTRATDE;
- because config index is outermost, every cell's first four config
  positions (the initial population batch) precede its fifth position
  (the feedback batch) — asserted structurally at build AND load.

The schedule also freezes the outer seed derivations (native iStratDE seed
per (task, search seed); RANDOM remainder seed per (task, search seed)) so
the manifest alone determines every paired vector source. Method never
enters either derivation: the three methods at one (task, search seed)
position are paired by construction.

The frozen artifact is written once and never silently replaced: freezing
onto existing different bytes fails closed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.comparison.protocol import (
    CONFIGS_PER_CELL,
    METHODS,
    NUM_INNER_RUNS,
    ComparisonSpecError,
    canonical_json,
    canonical_protocol,
    identity_of,
)
from research.comparison.specs import CellSpec

SCHEDULE_SCHEMA = "istratde-comparison-schedule.v1"

#: Domain separators for the outer seed derivations frozen in the schedule.
NATIVE_SEED_DOMAIN = "istratde-comparison-native-seed.v1"
RANDOM_REMAINDER_SEED_DOMAIN = "istratde-comparison-random-remainder.v1"

#: Number of configs per cell that form the paired initial population batch
#: (native pop_size = 4; configs 0..3 initial, configs 4..7 feedback).
INITIAL_BATCH_CONFIGS = 4

#: Frozen common per-job timeout for every inner run; bound into the
#: schedule manifest and every pre-launch intent (EXP-006 supervisor:
#: freeze the common timeout, reject drift before launch and on resume).
JOB_TIMEOUT_SECONDS = 600

_BATCH_INITIAL = "initial"
_BATCH_FEEDBACK = "feedback"
_BATCHES = frozenset({_BATCH_INITIAL, _BATCH_FEEDBACK})


def _derive_seed(domain: str, task_id: str, search_seed: int) -> int:
    digest = hashlib.sha256(
        f"{domain}|{task_id}|{search_seed}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def derive_native_seed(task_id: str, search_seed: int) -> int:
    """Native iStratDE seed for one (task, search seed) pair.

    Method- and config-independent by design: the pop_size=4 first ask of
    this seed IS the paired initial population for RANDOM and ISTRATDE
    (PRD: RANDOM initial four equal the paired iStratDE initial vectors).
    """
    _validate_pair(task_id, search_seed)
    return _derive_seed(NATIVE_SEED_DOMAIN, task_id, search_seed)


def derive_random_remainder_seed(task_id: str, search_seed: int) -> int:
    """Stdlib RNG seed for the RANDOM method's four remainder vectors.

    Method-independent derivation kept for audit symmetry; only RANDOM
    consumes it. Never a function of any score.
    """
    _validate_pair(task_id, search_seed)
    return _derive_seed(RANDOM_REMAINDER_SEED_DOMAIN, task_id, search_seed)


def _validate_pair(task_id: str, search_seed: int) -> None:
    proto = canonical_protocol()
    if task_id not in {task.task_id for task in proto.tasks}:
        raise ComparisonSpecError("unknown_task")
    if (
        isinstance(search_seed, bool)
        or not isinstance(search_seed, int)
        or search_seed not in proto.search_seeds
    ):
        raise ComparisonSpecError("unknown_search_seed")


def config_batch(config_index: int) -> str:
    """The batch label of a config index (0..3 initial, 4..7 feedback)."""
    if (
        isinstance(config_index, bool)
        or not isinstance(config_index, int)
        or not 0 <= config_index < CONFIGS_PER_CELL
    ):
        raise ComparisonSpecError("config_index_out_of_range")
    return _BATCH_INITIAL if config_index < INITIAL_BATCH_CONFIGS else _BATCH_FEEDBACK


@dataclass(frozen=True)
class SchedulePosition:
    """One execution position of the frozen schedule."""

    position: int
    cell_key: str
    config_index: int
    batch: str

    @property
    def run_key(self) -> str:
        return f"{self.cell_key}:{self.config_index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "run_key": self.run_key,
            "cell_key": self.cell_key,
            "config_index": self.config_index,
            "batch": self.batch,
        }


def _method_rotation(config_index: int, block_index: int) -> tuple[str, str, str]:
    """Deterministic method rotation for one (config, block) slot."""
    offset = (config_index + block_index) % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]  # type: ignore[return-value]


def _block_order() -> tuple[tuple[str, int], ...]:
    """The 9 (task, search seed) blocks in canonical order."""
    proto = canonical_protocol()
    return tuple(
        (task.task_id, seed) for task in proto.tasks for seed in proto.search_seeds
    )


def build_schedule(cells: Sequence[CellSpec]) -> tuple[SchedulePosition, ...]:
    """Build the 216-position schedule from the exact 27-cell roster."""
    proto = canonical_protocol()
    if len(cells) != proto.num_cells:
        raise ComparisonSpecError("schedule_roster_invalid")
    by_key: dict[str, CellSpec] = {}
    for cell in cells:
        if not isinstance(cell, CellSpec):
            raise ComparisonSpecError("schedule_roster_invalid")
        if cell.cell_key() in by_key:
            raise ComparisonSpecError("schedule_roster_invalid")
        by_key[cell.cell_key()] = cell
    expected_keys = {
        f"{task.task_id}:{seed}:{method}"
        for task in proto.tasks
        for seed in proto.search_seeds
        for method in proto.methods
    }
    if set(by_key) != expected_keys:
        raise ComparisonSpecError("schedule_roster_invalid")

    blocks = _block_order()
    positions: list[SchedulePosition] = []
    for config_index in range(CONFIGS_PER_CELL):
        for block_index, (task_id, seed) in enumerate(blocks):
            for method in _method_rotation(config_index, block_index):
                cell_key = f"{task_id}:{seed}:{method}"
                positions.append(
                    SchedulePosition(
                        position=len(positions),
                        cell_key=cell_key,
                        config_index=config_index,
                        batch=config_batch(config_index),
                    )
                )
    result = tuple(positions)
    _validate_schedule(result, expected_keys)
    return result


def _validate_schedule(
    positions: Sequence[SchedulePosition], expected_cell_keys: set[str]
) -> None:
    """Structural invariants; run at build time AND on every load."""
    if len(positions) != NUM_INNER_RUNS:
        raise ComparisonSpecError("schedule_length_invalid")
    seen_run_keys: set[str] = set()
    per_cell: dict[str, list[SchedulePosition]] = {k: [] for k in expected_cell_keys}
    for index, pos in enumerate(positions):
        if pos.position != index:
            raise ComparisonSpecError("schedule_position_sequence_invalid")
        if pos.batch not in _BATCHES or pos.batch != config_batch(pos.config_index):
            raise ComparisonSpecError("schedule_batch_invalid")
        if pos.run_key in seen_run_keys:
            raise ComparisonSpecError("schedule_duplicate_run")
        seen_run_keys.add(pos.run_key)
        if pos.cell_key not in per_cell:
            raise ComparisonSpecError("schedule_unknown_cell")
        per_cell[pos.cell_key].append(pos)
    for cell_key, cell_positions in per_cell.items():
        if len(cell_positions) != CONFIGS_PER_CELL:
            raise ComparisonSpecError("schedule_cell_config_count_invalid")
        indices = [p.config_index for p in cell_positions]
        # Per-cell launch order IS config-index order (config outermost), so
        # ties break by earliest actual evaluation order via config index.
        if indices != sorted(indices) or indices != list(range(CONFIGS_PER_CELL)):
            raise ComparisonSpecError("schedule_config_order_invalid")
        # Every first-batch (initial population) position of a cell precedes
        # every feedback-batch position of that cell.
        last_initial = max(p.position for p in cell_positions if p.batch == _BATCH_INITIAL)
        first_feedback = min(
            p.position for p in cell_positions if p.batch == _BATCH_FEEDBACK
        )
        if last_initial >= first_feedback:
            raise ComparisonSpecError("schedule_batch_order_invalid")
    # Rotation coverage: every (config, block) slot runs all three methods.
    blocks = _block_order()
    for config_index in range(CONFIGS_PER_CELL):
        for block_index, (task_id, seed) in enumerate(blocks):
            slot = [
                p
                for p in positions
                if p.config_index == config_index
                and p.cell_key.startswith(f"{task_id}:{seed}:")
                and p.position // len(blocks) // len(METHODS) == config_index
            ]
            methods = sorted(p.cell_key.rsplit(":", 1)[1] for p in slot)
            if len(slot) != len(METHODS) or methods != sorted(METHODS):
                raise ComparisonSpecError("schedule_rotation_invalid")
            expected = [
                f"{task_id}:{seed}:{m}" for m in _method_rotation(config_index, block_index)
            ]
            if [p.cell_key for p in slot] != expected:
                raise ComparisonSpecError("schedule_rotation_invalid")


def _roster_commit(cells: Sequence[CellSpec]) -> str:
    """The single implementation source commit bound by the roster."""
    commits = {cell.source_commit for cell in cells}
    if len(commits) != 1:
        raise ComparisonSpecError("source_commit_not_uniform")
    return commits.pop()


def schedule_payload(
    positions: Sequence[SchedulePosition], source_commit: str
) -> dict[str, Any]:
    """Canonical frozen payload incl. the outer seed derivations."""
    proto = canonical_protocol()
    return {
        "schema": SCHEDULE_SCHEMA,
        "protocol_identity": proto.identity(),
        "source_commit": source_commit,
        "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
        "native_seed_domain": NATIVE_SEED_DOMAIN,
        "random_remainder_seed_domain": RANDOM_REMAINDER_SEED_DOMAIN,
        "native_seeds": {
            f"{task.task_id}:{seed}": derive_native_seed(task.task_id, seed)
            for task in proto.tasks
            for seed in proto.search_seeds
        },
        "random_remainder_seeds": {
            f"{task.task_id}:{seed}": derive_random_remainder_seed(task.task_id, seed)
            for task in proto.tasks
            for seed in proto.search_seeds
        },
        "positions": [pos.to_dict() for pos in positions],
    }


def schedule_identity(
    positions: Sequence[SchedulePosition], source_commit: str
) -> str:
    return identity_of(schedule_payload(positions, source_commit))


def render_schedule(
    positions: Sequence[SchedulePosition], source_commit: str
) -> str:
    """Canonical serialization of the frozen schedule artifact."""
    payload = schedule_payload(positions, source_commit)
    payload["schedule_identity"] = identity_of(
        {k: v for k, v in payload.items() if k != "schedule_identity"}
    )
    return canonical_json(payload) + "\n"


def parse_schedule(text: str, cells: Sequence[CellSpec]) -> tuple[SchedulePosition, ...]:
    """Parse and fully re-validate a frozen schedule artifact.

    The parsed positions must reproduce ``build_schedule(cells)`` EXACTLY
    and the embedded identity must match; any drift fails closed.
    """
    import json

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ComparisonSpecError("schedule_malformed") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SCHEDULE_SCHEMA:
        raise ComparisonSpecError("schedule_schema_mismatch")
    commit = _roster_commit(cells)
    expected = build_schedule(cells)
    expected_payload = schedule_payload(expected, commit)
    expected_payload["schedule_identity"] = identity_of(expected_payload)
    if payload != expected_payload:
        raise ComparisonSpecError("schedule_mismatch")
    return expected


def freeze_schedule(path: Path, cells: Sequence[CellSpec]) -> tuple[SchedulePosition, ...]:
    """Write the frozen schedule once; refuse to replace different bytes.

    Symlink-safe: an aliased target (even with matching bytes) and a
    pre-existing temporary symlink are refused, never followed or
    overwritten.
    """
    positions = build_schedule(cells)
    rendered = render_schedule(positions, _roster_commit(cells))
    if path.is_symlink():
        raise ComparisonSpecError("schedule_symlink_refused")
    if path.exists():
        existing = path.read_bytes()
        if existing != rendered.encode("utf-8"):
            raise ComparisonSpecError("schedule_already_frozen_mismatch")
        return positions
    tmp = path.with_name(path.name + ".tmp")
    if tmp.is_symlink() or tmp.exists():
        raise ComparisonSpecError("schedule_tmp_refused")
    import os

    with open(tmp, "x", encoding="utf-8") as handle:
        handle.write(rendered)
    os.replace(tmp, path)
    return positions


def load_schedule(path: Path, cells: Sequence[CellSpec]) -> tuple[SchedulePosition, ...]:
    """Load the frozen schedule, fully re-validated against the roster."""
    if path.is_symlink() or not path.exists():
        raise ComparisonSpecError("schedule_missing")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ComparisonSpecError("schedule_malformed") from exc
    positions = parse_schedule(text, cells)
    if render_schedule(positions, _roster_commit(cells)).encode("utf-8") != text.encode(
        "utf-8"
    ):
        raise ComparisonSpecError("schedule_mismatch")
    return positions
