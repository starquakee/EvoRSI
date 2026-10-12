"""Typed cell and inner-run specs for the frozen comparison (EXP-001).

A CellSpec binds one (task, method, search seed) cell to:

- the hash-bound trusted task data identity (TaskBinding; the split hashes
  are registered by EXP-002 and enter every cell identity),
- the pinned model profile and per-cell budget,
- the implementation source commit under which the cell may run,
- the frozen protocol identity.

An InnerRunSpec adds the config index (0..7) and the derived inner seed.
The inner seed is a deterministic function of (task, search seed, config
index) ONLY — method never enters the derivation, so the paired config
position uses the same inner seed in all three methods (PRD requirement).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from research.comparison.protocol import (
    CONFIGS_PER_CELL,
    METHODS,
    SEARCH_SEEDS,
    SPLIT_SEED_BASE,
    TASKS_BY_ID,
    CellBudget,
    ComparisonProtocol,
    ComparisonSpecError,
    ModelProfile,
    canonical_protocol,
    identity_of,
)

_SHA256_HEX_LEN = 64
_COMMIT_HEX_LEN = 40
_HEX_DIGITS = frozenset("0123456789abcdef")

#: Domain separator for the inner-seed derivation; changing it changes all
#: seeds (which is why it is versioned).
INNER_SEED_DOMAIN = "istratde-comparison-inner-seed.v1"


def _require_hex(value: Any, length: int, rule: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or not set(value) <= _HEX_DIGITS
    ):
        raise ComparisonSpecError(rule)
    return value


def derive_inner_seed(task_id: str, search_seed: int, config_index: int) -> int:
    """Deterministic 32-bit inner seed for one config position.

    Method-independent BY DESIGN: FIXED, RANDOM and ISTRATDE use the same
    inner seed at the same (task, search seed, config index) position.
    """
    digest = hashlib.sha256(
        f"{INNER_SEED_DOMAIN}|{task_id}|{search_seed}|{config_index}".encode(
            "utf-8"
        )
    ).hexdigest()
    return int(digest[:8], 16)


@dataclass(frozen=True)
class TaskBinding:
    """Hash-bound trusted data identity for one task.

    Filled with real sha256 values by EXP-002 after the official UCI data
    is converted and split; the constructor already enforces the pinned
    task, split seed (20261010 + UCI id), shape and hash formats, so a
    malformed or mistargeted binding can never enter a cell spec.
    """

    task_id: str
    split_seed: int
    rows: int
    features: int
    train_csv_sha256: str
    validation_features_sha256: str
    validation_labels_sha256: str
    final_features_sha256: str
    final_labels_sha256: str

    def __post_init__(self) -> None:
        pin = TASKS_BY_ID.get(self.task_id) if isinstance(self.task_id, str) else None
        if pin is None:
            raise ComparisonSpecError("unknown_task")
        if (
            isinstance(self.split_seed, bool)
            or not isinstance(self.split_seed, int)
            or self.split_seed != SPLIT_SEED_BASE + pin.uci_id
        ):
            raise ComparisonSpecError("split_seed_mismatch")
        if (
            isinstance(self.rows, bool)
            or not isinstance(self.rows, int)
            or isinstance(self.features, bool)
            or not isinstance(self.features, int)
            or self.rows != pin.rows
            or self.features != pin.features
        ):
            raise ComparisonSpecError("task_shape_mismatch")
        for field_name in (
            "train_csv_sha256",
            "validation_features_sha256",
            "validation_labels_sha256",
            "final_features_sha256",
            "final_labels_sha256",
        ):
            _require_hex(
                getattr(self, field_name), _SHA256_HEX_LEN, "task_hash_invalid"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split_seed": self.split_seed,
            "rows": self.rows,
            "features": self.features,
            "train_csv_sha256": self.train_csv_sha256,
            "validation_features_sha256": self.validation_features_sha256,
            "validation_labels_sha256": self.validation_labels_sha256,
            "final_features_sha256": self.final_features_sha256,
            "final_labels_sha256": self.final_labels_sha256,
        }

    def identity(self) -> str:
        """Content identity; any split/data hash tampering changes it."""
        return identity_of(self.to_dict())


@dataclass(frozen=True)
class CellSpec:
    """One comparison cell: (task, method, search seed) fully bound."""

    task_binding: TaskBinding
    method: str
    search_seed: int
    model_profile: ModelProfile
    source_commit: str
    cell_budget: CellBudget
    protocol_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_binding, TaskBinding):
            raise ComparisonSpecError("task_binding_invalid")
        if self.method not in METHODS:
            raise ComparisonSpecError("unknown_method")
        if (
            isinstance(self.search_seed, bool)
            or not isinstance(self.search_seed, int)
            or self.search_seed not in SEARCH_SEEDS
        ):
            raise ComparisonSpecError("unknown_search_seed")
        # A well-formed but non-pinned profile or budget never enters a spec.
        if self.model_profile != ModelProfile():
            raise ComparisonSpecError("model_profile_not_pinned")
        if self.cell_budget != CellBudget():
            raise ComparisonSpecError("cell_budget_not_pinned")
        _require_hex(self.source_commit, _COMMIT_HEX_LEN, "source_commit_invalid")
        if self.protocol_identity != canonical_protocol().identity():
            raise ComparisonSpecError("protocol_identity_mismatch")

    @property
    def task_id(self) -> str:
        return self.task_binding.task_id

    def cell_key(self) -> str:
        """Canonical human-readable cell key ``task:seed:method``."""
        return f"{self.task_id}:{self.search_seed}:{self.method}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_identity": self.protocol_identity,
            "cell_key": self.cell_key(),
            "task_binding_identity": self.task_binding.identity(),
            "method": self.method,
            "search_seed": self.search_seed,
            "model_profile_identity": self.model_profile.identity(),
            "source_commit": self.source_commit,
            "cell_budget": self.cell_budget.to_dict(),
        }

    def identity(self) -> str:
        return identity_of(self.to_dict())

    def inner_run(self, config_index: int) -> "InnerRunSpec":
        """The spec for one config position of this cell (seed derived)."""
        return InnerRunSpec(
            cell=self,
            config_index=config_index,
            inner_seed=derive_inner_seed(
                self.task_id, self.search_seed, config_index
            ),
        )


@dataclass(frozen=True)
class InnerRunSpec:
    """One inner search: a cell plus a config index and its derived seed."""

    cell: CellSpec
    config_index: int
    inner_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.cell, CellSpec):
            raise ComparisonSpecError("cell_spec_invalid")
        if (
            isinstance(self.config_index, bool)
            or not isinstance(self.config_index, int)
            or not 0 <= self.config_index < CONFIGS_PER_CELL
        ):
            raise ComparisonSpecError("config_index_out_of_range")
        expected = derive_inner_seed(
            self.cell.task_id, self.cell.search_seed, self.config_index
        )
        if (
            isinstance(self.inner_seed, bool)
            or not isinstance(self.inner_seed, int)
            or self.inner_seed != expected
        ):
            # The seed is derived, never chosen: a mismatching seed is a
            # tampered spec, not an alternative configuration.
            raise ComparisonSpecError("inner_seed_mismatch")

    def run_key(self) -> str:
        """Canonical human-readable run key ``task:seed:method:config``."""
        return f"{self.cell.cell_key()}:{self.config_index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_identity": self.cell.identity(),
            "run_key": self.run_key(),
            "config_index": self.config_index,
            "inner_seed": self.inner_seed,
        }

    def identity(self) -> str:
        return identity_of(self.to_dict())


def iter_cells(
    bindings: Mapping[str, TaskBinding],
    source_commit: str,
    *,
    protocol: ComparisonProtocol | None = None,
) -> tuple[CellSpec, ...]:
    """All 27 cell specs in canonical order: task -> search seed -> method.

    ``bindings`` must carry exactly one TaskBinding per pinned task, keyed
    by task id. The enumeration order is the canonical identity order only;
    the interleaved execution schedule with rotated method order is a
    separate frozen artifact (EXP-006).
    """
    proto = canonical_protocol() if protocol is None else protocol
    expected_tasks = {task.task_id for task in proto.tasks}
    if set(bindings.keys()) != expected_tasks:
        raise ComparisonSpecError("task_bindings_incomplete")
    for key, binding in bindings.items():
        if not isinstance(binding, TaskBinding) or binding.task_id != key:
            raise ComparisonSpecError("task_binding_key_mismatch")
    profile = ModelProfile()
    budget = CellBudget()
    proto_identity = proto.identity()
    return tuple(
        CellSpec(
            task_binding=bindings[task.task_id],
            method=method,
            search_seed=seed,
            model_profile=profile,
            source_commit=source_commit,
            cell_budget=budget,
            protocol_identity=proto_identity,
        )
        for task in proto.tasks
        for seed in proto.search_seeds
        for method in proto.methods
    )


def iter_inner_runs(cells: Sequence[CellSpec]) -> tuple[InnerRunSpec, ...]:
    """All inner-run specs of the given cells (8 per cell, 216 for 27)."""
    return tuple(
        cell.inner_run(config_index)
        for cell in cells
        for config_index in range(CONFIGS_PER_CELL)
    )
