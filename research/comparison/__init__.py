"""EvoRSI iStratDE controlled pilot comparison (EXP-001..EXP-008).

Separate entry point from the US-009 acceptance stack: this package never
imports, reuses or mutates the US-009 runtime state, ledger or review gate.
The protocol and spec types are stdlib-only so they are importable from the
research venv, the Evo venv and the legacy JAX venv alike.
"""
from research.comparison.protocol import (
    COMPARISON_SCHEMA,
    COMPARISON_DIMS,
    CONFIGS_PER_CELL,
    FIXED_THETA,
    METHODS,
    NUM_CELLS,
    NUM_INNER_RUNS,
    SEARCH_SEEDS,
    SPLIT_SEED_BASE,
    TASKS,
    TASKS_BY_ID,
    CellBudget,
    ComparisonProtocol,
    ComparisonSpecError,
    GlobalBudget,
    InnerBudget,
    ModelProfile,
    TaskPin,
    canonical_protocol,
    decode_theta,
    identity_of,
    split_seed,
    theta_gate,
)
from research.comparison.specs import (
    CellSpec,
    InnerRunSpec,
    TaskBinding,
    derive_inner_seed,
    iter_cells,
    iter_inner_runs,
)

__all__ = [
    "COMPARISON_SCHEMA",
    "COMPARISON_DIMS",
    "CONFIGS_PER_CELL",
    "FIXED_THETA",
    "METHODS",
    "NUM_CELLS",
    "NUM_INNER_RUNS",
    "SEARCH_SEEDS",
    "SPLIT_SEED_BASE",
    "TASKS",
    "TASKS_BY_ID",
    "CellBudget",
    "CellSpec",
    "ComparisonProtocol",
    "ComparisonSpecError",
    "GlobalBudget",
    "InnerBudget",
    "InnerRunSpec",
    "ModelProfile",
    "TaskBinding",
    "TaskPin",
    "canonical_protocol",
    "decode_theta",
    "derive_inner_seed",
    "identity_of",
    "iter_cells",
    "iter_inner_runs",
    "split_seed",
    "theta_gate",
]
