"""Acceptance search vector: retained/dropped dims and decode/gate (US-007).

The OUTER iStratDE optimizer (see research.search.validation_config for the
INNER-vs-OUTER generation terminology) searches over config vectors. The
legacy experiment (research/adapters/safe_rsi_de_adapter.py — the frozen
record of the old baseline, not edited here) used an 8-dim vector. For the
acceptance run the supervisor decided to DISABLE prompt memory in the
validation config (it triggers auxiliary rich_memory_summary model calls), so
the three prompt-memory dims have no observable effect and are dropped.

Retained dims (legacy indices 0, 1, 2, 3, 7) each map to a branch/probability
with a documented real effect, proven Evo-side by deterministic solver tests:

- ``crossover_prob`` — the operator coin flip between the improve operator
  and the crossover operator in the solver loop.
- ``score_weight`` / ``delta_weight`` / ``novelty_weight`` — the components
  of the experience-based parent-selection utility.
- ``max_debug_depth`` — the number of debug-cycle attempts per candidate.

Dropped dims (legacy indices 4, 5, 6) are kept here only as documentation
(:data:`DROPPED_DIMS`) so evidence dumps can explain why they are absent from
the acceptance vector.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SearchDim:
    """One dimension of the OUTER DE search vector.

    ``effect`` is a one-line description of the real solver branch or
    probability this dim provably changes; it is empty for dropped dims
    (their rationale lives in :data:`DROP_RATIONALE`).
    """

    name: str
    lower: float
    upper: float
    kind: str  # "float" | "int"
    hydra_path: str
    effect: str


#: Why the prompt-memory dims are not part of the acceptance search vector.
DROP_RATIONALE = (
    "prompt_memory is disabled in the validation config to avoid auxiliary "
    "rich_memory_summary model calls; these dims only affect prompt-memory "
    "card selection (the solver reads them only on the prompt-memory path), "
    "so they have no observable branch effect and are dropped from the "
    "acceptance search vector"
)

_SOLVER = "search.runner.solver"
_WEIGHTS = f"{_SOLVER}.experience.parent_selection.weights"
_PROMPT_MEMORY = f"{_SOLVER}.experience.prompt_memory"

#: Dims of the acceptance search vector, in canonical (legacy-index) order:
#: legacy 0, 1, 2, 3, 7. Bounds and hydra paths match the legacy adapter.
RETAINED_DIMS: tuple[SearchDim, ...] = (
    SearchDim(
        name="crossover_prob",
        lower=0.05,
        upper=0.95,
        kind="float",
        hydra_path=f"{_SOLVER}.crossover_prob",
        effect="operator coin flip improve-vs-crossover",
    ),
    SearchDim(
        name="score_weight",
        lower=0.0,
        upper=2.0,
        kind="float",
        hydra_path=f"{_WEIGHTS}.score",
        effect="experience parent-selection utility component (score)",
    ),
    SearchDim(
        name="delta_weight",
        lower=0.0,
        upper=2.0,
        kind="float",
        hydra_path=f"{_WEIGHTS}.delta",
        effect="experience parent-selection utility component (delta)",
    ),
    SearchDim(
        name="novelty_weight",
        lower=0.0,
        upper=2.0,
        kind="float",
        hydra_path=f"{_WEIGHTS}.novelty",
        effect="experience parent-selection utility component (novelty)",
    ),
    SearchDim(
        name="max_debug_depth",
        lower=0.0,
        upper=6.0,
        kind="int",
        hydra_path=f"{_SOLVER}.max_debug_depth",
        effect="debug-cycle attempt count",
    ),
)

#: Legacy indices 4, 5, 6 — documented but absent from the acceptance vector.
DROPPED_DIMS: tuple[SearchDim, ...] = (
    SearchDim(
        name="max_related_cards",
        lower=1.0,
        upper=8.0,
        kind="int",
        hydra_path=f"{_PROMPT_MEMORY}.max_related_cards",
        effect="",
    ),
    SearchDim(
        name="ancestor_k",
        lower=0.0,
        upper=6.0,
        kind="int",
        hydra_path=f"{_PROMPT_MEMORY}.improve.ancestor_k",
        effect="",
    ),
    SearchDim(
        name="sibling_k",
        lower=0.0,
        upper=6.0,
        kind="int",
        hydra_path=f"{_PROMPT_MEMORY}.improve.sibling_k",
        effect="",
    ),
)

#: All dims known to this module (retained first, then dropped), for
#: evidence dumps and name lookups. The OUTER optimizer only ever sees
#: :data:`RETAINED_DIMS`.
DIMS: tuple[SearchDim, ...] = RETAINED_DIMS + DROPPED_DIMS

_DIMS_BY_NAME = {dim.name: dim for dim in DIMS}


def by_name(name: str) -> SearchDim:
    """Look up any known dim (retained or dropped) by name."""
    return _DIMS_BY_NAME[name]


def decode_retained(vector: Sequence[float]) -> dict[str, float | int]:
    """Decode one OUTER-vector individual into retained-dim values.

    Values are clipped to the dim bounds and integer dims are rounded
    (round-half-even, matching the legacy ``jnp.rint`` decode). The vector
    length must equal ``len(RETAINED_DIMS)`` — the acceptance vector contains
    ONLY retained dims; anything else raises ``ValueError(
    "vector_length_mismatch")`` so a stale 8-dim legacy vector fails loudly
    instead of being silently truncated.
    """
    if len(vector) != len(RETAINED_DIMS):
        raise ValueError("vector_length_mismatch")
    decoded: dict[str, float | int] = {}
    for dim, raw in zip(RETAINED_DIMS, vector):
        if isinstance(raw, bool) or not math.isfinite(float(raw)):
            raise ValueError("nonfinite_or_invalid_search_vector")
        value = min(max(float(raw), dim.lower), dim.upper)
        if dim.kind == "int":
            # round() is round-half-even on floats, matching the legacy
            # jnp.rint decode.
            decoded[dim.name] = int(round(value))
        else:
            decoded[dim.name] = value
    return decoded


def safety_gate(decoded: Mapping[str, Any]) -> None:
    """Fixed safety gate over decoded retained dims; the optimizer cannot
    change this policy.

    Mirrors the legacy gate restricted to retained dims: the dropped-dim
    check (``max_related_cards > 8``) is gone together with the dim, and the
    debug-depth message is renamed to the machine-readable
    ``max_debug_depth_out_of_range``.
    """
    for dim in RETAINED_DIMS:
        value = decoded[dim.name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("nonfinite_or_invalid_search_vector")
        if dim.kind == "int" and (not isinstance(value, int) or not dim.lower <= value <= dim.upper):
            raise ValueError("max_debug_depth_out_of_range")
        if dim.kind == "float" and not dim.lower <= value <= dim.upper:
            raise ValueError("search_dimension_out_of_range")
    if int(decoded["max_debug_depth"]) > 6:
        raise ValueError("max_debug_depth_out_of_range")
    weights = (
        float(decoded["score_weight"])
        + float(decoded["delta_weight"])
        + float(decoded["novelty_weight"])
    )
    if weights <= 0:
        raise ValueError("empty_parent_selection_weights")


def hydra_overrides(decoded: Mapping[str, Any]) -> list[str]:
    """Render decoded retained dims as OpenMLE-Evo Hydra overrides.

    The safety gate runs first (a rejected config raises ValueError naming
    the violated rule). Overrides are emitted in canonical RETAINED_DIMS
    order; float dims use the legacy 6-decimal format.
    """
    safety_gate(decoded)
    overrides: list[str] = []
    for dim in RETAINED_DIMS:
        value = decoded[dim.name]
        if dim.kind == "int":
            overrides.append(f"{dim.hydra_path}={int(value)}")
        else:
            overrides.append(f"{dim.hydra_path}={float(value):.6f}")
    return overrides


def mapping_explanation() -> dict[str, Any]:
    """Structured explanation of the vector mapping for evidence dumps."""
    return {
        "retained": {
            dim.name: {
                "bounds": [dim.lower, dim.upper],
                "kind": dim.kind,
                "hydra_path": dim.hydra_path,
                "effect": dim.effect,
            }
            for dim in RETAINED_DIMS
        },
        "dropped": {
            dim.name: {
                "bounds": [dim.lower, dim.upper],
                "kind": dim.kind,
                "hydra_path": dim.hydra_path,
                "rationale": DROP_RATIONALE,
            }
            for dim in DROPPED_DIMS
        },
        "inner_vs_outer": (
            "INNER generations are Evo solver generations "
            "(search.runner.solver.num_generations) inside one config "
            "evaluation; OUTER generations are iStratDE DE generations over "
            "config vectors, one fitness evaluation per inner run "
            "(fitness minimized natively, fitness=-accuracy)"
        ),
    }
