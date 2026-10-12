"""Theta plans for the three comparison methods (EXP-006).

- FIXED: the approved default theta, repeated for all eight configs; each
  config still runs as a genuinely fresh inner search (fresh run dir, no
  journal/memory/cache reuse — enforced by the runner). Scores are never
  consulted.
- RANDOM: configs 0..3 are EXACTLY the paired native iStratDE first ask
  (same (task, search seed) native seed, identical float32->float
  conversion via the shared provider output); configs 4..7 are independent
  uniform draws in the encoded bounds from a frozen stdlib RNG seed.
  Scores are never consulted.
- ISTRATDE: configs 0..3 are the same paired native first ask; configs
  4..7 are the native SECOND ask after telling the accepted first-batch
  utilities (``-utility`` fitness, failed config utility 0). This is the
  only method whose vectors depend on evaluation feedback.

All vectors are validated fail-closed: exact length, strict float typing,
finiteness, encoded-bounds membership, and the decoded weight-sum gate.
"""
from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

from research.comparison.native import NativeProvider
from research.comparison.protocol import (
    COMPARISON_DIMS,
    CONFIGS_PER_CELL,
    FIXED_THETA,
    ComparisonSpecError,
    decode_theta,
    identity_of,
)
from research.comparison.schedule import (
    INITIAL_BATCH_CONFIGS,
    derive_native_seed,
    derive_random_remainder_seed,
)
from research.comparison.specs import CellSpec

EncodedVector = tuple[float, ...]


def validate_encoded_vector(vector: Sequence[object]) -> EncodedVector:
    """Strict encoded-vector validation (length, types, finiteness, bounds).

    The weight-sum>0 gate is deliberately NOT applied here: a bounded
    native vector with all-zero weights is a legitimate optimizer output
    that must be RETAINED verbatim (RANDOM/ISTRATDE pairing forbids
    resampling); the gate rejects it at launch time instead, recording a
    truthful gate-rejected config with utility 0 (orchestrate.py).
    """
    if len(vector) != len(COMPARISON_DIMS):
        raise ComparisonSpecError("theta_length_mismatch")
    out: list[float] = []
    for dim, raw in zip(COMPARISON_DIMS, vector):
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            raise ComparisonSpecError("nonfinite_or_invalid_theta")
        value = float(raw)
        if not dim.lower <= value <= dim.upper:
            raise ComparisonSpecError("theta_dimension_out_of_range")
        out.append(value)
    decode_theta(tuple(out))  # encoding/rounding sanity (never gates)
    return tuple(out)


def encoded_vector_identity(vector: Sequence[float]) -> str:
    """Canonical identity of one accepted encoded vector."""
    return identity_of({"theta": [float(v) for v in vector]})


def validate_utilities(utilities: Sequence[object]) -> tuple[float, ...]:
    """Accepted first-batch utilities: finite accuracies in [0, 1]."""
    if len(utilities) != INITIAL_BATCH_CONFIGS:
        raise ComparisonSpecError("utility_batch_length_invalid")
    out: list[float] = []
    for raw in utilities:
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or not 0.0 <= float(raw) <= 1.0
        ):
            raise ComparisonSpecError("utility_out_of_range")
        out.append(float(raw))
    return tuple(out)


def random_remainder_vectors(task_id: str, search_seed: int) -> tuple[EncodedVector, ...]:
    """RANDOM configs 4..7: independent uniform draws in the encoded bounds.

    One frozen stdlib RNG per (task, search seed); never a function of any
    score, search result or native vector.
    """
    rng = random.Random(derive_random_remainder_seed(task_id, search_seed))
    return tuple(
        validate_encoded_vector(
            tuple(rng.uniform(dim.lower, dim.upper) for dim in COMPARISON_DIMS)
        )
        for _ in range(CONFIGS_PER_CELL - INITIAL_BATCH_CONFIGS)
    )


def paired_initial_vectors(
    cell: CellSpec, provider: NativeProvider
) -> tuple[EncodedVector, ...]:
    """The paired native first ask for a (task, search seed) position.

    Identical for RANDOM and ISTRATDE at the same position BY CONSTRUCTION
    (method never enters the native seed), including float32->float
    conversion, because both consume the same provider output numbers.
    """
    batch = provider.first_batch(derive_native_seed(cell.task_id, cell.search_seed))
    return tuple(validate_encoded_vector(vector) for vector in batch)


def initial_theta_plan(
    cell: CellSpec, provider: NativeProvider
) -> dict[int, EncodedVector]:
    """Configs 0..3 for the cell's method (frozen before any generation)."""
    if cell.method == "FIXED":
        return {
            index: validate_encoded_vector(FIXED_THETA)
            for index in range(INITIAL_BATCH_CONFIGS)
        }
    paired = paired_initial_vectors(cell, provider)
    return {index: paired[index] for index in range(INITIAL_BATCH_CONFIGS)}


def feedback_theta_plan(
    cell: CellSpec,
    provider: NativeProvider,
    utilities: Sequence[object],
    expected_first: Sequence[Sequence[float]],
) -> dict[int, EncodedVector]:
    """Configs 4..7 for the cell's method.

    ``utilities`` are the accepted first-batch utilities in config order
    0..3 (validation accuracy, 0.0 for a failed config). ``expected_first``
    is the persisted accepted first batch; for ISTRATDE the provider's
    echoed first batch must match it bit-for-bit (frozen seed + verified
    fitness history replay), otherwise the plan fails closed.
    """
    if cell.method == "FIXED":
        return {
            index: validate_encoded_vector(FIXED_THETA)
            for index in range(INITIAL_BATCH_CONFIGS, CONFIGS_PER_CELL)
        }
    if cell.method == "RANDOM":
        # Independent uniform remainder; utilities are deliberately ignored.
        remainder = random_remainder_vectors(cell.task_id, cell.search_seed)
        return {
            index: remainder[index - INITIAL_BATCH_CONFIGS]
            for index in range(INITIAL_BATCH_CONFIGS, CONFIGS_PER_CELL)
        }
    accepted_first = tuple(
        validate_encoded_vector(vector) for vector in expected_first
    )
    if len(accepted_first) != INITIAL_BATCH_CONFIGS:
        raise ComparisonSpecError("initial_batch_length_invalid")
    # ISTRATDE: second native ask from state preserved through tell.
    accepted_utilities = validate_utilities(utilities)
    fitness = [-utility for utility in accepted_utilities]
    echoed_first, second = provider.second_batch(
        derive_native_seed(cell.task_id, cell.search_seed), fitness
    )
    validated_second = tuple(validate_encoded_vector(vector) for vector in second)
    if tuple(tuple(float(v) for v in row) for row in echoed_first) != tuple(
        tuple(float(v) for v in row) for row in accepted_first
    ):
        raise ComparisonSpecError("native_replay_mismatch")
    return {
        index: validated_second[index - INITIAL_BATCH_CONFIGS]
        for index in range(INITIAL_BATCH_CONFIGS, CONFIGS_PER_CELL)
    }


def full_theta_plan(
    cell: CellSpec,
    provider: NativeProvider,
    utilities: Mapping[int, object] | None = None,
    accepted_vectors: Mapping[int, Sequence[float]] | None = None,
) -> dict[int, EncodedVector]:
    """All eight config vectors for one cell.

    For ISTRATDE the feedback half requires the accepted first-batch
    utilities and vectors; for FIXED/RANDOM those arguments are ignored
    (neither method ever consumes scores to choose configs).
    """
    plan = initial_theta_plan(cell, provider)
    if cell.method == "ISTRATDE":
        if utilities is None or accepted_vectors is None:
            raise ComparisonSpecError("istratde_feedback_required")
        plan.update(
            feedback_theta_plan(
                cell,
                provider,
                [utilities[index] for index in range(INITIAL_BATCH_CONFIGS)],
                [accepted_vectors[index] for index in range(INITIAL_BATCH_CONFIGS)],
            )
        )
    else:
        plan.update(feedback_theta_plan(cell, provider, (), ()))
    if sorted(plan) != list(range(CONFIGS_PER_CELL)):
        raise ComparisonSpecError("theta_plan_incomplete")
    return plan
