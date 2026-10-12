"""Contract tests for the three method theta plans (EXP-006).

Uses a fake NativeProvider (deterministic scripted batches); the REAL
native state proofs live in test_comparison_native.py against the pinned
legacy JAX interpreter.
"""
from __future__ import annotations

import pytest

from research.comparison.methods import (
    feedback_theta_plan,
    full_theta_plan,
    initial_theta_plan,
    random_remainder_vectors,
    validate_encoded_vector,
    validate_utilities,
)
from research.comparison.protocol import (
    COMPARISON_DIMS,
    FIXED_THETA,
    ComparisonSpecError,
    decode_theta,
)
from research.comparison.schedule import INITIAL_BATCH_CONFIGS, derive_native_seed

from research.tests.test_comparison_specs import make_bindings
from research.comparison.specs import iter_cells

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _cells():
    return iter_cells(make_bindings(), COMMIT)


def _cell(task_id: str, seed: int, method: str):
    return next(
        c
        for c in _cells()
        if c.task_id == task_id and c.search_seed == seed and c.method == method
    )


def _fake_batch(offset: float) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            min(dim.upper, max(dim.lower, 0.5 + offset + 0.01 * i + 0.001 * j))
            for j, dim in enumerate(COMPARISON_DIMS)
        )
        for i in range(INITIAL_BATCH_CONFIGS)
    )


class FakeNative:
    """Scripted native provider recording every call."""

    def __init__(self) -> None:
        self.first_calls: list[int] = []
        self.second_calls: list[tuple[int, tuple[float, ...]]] = []

    def setup_population(self, seed: int):
        return _fake_batch(-0.2)

    def first_batch(self, seed: int):
        self.first_calls.append(seed)
        return _fake_batch(0.0)

    def second_batch(self, seed: int, fitness):
        self.second_calls.append((seed, tuple(fitness)))
        return _fake_batch(0.0), _fake_batch(0.3)


def test_fixed_repeats_default_theta_eight_times():
    provider = FakeNative()
    cell = _cell("banknote", 11, "FIXED")
    plan = full_theta_plan(cell, provider)
    assert sorted(plan) == list(range(8))
    for vector in plan.values():
        assert vector == FIXED_THETA
    # FIXED never touches the native provider.
    assert provider.first_calls == [] and provider.second_calls == []


def test_random_first_four_match_paired_native_initial():
    provider = FakeNative()
    random_cell = _cell("wdbc", 22, "RANDOM")
    istratde_cell = _cell("wdbc", 22, "ISTRATDE")
    random_plan = initial_theta_plan(random_cell, provider)
    istratde_plan = initial_theta_plan(istratde_cell, provider)
    assert provider.first_calls == [
        derive_native_seed("wdbc", 22),
        derive_native_seed("wdbc", 22),
    ]
    for index in range(INITIAL_BATCH_CONFIGS):
        assert random_plan[index] == istratde_plan[index]
        assert random_plan[index] == _fake_batch(0.0)[index]


def test_random_remainder_independent_uniform_and_score_free():
    provider = FakeNative()
    cell = _cell("spambase", 33, "RANDOM")
    plan = full_theta_plan(cell, provider, utilities={0: 0.9, 1: 0.1, 2: 0.5, 3: 0.0})
    # The remainder comes from the frozen stdlib RNG, not the provider and
    # not the scores: second_batch was never called.
    assert provider.second_calls == []
    expected = random_remainder_vectors("spambase", 33)
    for index in range(INITIAL_BATCH_CONFIGS, 8):
        assert plan[index] == expected[index - INITIAL_BATCH_CONFIGS]
        for value, dim in zip(plan[index], COMPARISON_DIMS):
            assert dim.lower <= value <= dim.upper
    # Different (task, seed) pairs give different remainders.
    assert random_remainder_vectors("spambase", 33) != random_remainder_vectors(
        "spambase", 22
    )
    assert random_remainder_vectors("spambase", 33) != random_remainder_vectors(
        "banknote", 33
    )
    # Deterministic.
    assert random_remainder_vectors("spambase", 33) == random_remainder_vectors(
        "spambase", 33
    )


def test_istratde_feedback_uses_negated_utilities_and_echo_check():
    provider = FakeNative()
    cell = _cell("banknote", 11, "ISTRATDE")
    initial = initial_theta_plan(cell, provider)
    utilities = [0.7, 0.0, 0.45, 1.0]
    plan = feedback_theta_plan(
        cell, provider, utilities, [list(v) for v in initial.values()]
    )
    assert provider.second_calls == [
        (derive_native_seed("banknote", 11), (-0.7, -0.0, -0.45, -1.0))
    ]
    assert sorted(plan) == [4, 5, 6, 7]
    for vector in plan.values():
        assert vector == _fake_batch(0.3)[0] or vector in _fake_batch(0.3)


def test_istratde_feedback_rejects_echo_mismatch():
    class BadEcho(FakeNative):
        def second_batch(self, seed, fitness):
            return _fake_batch(0.9), _fake_batch(0.3)

    cell = _cell("banknote", 11, "ISTRATDE")
    initial = initial_theta_plan(cell, FakeNative())
    with pytest.raises(ComparisonSpecError, match="native_replay_mismatch"):
        feedback_theta_plan(
            cell, BadEcho(), [0.5] * 4, [list(v) for v in initial.values()]
        )


def test_istratde_requires_feedback_inputs():
    cell = _cell("banknote", 11, "ISTRATDE")
    with pytest.raises(ComparisonSpecError, match="istratde_feedback_required"):
        full_theta_plan(cell, FakeNative())


def test_validate_utilities_strict():
    assert validate_utilities([0.0, 0.5, 1.0, 0]) == (0.0, 0.5, 1.0, 0.0)
    for bad in (
        [0.5] * 3,  # wrong length
        [0.5, 0.5, 0.5, 1.5],  # > 1
        [0.5, 0.5, 0.5, -0.1],  # < 0
        [0.5, 0.5, 0.5, float("nan")],
        [0.5, 0.5, 0.5, float("inf")],
        [0.5, 0.5, 0.5, True],
        [0.5, 0.5, 0.5, "0.5"],
    ):
        with pytest.raises(ComparisonSpecError):
            validate_utilities(bad)


def test_validate_encoded_vector_strict():
    assert validate_encoded_vector(FIXED_THETA) == FIXED_THETA
    for bad in (
        (0.5, 1.0, 0.4, 0.25),  # length
        (0.5, 1.0, 0.4, 0.25, 3.0),  # debug depth above bound
        (0.5, 1.0, 0.4, 0.25, -1.0),
        (0.0, 1.0, 0.4, 0.25, 1.0),  # crossover below bound
        (0.5, 1.0, 0.4, 0.25, float("nan")),
        (0.5, 1.0, 0.4, 0.25, True),
    ):
        with pytest.raises(ComparisonSpecError):
            validate_encoded_vector(bad)


def test_bounded_zero_weight_vector_retained_gate_applies_at_launch():
    # A bounded all-zero-weight vector is a legitimate native output and
    # must be RETAINED verbatim (no resampling, pairing preserved); the
    # weight-sum gate rejects it only at launch time (orchestrate.py).
    retained = validate_encoded_vector((0.5, 0.0, 0.0, 0.0, 1.0))
    assert retained == (0.5, 0.0, 0.0, 0.0, 1.0)
    from research.comparison.protocol import theta_gate

    with pytest.raises(ComparisonSpecError, match="empty_parent_selection_weights"):
        theta_gate(decode_theta(retained))


def test_decoded_plans_pass_gate_and_debug_depth_rounds():
    provider = FakeNative()
    for method in ("FIXED", "RANDOM", "ISTRATDE"):
        cell = _cell("banknote", 11, method)
        if method == "ISTRATDE":
            initial = initial_theta_plan(cell, provider)
            plan = full_theta_plan(
                cell,
                provider,
                utilities={i: 0.5 for i in range(4)},
                accepted_vectors={i: list(v) for i, v in initial.items()},
            )
        else:
            plan = full_theta_plan(cell, provider)
        for vector in plan.values():
            decoded = decode_theta(vector)
            assert isinstance(decoded["max_debug_depth"], int)
            assert 0 <= decoded["max_debug_depth"] <= 2
