"""Offline tests for research/comparison/analysis.py (EXP-008, FR7).

Pure statistics: within-task/seed pairing, equal-weight task
macroaverage, the frozen 10000-replicate paired-seed bootstrap with RNG
20261010, partial-observation honesty and the conclusion rule. No I/O.
"""
from __future__ import annotations

import pytest

from research.comparison.analysis import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CellOutcome,
    analyze_contrast,
    bootstrap_macroaverage_interval,
    conclude,
    paired_differences,
)

METHODS = ("FIXED", "RANDOM", "ISTRATDE")
TASKS = ("banknote", "wdbc", "spambase")
SEEDS = (11, 22, 33)


def make_outcomes(
    utilities: dict[tuple[str, int, str], float | None],
) -> dict[str, CellOutcome]:
    """utility float = observed success; None-with-observed = failure."""
    outcomes: dict[str, CellOutcome] = {}
    for task in TASKS:
        for seed in SEEDS:
            for method in METHODS:
                key = f"{task}:{seed}:{method}"
                value = utilities.get((task, seed, method), "unobserved")
                if value == "unobserved":
                    outcomes[key] = CellOutcome(
                        key, task, method, seed, False, None, None
                    )
                elif value is None:
                    outcomes[key] = CellOutcome(
                        key, task, method, seed, True, None, 0.0
                    )
                else:
                    outcomes[key] = CellOutcome(
                        key, task, method, seed, True, float(value), float(value)
                    )
    return outcomes


def full_matrix(istratde_bonus: float = 0.0) -> dict[tuple[str, int, str], float]:
    utilities: dict[tuple[str, int, str], float] = {}
    for task in TASKS:
        for seed in SEEDS:
            utilities[(task, seed, "FIXED")] = 0.7
            utilities[(task, seed, "RANDOM")] = 0.8
            utilities[(task, seed, "ISTRATDE")] = 0.8 + istratde_bonus
    return utilities


class TestPairedDifferences:
    def test_pairing_preserved_within_task_and_seed(self):
        outcomes = make_outcomes(full_matrix(0.05))
        pairing = paired_differences(
            outcomes, treatment="ISTRATDE", baseline="RANDOM"
        )
        assert pairing["observed_pairs"] == 9
        assert pairing["unobserved_pairs"] == []
        for task in TASKS:
            assert pairing["diffs_by_task"][task] == {
                seed: pytest.approx(0.05) for seed in SEEDS
            }

    def test_observed_failure_is_zero_utility_not_missing(self):
        utilities = full_matrix()
        utilities[("banknote", 11, "ISTRATDE")] = None  # genuine failure
        outcomes = make_outcomes(utilities)
        pairing = paired_differences(
            outcomes, treatment="ISTRATDE", baseline="RANDOM"
        )
        assert pairing["observed_pairs"] == 9
        assert pairing["diffs_by_task"]["banknote"][11] == pytest.approx(-0.8)

    def test_unobserved_pair_is_explicit_never_zero_filled(self):
        utilities = full_matrix()
        del utilities[("wdbc", 22, "RANDOM")]  # baseline never observed
        outcomes = make_outcomes(utilities)
        pairing = paired_differences(
            outcomes, treatment="ISTRATDE", baseline="RANDOM"
        )
        assert pairing["observed_pairs"] == 8
        assert pairing["unobserved_pairs"] == ["wdbc:22"]
        assert 22 not in pairing["diffs_by_task"]["wdbc"]


class TestBootstrap:
    def test_deterministic_with_frozen_seed(self):
        diffs = {"banknote": [0.1, -0.2, 0.05], "wdbc": [0.0, 0.3, -0.1],
                 "spambase": [0.02, 0.02, -0.04]}
        first = bootstrap_macroaverage_interval(diffs)
        second = bootstrap_macroaverage_interval(diffs)
        assert first == second
        assert first["replicates"] == BOOTSTRAP_REPLICATES
        assert first["seed"] == BOOTSTRAP_SEED
        assert first["level"] == 0.95
        assert first["low"] <= first["high"]

    def test_constant_positive_difference_has_degenerate_interval(self):
        diffs = {task: [0.1, 0.1, 0.1] for task in TASKS}
        interval = bootstrap_macroaverage_interval(diffs)
        assert interval["low"] == pytest.approx(0.1)
        assert interval["high"] == pytest.approx(0.1)

    def test_interval_covers_sign(self):
        diffs = {task: [-0.5, 0.5, 0.0] for task in TASKS}
        interval = bootstrap_macroaverage_interval(diffs)
        assert interval["low"] < 0.0 < interval["high"]

    def test_empty_observation_refused(self):
        with pytest.raises(ValueError, match="bootstrap_requires_observed_pairs"):
            bootstrap_macroaverage_interval({})

    def test_tasks_resampled_independently(self):
        # A single-task spike must not drag all tasks in lockstep: the
        # interval must be wider than the per-task constant case.
        diffs = {"banknote": [0.9, -0.9, 0.0], "wdbc": [0.0, 0.0, 0.0],
                 "spambase": [0.0, 0.0, 0.0]}
        interval = bootstrap_macroaverage_interval(diffs)
        assert interval["high"] - interval["low"] > 0.05


class TestContrast:
    def test_primary_contrast_complete(self):
        outcomes = make_outcomes(full_matrix(0.05))
        result = analyze_contrast(
            outcomes, treatment="ISTRATDE", baseline="RANDOM", expected_pairs=9
        )
        assert result["partial"] is False
        assert result["estimate"] == pytest.approx(0.05)
        assert result["interval"]["low"] == pytest.approx(0.05)
        assert result["paired_differences"]["banknote"]["11"] == pytest.approx(0.05)

    def test_partial_contrast_marked(self):
        utilities = full_matrix(0.05)
        del utilities[("spambase", 33, "ISTRATDE")]
        outcomes = make_outcomes(utilities)
        result = analyze_contrast(
            outcomes, treatment="ISTRATDE", baseline="RANDOM", expected_pairs=9
        )
        assert result["partial"] is True
        assert result["observed_pairs"] == 8
        assert result["unobserved_pairs"] == ["spambase:33"]
        assert result["estimate"] is not None  # partial estimate allowed

    def test_no_observed_pairs(self):
        outcomes = make_outcomes({})
        result = analyze_contrast(
            outcomes, treatment="ISTRATDE", baseline="RANDOM", expected_pairs=9
        )
        assert result["estimate"] is None
        assert result["interval"] is None
        assert result["partial"] is True


class TestConclusion:
    def test_positive_requires_complete_matrix_and_interval_above_zero(self):
        outcomes = make_outcomes(full_matrix(0.05))
        primary = analyze_contrast(
            outcomes, treatment="ISTRATDE", baseline="RANDOM", expected_pairs=9
        )
        assert conclude(primary, matrix_complete=True)["conclusion"] == "positive"

    def test_no_gain_when_interval_spans_zero(self):
        utilities = full_matrix()
        utilities[("banknote", 11, "ISTRATDE")] = 0.4  # one big loss
        outcomes = make_outcomes(utilities)
        primary = analyze_contrast(
            outcomes, treatment="ISTRATDE", baseline="RANDOM", expected_pairs=9
        )
        assert conclude(primary, matrix_complete=True)["conclusion"] == "no-gain"

    def test_incomplete_matrix_is_insufficient_even_when_positive(self):
        outcomes = make_outcomes(full_matrix(0.05))
        primary = analyze_contrast(
            outcomes, treatment="ISTRATDE", baseline="RANDOM", expected_pairs=9
        )
        result = conclude(primary, matrix_complete=False)
        assert result["conclusion"] == "insufficient-evidence"
        assert "partial" in result["reason"]
