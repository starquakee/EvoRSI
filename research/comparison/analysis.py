"""Predeclared offline analysis for the iStratDE comparison (EXP-008, FR7).

Pure statistics over observed cell outcomes — no I/O, no network, no
ledger access. The PRIMARY contrast is ISTRATDE - RANDOM and the
SECONDARY contrast is ISTRATDE - FIXED, paired within (task, search
seed), aggregated as an equal-weight task macroaverage. Uncertainty uses
10000 bootstrap replicates that resample paired seed indices
INDEPENDENTLY within each task from ``random.Random(20261010)`` and a
percentile 95% interval (linear interpolation on the sorted replicate
statistics).

The estimand is the predeclared utility: final accuracy on success, 0 on
a genuine observed failure. A cell whose final outcome was never observed
(not started / interrupted / missing final) is NOT a zero — the pair is
unobserved, the estimate is computed over the observed pairs only and is
marked ``partial``; the predeclared full-matrix comparison is only
claimed when all 27 cells have observed final outcomes.

All estimates are conditional on the three fixed test sets, exploratory,
and based on only three seeds per task. No new seeds, tasks, runs or
budget are implied by any result.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261010
CONFIDENCE_LEVEL = 0.95

#: The predeclared contrasts: (name, treatment method, baseline method).
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("primary", "ISTRATDE", "RANDOM"),
    ("secondary", "ISTRATDE", "FIXED"),
)

ANALYSIS_CAVEATS: tuple[str, ...] = (
    "Conditional on the three fixed test sets (banknote/wdbc/spambase "
    "final folds); no generalization to other data is claimed.",
    "Exploratory evidence only: three search seeds per task.",
    "Utility is final accuracy on success and 0 on genuine observed "
    "failure; unobserved cells are excluded, never zero-filled.",
    "Equal candidate opportunities and caps do not imply equal realized "
    "token cost; cost totals are reported separately.",
)


@dataclass(frozen=True)
class CellOutcome:
    """One observed (or explicitly unobserved) cell outcome."""

    cell_key: str
    task_id: str
    method: str
    search_seed: int
    #: True iff the final outcome was actually observed (a persisted
    #: final result: completed / failed / no_valid_program).
    observed: bool
    #: Raw final accuracy, None for genuine failures and unobserved cells.
    raw_accuracy: float | None
    #: Predeclared utility: accuracy on success, 0.0 on observed failure,
    #: None when unobserved (never zero-filled).
    utility: float | None


def _task_macroaverage(diffs_by_task: Mapping[str, Sequence[float]]) -> float:
    """Equal-weight task macroaverage of within-task mean differences."""
    means = [
        sum(diffs) / len(diffs)
        for task, diffs in sorted(diffs_by_task.items())
        if diffs
    ]
    return sum(means) / len(means)


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' method)."""
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    position = (n - 1) * p
    lower = int(position)
    upper = min(lower + 1, n - 1)
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def bootstrap_macroaverage_interval(
    diffs_by_task: Mapping[str, Sequence[float]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    level: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Percentile bootstrap interval for the task macroaverage.

    Each replicate resamples the paired seed differences WITHIN each task
    independently (with replacement) and averages the resampled task
    means with equal task weights. Deterministic for the frozen seed.
    """
    tasks = [task for task, diffs in sorted(diffs_by_task.items()) if diffs]
    if not tasks:
        raise ValueError("bootstrap_requires_observed_pairs")
    rng = random.Random(seed)
    statistics: list[float] = []
    for _ in range(replicates):
        task_means: list[float] = []
        for task in tasks:
            diffs = diffs_by_task[task]
            sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
            task_means.append(sum(sample) / len(sample))
        statistics.append(sum(task_means) / len(task_means))
    statistics.sort()
    alpha = 1.0 - level
    return {
        "replicates": replicates,
        "seed": seed,
        "level": level,
        "method": "percentile",
        "low": _percentile(statistics, alpha / 2.0),
        "high": _percentile(statistics, 1.0 - alpha / 2.0),
    }


def paired_differences(
    outcomes: Mapping[str, CellOutcome],
    *,
    treatment: str,
    baseline: str,
) -> dict[str, Any]:
    """Per-(task, seed) paired utility differences treatment - baseline.

    Pairing is preserved across methods: a pair enters only when BOTH
    cells have observed final outcomes. Returns per-pair differences and
    the unobserved pairs explicitly.
    """
    paired: dict[str, dict[int, float]] = {}
    unobserved: list[str] = []
    keys = {
        (outcome.task_id, outcome.search_seed) for outcome in outcomes.values()
    }
    for task_id, seed in sorted(keys):
        treated = next(
            (
                o
                for o in outcomes.values()
                if o.task_id == task_id
                and o.search_seed == seed
                and o.method == treatment
            ),
            None,
        )
        base = next(
            (
                o
                for o in outcomes.values()
                if o.task_id == task_id
                and o.search_seed == seed
                and o.method == baseline
            ),
            None,
        )
        pair_key = f"{task_id}:{seed}"
        if (
            treated is None
            or base is None
            or not treated.observed
            or not base.observed
        ):
            unobserved.append(pair_key)
            continue
        # Both observed: utilities are floats (failures are genuine 0.0).
        assert treated.utility is not None and base.utility is not None
        paired.setdefault(task_id, {})[seed] = treated.utility - base.utility
    return {
        "diffs_by_task": {
            task: {seed: pairs[seed] for seed in sorted(pairs)}
            for task, pairs in sorted(paired.items())
        },
        "unobserved_pairs": unobserved,
        "observed_pairs": sum(len(pairs) for pairs in paired.values()),
    }


def analyze_contrast(
    outcomes: Mapping[str, CellOutcome],
    *,
    treatment: str,
    baseline: str,
    expected_pairs: int,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """One predeclared contrast with pairing, macroaverage and interval."""
    pairing = paired_differences(
        outcomes, treatment=treatment, baseline=baseline
    )
    diffs_by_seed: dict[str, dict[int, float]] = pairing["diffs_by_task"]
    diffs_by_task: dict[str, list[float]] = {
        task: [pairs[seed] for seed in sorted(pairs)]
        for task, pairs in diffs_by_seed.items()
    }
    observed = int(pairing["observed_pairs"])
    result: dict[str, Any] = {
        "treatment": treatment,
        "baseline": baseline,
        "expected_pairs": expected_pairs,
        "observed_pairs": observed,
        "unobserved_pairs": pairing["unobserved_pairs"],
        "partial": observed != expected_pairs,
        "paired_differences": {
            task: {str(seed): diff for seed, diff in pairs.items()}
            for task, pairs in diffs_by_seed.items()
        },
        "estimate": None,
        "interval": None,
    }
    if observed == 0:
        return result
    result["estimate"] = _task_macroaverage(diffs_by_task)
    result["interval"] = bootstrap_macroaverage_interval(
        diffs_by_task, replicates=replicates, seed=bootstrap_seed
    )
    return result


def conclude(
    primary: Mapping[str, Any],
    *,
    matrix_complete: bool,
) -> dict[str, Any]:
    """Honest positive / no-gain / insufficient-evidence conclusion.

    The predeclared full-matrix comparison is claimed ONLY when every
    planned cell has an observed final outcome; otherwise the evidence is
    insufficient no matter how the partial interval looks. On a complete
    matrix, positive requires the primary 95% interval entirely above 0.
    """
    if not matrix_complete:
        return {
            "conclusion": "insufficient-evidence",
            "reason": "incomplete_matrix:the predeclared 27-cell comparison "
            "was not fully observed; any estimate shown is partial",
        }
    interval = primary.get("interval")
    estimate = primary.get("estimate")
    if not isinstance(interval, dict) or estimate is None:
        return {
            "conclusion": "insufficient-evidence",
            "reason": "no_observed_pairs",
        }
    if float(interval["low"]) > 0.0:
        return {
            "conclusion": "positive",
            "reason": "primary ISTRATDE-RANDOM percentile 95% interval "
            "lies entirely above 0 on the complete 27-cell matrix",
        }
    return {
        "conclusion": "no-gain",
        "reason": "primary ISTRATDE-RANDOM percentile 95% interval does "
        "not lie entirely above 0 on the complete 27-cell matrix",
    }
