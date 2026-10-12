"""Offline-only schedule and descriptive analysis for calibration round two.

No credentials, network, model, sandbox, prior result, or final data access.
The plan is NOT authorization to execute. Statistical thresholds are exploratory.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def make_plan(protocol, option):
    if option not in ("small", "more_repeats"):
        raise ValueError("unknown_budget_option")
    budget = protocol["budget_options"][option]
    repeats = budget["repeats"]
    if repeats not in (3, 5):
        raise ValueError("invalid_repeats")
    grid = protocol["grid"]
    configurations = [
        {"config_id": f"c{index:02d}", "decoded": {
            **protocol["fixed_parameters"], "crossover_prob": crossover,
            "score_weight": score,
        }}
        for index, (crossover, score) in enumerate(itertools.product(
            grid["crossover_prob"], grid["score_weight"]))
    ]
    if len(configurations) != 9:
        raise ValueError("nine_configurations_required")
    rng = random.Random(protocol["schedule"]["schedule_seed"])
    schedule = []
    for repeat in range(repeats):
        inner_seed = int(hashlib.sha256(
            f"calibration-repeatability.v1|spambase|{repeat}".encode()
        ).hexdigest()[:8], 16)
        order = list(configurations)
        rng.shuffle(order)
        for config in order:
            schedule.append({"position": len(schedule), "repeat": repeat,
                             "trial_id": f"r{repeat:02d}-{config['config_id']}",
                             "inner_seed": inner_seed, **config})
    if len(schedule) != budget["trials"]:
        raise ValueError("trial_count_mismatch")
    plan = {"schema": "calibration-offline-plan.v1",
            "status": "PREPARED_ONLY_NOT_AUTHORIZED_FOR_LIVE_EXECUTION",
            "protocol_identity": identity(protocol), "budget_option": option,
            "budget": budget, "configurations": configurations,
            "center_config_id": "c04", "schedule": schedule}
    return {**plan, "plan_identity": identity(plan)}


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            result[index] = rank
        position = end
    return result


def spearman(left, right):
    x, y = ranks(left), ranks(right)
    xmean, ymean = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - xmean) * (b - ymean) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - xmean) ** 2 for a in x) *
                            sum((b - ymean) ** 2 for b in y))
    return numerator / denominator if denominator else None


def analyze(plan, outcomes):
    """Accept validation-only normalized outcomes, with every failure retained.

    Real output must be normalized by a separately reviewed trusted-result
    collector. This offline function does not authenticate sandbox evidence.
    """
    payload = {k: v for k, v in plan.items() if k != "plan_identity"}
    if identity(payload) != plan.get("plan_identity"):
        raise ValueError("plan_identity_mismatch")
    expected = {trial["trial_id"]: trial for trial in plan["schedule"]}
    by_id = {}
    total_requests = total_tokens = 0
    allowed_keys = {"trial_id", "status", "validation_accuracy", "requests",
                    "charged_tokens", "reason"}
    for outcome in outcomes:
        if set(outcome) - allowed_keys:
            raise ValueError("unexpected_outcome_fields")
        trial_id = outcome["trial_id"]
        if trial_id not in expected or trial_id in by_id:
            raise ValueError("unknown_or_duplicate_trial")
        status = outcome["status"]
        if status not in ("completed", "failed", "terminated"):
            raise ValueError("unknown_outcome_status")
        value = outcome.get("validation_accuracy")
        if value is not None and (type(value) not in (int, float) or
                not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("invalid_validation_accuracy")
        for name in ("requests", "charged_tokens"):
            if type(outcome.get(name)) is not int or outcome[name] < 0:
                raise ValueError("missing_or_invalid_cost")
        total_requests += outcome["requests"]
        total_tokens += outcome["charged_tokens"]
        by_id[trial_id] = outcome
    repeats = plan["budget"]["repeats"]
    configs = [config["config_id"] for config in plan["configurations"]]
    matrix = {config: [None] * repeats for config in configs}
    incomplete = []
    for trial_id, trial in expected.items():
        outcome = by_id.get(trial_id)
        if outcome and outcome["status"] == "completed" and outcome.get("validation_accuracy") is not None:
            matrix[trial["config_id"]][trial["repeat"]] = outcome["validation_accuracy"]
        else:
            incomplete.append({"trial_id": trial_id, "status": outcome["status"] if outcome else "missing",
                               "reason": outcome.get("reason") if outcome else "not_run"})
    summaries = []
    for config, values in matrix.items():
        observed = [value for value in values if value is not None]
        paired = [value - center for value, center in zip(values, matrix[plan["center_config_id"]])
                  if value is not None and center is not None]
        summaries.append({"config_id": config, "completed_observations": len(observed),
                          "mean": statistics.mean(observed) if observed else None,
                          "sample_sd": statistics.stdev(observed) if len(observed) > 1 else None,
                          "paired_center_count": len(paired),
                          "mean_paired_difference_from_center": statistics.mean(paired) if paired else None,
                          "repeat_values": values})
    report = {"schema": "calibration-offline-analysis.v1", "plan_identity": plan["plan_identity"],
              "provenance": "Normalized input only; trusted-result provenance must be checked before ingestion.",
              "expected_trials": len(expected), "reported_trials": len(by_id),
              "incomplete_trials": incomplete, "configuration_summaries": summaries,
              "recorded_cost": {"requests": total_requests, "charged_tokens": total_tokens},
              "verdict": "insufficient_complete_evidence", "variance": None, "rank_agreement": None}
    if incomplete:
        return report
    rows = [matrix[config] for config in configs]
    grand = statistics.mean(value for row in rows for value in row)
    row_means = [statistics.mean(row) for row in rows]
    column_means = [statistics.mean(row[repeat] for row in rows) for repeat in range(repeats)]
    ms_config = repeats * sum((mean - grand) ** 2 for mean in row_means) / (len(configs) - 1)
    ms_error = sum((value - row_means[i] - column_means[j] + grand) ** 2
                   for i, row in enumerate(rows) for j, value in enumerate(row)) / ((len(configs) - 1) * (repeats - 1))
    config_variance = max(0.0, (ms_config - ms_error) / repeats)
    denominator = config_variance + ms_error
    variance_fraction = config_variance / denominator if denominator else None
    correlations = []
    for first, second in itertools.combinations(range(repeats), 2):
        correlations.append(spearman([row[first] for row in rows], [row[second] for row in rows]))
    median_rank = statistics.median(correlations) if all(value is not None for value in correlations) else None
    report["variance"] = {"ms_configuration": ms_config, "ms_residual": ms_error,
                          "estimated_configuration_variance": config_variance,
                          "estimated_configuration_variance_fraction": variance_fraction}
    report["rank_agreement"] = {"pairwise_spearman": correlations, "median_spearman": median_rank}
    report["verdict"] = ("exploratory_stable_configuration_signal" if variance_fraction is not None
                         and variance_fraction >= 0.5 and median_rank is not None and median_rank >= 0.5
                         else "stable_configuration_signal_not_demonstrated")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--option", choices=("small", "more_repeats"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = make_plan(json.loads(args.protocol.read_text(encoding="utf-8")), args.option)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
