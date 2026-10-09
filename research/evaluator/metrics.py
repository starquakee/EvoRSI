"""Trusted metric implementations for the independent evaluator (US-005).

These are the ONLY metric implementations the evaluator may run. They are
part of the trusted code base: the evaluator never imports candidate- or
task-supplied Python (no metric.py from any data directory). A registry
entry naming a metric absent from METRIC_FUNCTIONS fails closed at load.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Classification accuracy in [0, 1]; higher is better."""
    if len(y_true) != len(y_pred):
        raise ValueError("metric_input_length_mismatch")
    if not y_true:
        raise ValueError("metric_input_empty")
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    score = round(correct / len(y_true), 6)
    if not math.isfinite(score):
        raise ValueError("metric_nonfinite_score")
    return score


METRIC_FUNCTIONS: dict[str, Callable[[Sequence[int], Sequence[int]], float]] = {
    "accuracy": accuracy,
}
