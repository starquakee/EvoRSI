"""Trusted independent scoring service (US-005).

- service: allowlisted task registry, bounded CSV scoring, prediction
  file defenses (symlink/traversal/oversize), worker-mount plan
  validation. The evaluator never runs submitted Python and never
  imports candidate-supplied metrics.
- metrics: the only metric implementations the evaluator may run.
"""
from .service import (
    DEFAULT_REGISTRY_PATH,
    REGISTRY_VERSION,
    EvaluatorError,
    EvaluatorRegistry,
    MountDenial,
    ScoreOutcome,
    TaskSpec,
    read_prediction_file,
    score_submission,
    validate_worker_mount_plan,
)

__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "REGISTRY_VERSION",
    "EvaluatorError",
    "EvaluatorRegistry",
    "MountDenial",
    "ScoreOutcome",
    "TaskSpec",
    "read_prediction_file",
    "score_submission",
    "validate_worker_mount_plan",
]
