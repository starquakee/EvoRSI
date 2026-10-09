"""Shared contracts for the trustworthy loop (US-003).

- results: common evaluation result schema, explicit metric direction,
  fail-closed finalization.
- cache: deterministic scored-result cache; hits cost zero and retain
  original provenance.
- budget: persistent serial budget ledger (reserve before every model
  attempt, never reset on restart) and checkpoint-safe cancellation.
- source_gate: versioned pre-execution source policy shared by every
  submission path; machine-readable denials, fail-closed, advisory only
  (never a security boundary).
"""
from .budget import (
    BudgetError,
    BudgetExhausted,
    BudgetLedger,
    BudgetLimits,
    Cancelled,
    CancellationToken,
    LedgerLocked,
    LedgerStopped,
    ModelUsage,
    Reservation,
    call_with_retries,
    guarded_attempt,
)
from .cache import CacheCorruptError, CacheIdentity, ResultCache
from .source_gate import (
    POLICY_PATH_ENV,
    GateDenial,
    SourcePolicy,
    SourcePolicyError,
    SourceVerdict,
    check_source_bytes,
    check_source_tree,
    load_policy,
    policy_unavailable_verdict,
    validate_job_fields,
)
from .results import (
    METRIC_DIRECTIONS,
    ZERO_COST,
    EvaluationResult,
    IncrementalCost,
    MetricDirection,
    SafetyVerdict,
    finalize_evaluation,
    hash_config,
    hash_source,
    is_better,
    metric_direction,
)

__all__ = [
    "BudgetError",
    "BudgetExhausted",
    "BudgetLedger",
    "BudgetLimits",
    "CacheCorruptError",
    "CacheIdentity",
    "Cancelled",
    "CancellationToken",
    "EvaluationResult",
    "IncrementalCost",
    "LedgerLocked",
    "LedgerStopped",
    "METRIC_DIRECTIONS",
    "MetricDirection",
    "ModelUsage",
    "Reservation",
    "ResultCache",
    "SafetyVerdict",
    "SourcePolicy",
    "SourcePolicyError",
    "SourceVerdict",
    "POLICY_PATH_ENV",
    "ZERO_COST",
    "call_with_retries",
    "check_source_bytes",
    "check_source_tree",
    "finalize_evaluation",
    "guarded_attempt",
    "GateDenial",
    "hash_config",
    "hash_source",
    "is_better",
    "load_policy",
    "metric_direction",
    "policy_unavailable_verdict",
    "validate_job_fields",
]
