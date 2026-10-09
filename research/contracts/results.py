"""Common evaluation result schema with explicit metric direction (US-003).

Every evaluation the trustworthy loop records uses this contract:

- Metric direction is explicit: accuracy maximizes, logloss minimizes;
  unknown metrics fail closed (ValueError) instead of guessing.
- Provenance (source hash, policy version) and the safety verdict are
  bound to the result. An absent safety verdict fails closed.
- Missing artifacts, nonzero exit codes and nonfinite scores fail closed:
  they never produce a scored result.
- Cache hits share this schema: zero incremental cost, while the original
  provenance (created_at, recorded_cost, source_hash, policy_version) is
  retained on the returned record.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping


class MetricDirection(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "accuracy": MetricDirection.MAXIMIZE,
    "logloss": MetricDirection.MINIMIZE,
}

STATUS_SCORED = "scored"
STATUS_FAILED = "failed"
STATUS_REJECTED = "rejected"
_STATUSES = frozenset({STATUS_SCORED, STATUS_FAILED, STATUS_REJECTED})

ORIGIN_FRESH = "fresh"
ORIGIN_CACHE = "cache"
_ORIGINS = frozenset({ORIGIN_FRESH, ORIGIN_CACHE})


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def metric_direction(metric: str) -> MetricDirection:
    """Explicit direction per metric; unknown metrics fail closed."""
    try:
        return METRIC_DIRECTIONS[metric]
    except KeyError:
        raise ValueError(f"unknown_metric:{metric}") from None


def is_better(candidate: float, incumbent: float, direction: MetricDirection) -> bool:
    """Strict comparison honoring direction; nonfinite candidates never win."""
    if not math.isfinite(candidate):
        return False
    if not math.isfinite(incumbent):
        return True
    if direction is MetricDirection.MAXIMIZE:
        return candidate > incumbent
    return candidate < incumbent


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_source(source: str | bytes) -> str:
    """Content hash of the exact submitted source bytes."""
    data = source.encode("utf-8") if isinstance(source, str) else source
    return hash_bytes(data)


def hash_config(config: Mapping[str, Any]) -> str:
    """Canonical hash of a config mapping (sorted keys, stable separators)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hash_bytes(canonical.encode("utf-8"))


@dataclass(frozen=True)
class SafetyVerdict:
    """Outcome of the fixed pre-execution source policy (bound per result)."""

    ok: bool
    reason: str
    policy_version: str

    def __post_init__(self) -> None:
        if type(self.ok) is not bool:
            raise ValueError("safety_verdict_requires_boolean")
        if not isinstance(self.policy_version, str) or not self.policy_version:
            raise ValueError("missing_policy_version")


@dataclass(frozen=True)
class IncrementalCost:
    """Cost attributable to one evaluation: model calls, tokens, wall seconds."""

    calls: int = 0
    tokens: int = 0
    wall_seconds: float = 0.0

    def __post_init__(self) -> None:
        if type(self.calls) is not int or type(self.tokens) is not int:
            raise ValueError("cost_counts_require_integers")
        if not _finite_number(self.wall_seconds):
            raise ValueError("nonfinite_cost")
        if self.calls < 0 or self.tokens < 0 or self.wall_seconds < 0:
            raise ValueError("negative_cost")

    @classmethod
    def zero(cls) -> "IncrementalCost":
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "tokens": self.tokens,
            "wall_seconds": self.wall_seconds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IncrementalCost":
        return cls(
            calls=data["calls"],
            tokens=data["tokens"],
            wall_seconds=data["wall_seconds"],
        )


ZERO_COST = IncrementalCost()


@dataclass(frozen=True)
class EvaluationResult:
    """One recorded evaluation. Invariants are enforced on construction.

    status:    scored / failed / rejected (machine-readable reason always set)
    cache_origin: fresh (this evaluation ran) / cache (served from ResultCache)
    incremental_cost: cost attributable to THIS record; always zero on a hit
    recorded_cost:    cost of the original evaluation (provenance, retained
                      on cache hits)
    created_at:       epoch seconds of the ORIGINAL evaluation
    """

    metric: str
    direction: MetricDirection
    score: float | None
    status: str
    reason: str
    source_hash: str
    policy_version: str
    safety: SafetyVerdict
    cache_origin: str
    incremental_cost: IncrementalCost
    recorded_cost: IncrementalCost
    created_at: float

    def __post_init__(self) -> None:
        if not _finite_number(self.created_at) or self.created_at < 0:
            raise ValueError("invalid_created_at")
        if self.safety.policy_version != self.policy_version:
            raise ValueError("safety_policy_version_mismatch")
        if self.status not in _STATUSES:
            raise ValueError(f"unknown_status:{self.status}")
        if self.cache_origin not in _ORIGINS:
            raise ValueError(f"unknown_cache_origin:{self.cache_origin}")
        if metric_direction(self.metric) is not self.direction:
            raise ValueError(f"direction_mismatch:{self.metric}")
        if self.status == STATUS_SCORED:
            if not _finite_number(self.score):
                raise ValueError("scored_requires_finite_score")
            if not self.safety.ok:
                raise ValueError("scored_requires_safety_ok")
        elif self.score is not None:
            raise ValueError("unscored_result_must_not_carry_score")
        if self.cache_origin == ORIGIN_CACHE and self.incremental_cost != ZERO_COST:
            raise ValueError("cache_hit_must_have_zero_incremental_cost")

    def mark_cached(self) -> "EvaluationResult":
        """Return the cache-hit view: zero incremental cost, same provenance."""
        return replace(
            self, cache_origin=ORIGIN_CACHE, incremental_cost=ZERO_COST,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "score": self.score,
            "status": self.status,
            "reason": self.reason,
            "source_hash": self.source_hash,
            "policy_version": self.policy_version,
            "safety": {
                "ok": self.safety.ok,
                "reason": self.safety.reason,
                "policy_version": self.safety.policy_version,
            },
            "cache_origin": self.cache_origin,
            "incremental_cost": self.incremental_cost.to_dict(),
            "recorded_cost": self.recorded_cost.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationResult":
        """Strict deserialize; malformed records raise (fail closed)."""
        safety = data["safety"]
        if not isinstance(safety, Mapping):
            raise ValueError("malformed_safety_verdict")
        return cls(
            metric=str(data["metric"]),
            direction=MetricDirection(str(data["direction"])),
            score=data["score"],
            status=str(data["status"]),
            reason=str(data["reason"]),
            source_hash=str(data["source_hash"]),
            policy_version=str(data["policy_version"]),
            safety=SafetyVerdict(
                ok=safety["ok"],
                reason=str(safety["reason"]),
                policy_version=str(safety["policy_version"]),
            ),
            cache_origin=str(data["cache_origin"]),
            incremental_cost=IncrementalCost.from_dict(data["incremental_cost"]),
            recorded_cost=IncrementalCost.from_dict(data["recorded_cost"]),
            created_at=data["created_at"],
        )


def finalize_evaluation(
    *,
    metric: str,
    exit_code: int | None,
    artifacts: Mapping[str, bool],
    score: float | None,
    safety: SafetyVerdict | None,
    source_hash: str,
    policy_version: str,
    cost: IncrementalCost,
    created_at: float,
) -> EvaluationResult:
    """Build an EvaluationResult from raw execution evidence, fail closed.

    Check order is fixed: an absent safety verdict means nothing else can
    be trusted; a rejected source never reaches execution evidence; only
    then are artifacts, exit code and score considered.
    """
    direction = metric_direction(metric)

    def _result(status: str, reason: str, final_score: float | None) -> EvaluationResult:
        verdict = safety if safety is not None else SafetyVerdict(
            ok=False, reason="missing_safety_verdict", policy_version=policy_version,
        )
        return EvaluationResult(
            metric=metric,
            direction=direction,
            score=final_score,
            status=status,
            reason=reason,
            source_hash=source_hash,
            policy_version=policy_version,
            safety=verdict,
            cache_origin=ORIGIN_FRESH,
            incremental_cost=cost,
            recorded_cost=cost,
            created_at=created_at,
        )

    if safety is None:
        return _result(STATUS_FAILED, "missing_safety_verdict", None)
    if not safety.ok:
        return _result(STATUS_REJECTED, safety.reason, None)
    if not artifacts:
        return _result(STATUS_FAILED, "missing_artifacts", None)
    if any(type(present) is not bool for present in artifacts.values()):
        return _result(STATUS_FAILED, "malformed_artifact_evidence", None)
    missing = sorted(name for name, present in artifacts.items() if not present)
    if missing:
        return _result(STATUS_FAILED, "missing_artifact:" + ",".join(missing), None)
    if exit_code is None:
        return _result(STATUS_FAILED, "missing_exit_status", None)
    if type(exit_code) is not int:
        return _result(STATUS_FAILED, "invalid_exit_status", None)
    if exit_code != 0:
        return _result(STATUS_FAILED, f"nonzero_exit:{exit_code}", None)
    if score is None:
        return _result(STATUS_FAILED, "missing_score", None)
    if type(score) not in (int, float):
        return _result(STATUS_FAILED, "invalid_score", None)
    if not math.isfinite(score):
        return _result(STATUS_FAILED, "nonfinite_score", None)
    return _result(STATUS_SCORED, "ok", float(score))
