"""Contract tests for research/contracts/results.py (US-003).

Pins the common result schema: explicit metric direction, fail-closed
finalization for every degraded outcome, and the cache-hit view
(zero incremental cost, retained provenance). Fully offline.
"""
from __future__ import annotations

import math

import pytest

from research.contracts.results import (
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

POLICY = "source-policy-v1"
SAFE = SafetyVerdict(ok=True, reason="ok", policy_version=POLICY)
COST = IncrementalCost(calls=2, tokens=1234, wall_seconds=12.5)
SOURCE_HASH = hash_source("print('hello')\n")


def _scored(score: float = 0.9, metric: str = "accuracy") -> EvaluationResult:
    return finalize_evaluation(
        metric=metric,
        exit_code=0,
        artifacts={"predictions.csv": True},
        score=score,
        safety=SAFE,
        source_hash=SOURCE_HASH,
        policy_version=POLICY,
        cost=COST,
        created_at=1_700_000_000.0,
    )


class TestMetricDirection:
    def test_accuracy_maximizes(self):
        assert metric_direction("accuracy") is MetricDirection.MAXIMIZE

    def test_logloss_minimizes(self):
        assert metric_direction("logloss") is MetricDirection.MINIMIZE

    def test_unknown_metric_fails_closed(self):
        with pytest.raises(ValueError, match="unknown_metric"):
            metric_direction("f1")

    def test_declared_map_is_exactly_the_contract(self):
        assert set(METRIC_DIRECTIONS) == {"accuracy", "logloss"}


class TestIsBetter:
    def test_maximize_direction(self):
        assert is_better(0.9, 0.8, MetricDirection.MAXIMIZE)
        assert not is_better(0.8, 0.9, MetricDirection.MAXIMIZE)
        assert not is_better(0.8, 0.8, MetricDirection.MAXIMIZE)

    def test_minimize_direction(self):
        assert is_better(0.1, 0.2, MetricDirection.MINIMIZE)
        assert not is_better(0.2, 0.1, MetricDirection.MINIMIZE)

    def test_nonfinite_never_wins(self):
        assert not is_better(math.inf, 0.0, MetricDirection.MAXIMIZE)
        assert not is_better(math.nan, 0.0, MetricDirection.MINIMIZE)
        assert is_better(0.5, math.inf, MetricDirection.MINIMIZE)


class TestHashing:
    def test_source_hash_is_content_hash(self):
        assert hash_source("abc") == hash_source(b"abc")
        assert hash_source("abc") != hash_source("abd")
        assert len(hash_source("abc")) == 64

    def test_config_hash_key_order_invariant(self):
        assert hash_config({"a": 1, "b": 2}) == hash_config({"b": 2, "a": 1})
        assert hash_config({"a": 1}) != hash_config({"a": 2})


class TestFinalizeEvaluation:
    def test_scored_success(self):
        result = _scored(0.9)
        assert result.status == "scored"
        assert result.reason == "ok"
        assert result.score == pytest.approx(0.9)
        assert result.direction is MetricDirection.MAXIMIZE
        assert result.cache_origin == "fresh"
        assert result.incremental_cost == COST
        assert result.recorded_cost == COST

    def test_missing_safety_verdict_fails_closed(self):
        result = finalize_evaluation(
            metric="accuracy", exit_code=0,
            artifacts={"predictions.csv": True}, score=0.9,
            safety=None, source_hash=SOURCE_HASH, policy_version=POLICY,
            cost=COST, created_at=1.0,
        )
        assert result.status == "failed"
        assert result.reason == "missing_safety_verdict"
        assert result.score is None
        assert not result.safety.ok

    def test_rejected_source(self):
        verdict = SafetyVerdict(
            ok=False, reason="forbidden_marker:subprocess", policy_version=POLICY,
        )
        result = finalize_evaluation(
            metric="accuracy", exit_code=None, artifacts={}, score=None,
            safety=verdict, source_hash=SOURCE_HASH, policy_version=POLICY,
            cost=ZERO_COST, created_at=1.0,
        )
        assert result.status == "rejected"
        assert result.reason == "forbidden_marker:subprocess"
        assert result.score is None

    def test_missing_artifact_fails_closed(self):
        result = finalize_evaluation(
            metric="accuracy", exit_code=0,
            artifacts={"predictions.csv": False, "metrics.json": True},
            score=0.9, safety=SAFE, source_hash=SOURCE_HASH,
            policy_version=POLICY, cost=COST, created_at=1.0,
        )
        assert result.status == "failed"
        assert result.reason == "missing_artifact:predictions.csv"
        assert result.score is None

    def test_nonzero_exit_fails_closed(self):
        result = finalize_evaluation(
            metric="logloss", exit_code=3,
            artifacts={"predictions.csv": True}, score=0.2,
            safety=SAFE, source_hash=SOURCE_HASH, policy_version=POLICY,
            cost=COST, created_at=1.0,
        )
        assert result.status == "failed"
        assert result.reason == "nonzero_exit:3"
        assert result.score is None
        assert result.direction is MetricDirection.MINIMIZE

    def test_missing_exit_status_fails_closed(self):
        result = finalize_evaluation(
            metric="accuracy", exit_code=None,
            artifacts={"predictions.csv": True}, score=0.9,
            safety=SAFE, source_hash=SOURCE_HASH, policy_version=POLICY,
            cost=COST, created_at=1.0,
        )
        assert result.status == "failed"
        assert result.reason == "missing_exit_status"

    def test_missing_score_fails_closed(self):
        result = finalize_evaluation(
            metric="accuracy", exit_code=0,
            artifacts={"predictions.csv": True}, score=None,
            safety=SAFE, source_hash=SOURCE_HASH, policy_version=POLICY,
            cost=COST, created_at=1.0,
        )
        assert result.status == "failed"
        assert result.reason == "missing_score"

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_nonfinite_score_fails_closed(self, bad: float):
        result = finalize_evaluation(
            metric="accuracy", exit_code=0,
            artifacts={"predictions.csv": True}, score=bad,
            safety=SAFE, source_hash=SOURCE_HASH, policy_version=POLICY,
            cost=COST, created_at=1.0,
        )
        assert result.status == "failed"
        assert result.reason == "nonfinite_score"
        assert result.score is None

    def test_safety_check_precedes_execution_evidence(self):
        # No safety verdict: even perfect artifacts/score cannot rescue it.
        result = finalize_evaluation(
            metric="accuracy", exit_code=0,
            artifacts={"predictions.csv": True}, score=1.0,
            safety=None, source_hash=SOURCE_HASH, policy_version=POLICY,
            cost=COST, created_at=1.0,
        )
        assert result.status == "failed"
        assert result.reason == "missing_safety_verdict"

    def test_unknown_metric_fails_closed(self):
        with pytest.raises(ValueError, match="unknown_metric"):
            finalize_evaluation(
                metric="bleu", exit_code=0, artifacts={}, score=1.0,
                safety=SAFE, source_hash=SOURCE_HASH, policy_version=POLICY,
                cost=COST, created_at=1.0,
            )


class TestResultInvariants:
    def test_scored_requires_finite_score(self):
        with pytest.raises(ValueError, match="finite_score"):
            EvaluationResult(
                metric="accuracy", direction=MetricDirection.MAXIMIZE,
                score=math.inf, status="scored", reason="ok",
                source_hash=SOURCE_HASH, policy_version=POLICY, safety=SAFE,
                cache_origin="fresh", incremental_cost=COST,
                recorded_cost=COST, created_at=1.0,
            )

    def test_failed_result_must_not_carry_score(self):
        with pytest.raises(ValueError, match="must_not_carry_score"):
            EvaluationResult(
                metric="accuracy", direction=MetricDirection.MAXIMIZE,
                score=0.5, status="failed", reason="nonzero_exit:1",
                source_hash=SOURCE_HASH, policy_version=POLICY, safety=SAFE,
                cache_origin="fresh", incremental_cost=COST,
                recorded_cost=COST, created_at=1.0,
            )

    def test_direction_mismatch_rejected(self):
        with pytest.raises(ValueError, match="direction_mismatch"):
            EvaluationResult(
                metric="accuracy", direction=MetricDirection.MINIMIZE,
                score=0.5, status="scored", reason="ok",
                source_hash=SOURCE_HASH, policy_version=POLICY, safety=SAFE,
                cache_origin="fresh", incremental_cost=COST,
                recorded_cost=COST, created_at=1.0,
            )

    def test_negative_cost_rejected(self):
        with pytest.raises(ValueError, match="negative_cost"):
            IncrementalCost(calls=-1)

    def test_serialization_roundtrip(self):
        result = _scored(0.75, metric="logloss")
        clone = EvaluationResult.from_dict(result.to_dict())
        assert clone == result


class TestCacheHitView:
    def test_mark_cached_zeroes_incremental_and_keeps_provenance(self):
        fresh = _scored(0.9)
        hit = fresh.mark_cached()
        assert hit.cache_origin == "cache"
        assert hit.incremental_cost == ZERO_COST
        # Original provenance retained.
        assert hit.recorded_cost == COST
        assert hit.created_at == fresh.created_at
        assert hit.source_hash == fresh.source_hash
        assert hit.policy_version == fresh.policy_version
        assert hit.score == pytest.approx(0.9)
        assert hit.status == "scored"

    def test_cache_hit_with_nonzero_incremental_is_rejected(self):
        fresh = _scored(0.9)
        with pytest.raises(ValueError, match="zero_incremental"):
            EvaluationResult(
                metric=fresh.metric, direction=fresh.direction,
                score=fresh.score, status=fresh.status, reason=fresh.reason,
                source_hash=fresh.source_hash,
                policy_version=fresh.policy_version, safety=fresh.safety,
                cache_origin="cache", incremental_cost=COST,
                recorded_cost=COST, created_at=fresh.created_at,
            )
