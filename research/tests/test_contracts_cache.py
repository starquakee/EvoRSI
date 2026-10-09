"""Contract tests for research/contracts/cache.py (US-003).

Pins cache-key identity (seed/model/task/data/metric/policy/config/source),
zero incremental cost on hits with retained provenance, fail-closed
refusals (only fresh scored results cached) and fail-closed handling of
corrupt cache files. Fully offline, tmp_path only.
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from research.contracts.cache import (
    CacheCorruptError,
    CacheIdentity,
    ResultCache,
)
from research.contracts.results import (
    EvaluationResult,
    IncrementalCost,
    SafetyVerdict,
    finalize_evaluation,
    hash_config,
    hash_source,
)

POLICY = "source-policy-v1"
SOURCE_HASH = hash_source("print('candidate')\n")
CONFIG_HASH = hash_config({"crossover_prob": 0.5, "max_debug_depth": 4})


def _identity(**overrides: Any) -> CacheIdentity:
    base: dict[str, Any] = {
        "seed": 7,
        "model": "k3",
        "task_id": "hello_synth",
        "data_version": "v2026-10-01",
        "metric": "accuracy",
        "policy_version": POLICY,
        "config_hash": CONFIG_HASH,
        "source_hash": SOURCE_HASH,
    }
    base.update(overrides)
    return CacheIdentity(**base)


def _scored(score: float = 0.9, created_at: float = 1_700_000_000.0):
    cost = IncrementalCost(calls=3, tokens=4321, wall_seconds=21.0)
    result = finalize_evaluation(
        metric="accuracy",
        exit_code=0,
        artifacts={"predictions.csv": True},
        score=score,
        safety=SafetyVerdict(ok=True, reason="ok", policy_version=POLICY),
        source_hash=SOURCE_HASH,
        policy_version=POLICY,
        cost=cost,
        created_at=created_at,
    )
    assert result.status == "scored"
    return result


class TestCacheIdentity:
    def test_key_is_deterministic(self):
        assert _identity().key() == _identity().key()
        assert len(_identity().key()) == 64

    @pytest.mark.parametrize(
        "field,value",
        [
            ("seed", 8),
            ("model", "other-model"),
            ("task_id", "other_task"),
            ("data_version", "v2026-10-02"),
            ("metric", "logloss"),
            ("policy_version", "source-policy-v2"),
            ("config_hash", hash_config({"crossover_prob": 0.6})),
            ("source_hash", hash_source("print('other')\n")),
        ],
    )
    def test_every_identity_field_changes_key(self, field: str, value):
        assert _identity(**{field: value}).key() != _identity().key()


class TestResultCache:
    def test_miss_returns_none(self, tmp_path):
        cache = ResultCache(tmp_path / "cache.json")
        assert cache.get(_identity()) is None
        assert len(cache) == 0

    def test_put_get_roundtrip_within_process(self, tmp_path):
        cache = ResultCache(tmp_path / "cache.json")
        cache.put(_identity(), _scored(0.9))
        hit = cache.get(_identity())
        assert hit is not None
        assert hit.score == pytest.approx(0.9)

    def test_hit_has_zero_incremental_cost_and_original_provenance(
        self, tmp_path,
    ):
        cache = ResultCache(tmp_path / "cache.json")
        original = _scored(0.9, created_at=1_700_000_123.0)
        cache.put(_identity(), original)
        hit = cache.get(_identity())
        assert hit is not None
        assert hit.cache_origin == "cache"
        assert hit.incremental_cost.calls == 0
        assert hit.incremental_cost.tokens == 0
        assert hit.incremental_cost.wall_seconds == 0.0
        # Original provenance retained, not refreshed.
        assert hit.recorded_cost == original.recorded_cost
        assert hit.recorded_cost.tokens == 4321
        assert hit.created_at == pytest.approx(1_700_000_123.0)
        assert hit.source_hash == SOURCE_HASH
        assert hit.policy_version == POLICY
        assert hit.status == "scored"

    def test_persistence_across_instances(self, tmp_path):
        path = tmp_path / "cache.json"
        ResultCache(path).put(_identity(), _scored(0.8))
        reopened = ResultCache(path)
        assert len(reopened) == 1
        hit = reopened.get(_identity())
        assert hit is not None
        assert hit.score == pytest.approx(0.8)
        assert hit.cache_origin == "cache"

    def test_different_identity_misses(self, tmp_path):
        cache = ResultCache(tmp_path / "cache.json")
        cache.put(_identity(), _scored(0.8))
        assert cache.get(_identity(seed=99)) is None

    def test_failed_result_is_not_cached(self, tmp_path):
        cache = ResultCache(tmp_path / "cache.json")
        failed = finalize_evaluation(
            metric="accuracy", exit_code=1,
            artifacts={"predictions.csv": True}, score=None,
            safety=SafetyVerdict(ok=True, reason="ok", policy_version=POLICY),
            source_hash=SOURCE_HASH, policy_version=POLICY,
            cost=IncrementalCost(calls=1, tokens=10, wall_seconds=1.0),
            created_at=1.0,
        )
        assert failed.status == "failed"
        with pytest.raises(ValueError, match="only_fresh_scored"):
            cache.put(_identity(), failed)

    def test_cache_hit_cannot_be_recached(self, tmp_path):
        cache = ResultCache(tmp_path / "cache.json")
        cache.put(_identity(), _scored(0.8))
        hit = cache.get(_identity())
        assert hit is not None
        with pytest.raises(ValueError, match="only_fresh_scored"):
            cache.put(_identity(), hit)

    def test_source_hash_mismatch_refused(self, tmp_path):
        cache = ResultCache(tmp_path / "cache.json")
        with pytest.raises(ValueError, match="source_hash_mismatch"):
            cache.put(_identity(source_hash=hash_source("x")), _scored(0.8))


class TestCorruptCacheFailsClosed:
    def test_malformed_json(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(CacheCorruptError):
            ResultCache(path)

    def test_wrong_schema_version(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({"schema_version": 999, "records": {}}))
        with pytest.raises(CacheCorruptError, match="schema_version"):
            ResultCache(path)

    def test_invalid_record(self, tmp_path):
        path = tmp_path / "cache.json"
        record = _scored(0.8).to_dict()
        record["score"] = "not-a-number"
        key = _identity().key()
        path.write_text(json.dumps({
            "schema_version": 1, "records": {key: record},
        }))
        with pytest.raises(CacheCorruptError, match="invalid_record"):
            ResultCache(path)

    def test_non_object_payload(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(CacheCorruptError):
            ResultCache(path)


class TestIdentityStabilityAcrossDataclass:
    def test_replace_changes_only_named_field(self):
        base = _identity()
        other = replace(base, metric="logloss")
        assert other.key() != base.key()
        assert replace(base, seed=base.seed).key() == base.key()
