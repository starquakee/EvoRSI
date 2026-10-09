"""Strict deserialization and cache provenance regression tests."""
import json
from dataclasses import replace
from pathlib import Path
import pytest
from research.contracts.results import EvaluationResult, IncrementalCost, SafetyVerdict, finalize_evaluation, hash_source
from research.contracts.cache import ResultCache, CacheIdentity, CacheCorruptError

def scored():
    return finalize_evaluation(metric="accuracy", exit_code=0, artifacts={"predictions.csv": True}, score=0.9,
       safety=SafetyVerdict(True,"ok","v1"), source_hash=hash_source("print(1)"),policy_version="v1",cost=IncrementalCost(),created_at=0.0)

@pytest.mark.parametrize("bad", ["false", "true", 0, 1, None])
def test_safety_bool_is_not_truthiness(bad):
    data=scored().to_dict(); data["safety"]["ok"]=bad
    with pytest.raises(ValueError): EvaluationResult.from_dict(data)

def test_verdict_must_match_result_policy():
    with pytest.raises(ValueError): replace(scored(), policy_version="v2")

@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_cost_requires_finite_time(bad):
    with pytest.raises(ValueError): IncrementalCost(wall_seconds=bad)

def test_cache_rejects_other_policy_or_metric(tmp_path: Path):
    result=scored(); cache=ResultCache(tmp_path/"cache.json")
    key=CacheIdentity(42,"model","task","data","accuracy","v1","config",result.source_hash)
    for changed in [replace(key,policy_version="v2"),replace(key,metric="logloss")]:
        with pytest.raises(ValueError):cache.put(changed,result)

def test_cache_rejects_corrupted_record_provenance(tmp_path: Path):
    result=scored(); path=tmp_path/"cache.json";cache=ResultCache(path)
    key=CacheIdentity(42,"model","task","data","accuracy","v1","config",result.source_hash)
    cache.put(key,result); data=json.loads(path.read_text())
    data["records"][key.key()]["source_hash"]="other"
    path.write_text(json.dumps(data)); reopened=ResultCache(path)
    with pytest.raises(CacheCorruptError):reopened.get(key)
