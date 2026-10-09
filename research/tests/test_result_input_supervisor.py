"""Raw/cached evidence must not become a score by Python truthiness."""
import pytest
from research.contracts.results import EvaluationResult, IncrementalCost, SafetyVerdict, finalize_evaluation, hash_source


def result(**overrides):
    args=dict(metric='accuracy',exit_code=0,artifacts={'predictions.csv':True},score=0.9,
              safety=SafetyVerdict(True,'ok','v1'),source_hash=hash_source('print(1)'),
              policy_version='v1',cost=IncrementalCost(),created_at=1.0)
    args.update(overrides)
    return finalize_evaluation(**args)


@pytest.mark.parametrize('overrides',[
    {'artifacts':{}}, {'artifacts':{'predictions.csv':'false'}},
    {'score':True}, {'exit_code':False},
])
def test_missing_or_malformed_evidence_is_not_scored(overrides):
    assert result(**overrides).status!='scored'


@pytest.mark.parametrize('field,value',[
    ('score',True),('score','0.9'),('created_at',float('nan')),('created_at',True),
])
def test_cached_numeric_fields_do_not_coerce_invalid_types(field,value):
    payload=result().to_dict();payload[field]=value
    with pytest.raises(ValueError):
        EvaluationResult.from_dict(payload)
