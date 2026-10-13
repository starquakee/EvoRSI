"""Offline contract checks; no credentials, provider, or sandbox use."""
from pathlib import Path

import pytest

from research.shared_start import specs

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_schedule_has_one_pool_then_exact_nine_configs_per_block():
    plan = specs.selected_plan()
    assert len(plan['schedule']) == 30
    assert len(plan['configurations']) == 9
    assert [r['position'] for r in plan['schedule']] == list(range(30))
    for block in range(3):
        items = plan['schedule'][block * 10:(block + 1) * 10]
        pool, *continuations = items
        assert pool['trial_id'] == f'b{block:02d}-pool'
        assert pool['kind'] == 'fresh_seed_pool'
        assert pool['config_id'] == 'c04'
        assert pool['limits'] == specs.POOL_LIMITS
        assert {r['config_id'] for r in continuations} == {f'c{i:02d}' for i in range(9)}
        assert {r['kind'] for r in continuations} == {'continuation'}
        assert {r['pool_id'] for r in items} == {pool['trial_id']}
        assert len({r['inner_seed'] for r in items}) == 1
        assert {r['repeat'] for r in items} == {block}
        assert [r['position_in_block'] for r in items] == list(range(10))
    assert sum(r['limits']['max_requests'] for r in plan['schedule']) == 174
    assert sum(r['limits']['max_main_requests'] for r in plan['schedule']) == 114
    assert sum(r['limits']['max_debug_requests'] for r in plan['schedule']) == 60


def test_approved_source_bytes_are_pinned(monkeypatch):
    original = Path.read_bytes
    def changed(path):
        value = original(path)
        if path.name == 'approved-plan.v1.json':
            return value.replace(b'2675261934', b'2675261935')
        return value
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(specs.CalibrationError, match='approved_plan_hash_mismatch'):
        specs.selected_plan()


def test_returned_contract_cannot_mutate_future_calls():
    plan = specs.selected_plan()
    identity = plan['plan_identity']
    plan['schedule'][0]['limits']['max_requests'] = 999
    plan['schedule'][1]['decoded']['score_weight'] = 999
    assert specs.selected_plan()['plan_identity'] == identity
    assert specs.selected_plan()['schedule'][0]['limits'] == specs.POOL_LIMITS


def test_kind_limits_and_new_identities_are_bound():
    roster = specs.trials(ROOT, 'a' * 40)
    pool, continuation = roster[:2]
    assert pool.main_target == 2 and continuation.main_target == 4
    assert pool.limits['max_tokens'] == 20000
    assert continuation.limits['max_tokens'] == 40000
    assert pool.to_dict()['schema'] == 'spambase-shared-start-pilot.v1'
    assert pool.pool_id == continuation.pool_id
    limits = pool.limits
    limits['max_tokens'] = 999999
    assert pool.limits['max_tokens'] == 20000
    assert specs.protocol()['retry_or_resume'] is False
    assert specs.protocol()['new_final_scoring'] is False
    assert specs.protocol()['global_limits']['max_requests'] == 174
    assert specs.RunSpec(pool, 'b' * 64).to_dict()['schema'] == 'shared-start-run-spec.v1'
