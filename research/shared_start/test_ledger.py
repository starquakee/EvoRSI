"""Synthetic regression tests only: no credentials, model, or sandbox access."""
import json
from pathlib import Path

import pytest

from research.shared_start.ledger import CalibrationLedger
from research.shared_start.specs import CalibrationError, RunSpec, protocol, selected_plan, trials
from research.comparison.budget import (
    CellBudgetExhausted, ComparisonBudgetError, ComparisonLedgerLocked,
    ComparisonLedgerStopped, GlobalBudgetExhausted, InnerRequestLimitReached,
    ProviderUsage, guarded_comparison_attempt,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def roster():
    return trials(ROOT, 'a' * 40)


@pytest.fixture
def now():
    return [1000.0]


@pytest.fixture
def ledger(tmp_path, roster, now):
    handle = CalibrationLedger.create(tmp_path / 'ledger.json', roster=roster,
                                     manifest_identity='b' * 64, now_fn=lambda: now[0])
    yield handle
    handle.close()


def begin(ledger, roster):
    ledger.start_clock()
    ledger.begin_trial(roster[0])
    return roster[0]


def begin_at(ledger, roster, index):
    ledger.start_clock()
    for previous in roster[:index]:
        ledger.begin_trial(previous)
        ledger.finish_trial(previous, 'completed')
        if previous.kind == 'fresh_seed_pool':
            ledger.seal_seed_pool(previous, 'd' * 64)
    ledger.begin_trial(roster[index])
    return roster[index]


def attempt(ledger, run, *, tokens=10, kind='main', input_estimate=100):
    return guarded_comparison_attempt(ledger, run, kind, lambda _: 'response', None,
            estimated_input_tokens=input_estimate,
            usage_fn=lambda _: ProviderUsage(tokens, 0), debug_limit=2 if kind == 'debug' else None)


def test_selected_roster_and_limits_are_independent_of_old_comparison(roster):
    assert len(roster) == 30
    assert len({trial.identity() for trial in roster}) == 30
    assert {trial.task for trial in roster} == {'spambase'}
    assert protocol()['global_limits'] == {'max_requests': 174, 'max_tokens': 600000, 'max_elapsed_seconds': 14400}
    spec = RunSpec(roster[0], 'b' * 64)
    assert 'shared-start-pilot' in spec.run_dir
    assert 'comparison-istratde' not in spec.ledger_path
    assert spec.to_dict()['decoded'] == selected_plan()['schedule'][0]['decoded']
    with pytest.raises(CalibrationError):
        RunSpec(roster[0], 'b' * 64, job_timeout_seconds=601).validate()


def test_clock_and_trial_required_before_reserving(ledger, roster):
    with pytest.raises(ComparisonBudgetError, match='clock_not_started'):
        ledger.reserve(roster[0], 'main', estimated_input_tokens=100)
    ledger.start_clock()
    with pytest.raises(ComparisonBudgetError, match='trial_not_active'):
        ledger.reserve(roster[0], 'main', estimated_input_tokens=100)
    with pytest.raises(ComparisonBudgetError, match='clock_already'):
        ledger.start_clock()


def test_atomic_reservation_precedes_transport_and_settlement(ledger, roster):
    run = begin(ledger, roster)
    def transport(_):
        state = json.loads(ledger.path.read_text())
        assert len(state['requests']) == 1
        assert state['requests'][0]['status'] == 'reserved'
        assert state['requests'][0]['charged_tokens'] == 8292
        return 'response'
    guarded_comparison_attempt(ledger, run, 'main', transport, None,
        estimated_input_tokens=100, usage_fn=lambda _: ProviderUsage(20, 30))
    snapshot = ledger.snapshot()
    assert snapshot['global']['committed'] == {'requests': 1, 'tokens': 50}
    assert snapshot['cells'][run.trial_id]['remaining']['tokens'] == 19950


@pytest.mark.parametrize('index,main_limit,total', [(0, 2, 4), (1, 4, 6)])
def test_kind_specific_main_and_debug_ceiling(ledger, roster, index, main_limit, total):
    run = begin_at(ledger, roster, index)
    for _ in range(main_limit):
        attempt(ledger, run)
    with pytest.raises(InnerRequestLimitReached):
        attempt(ledger, run)
    for _ in range(2):
        attempt(ledger, run, kind='debug')
    with pytest.raises(InnerRequestLimitReached):
        attempt(ledger, run, kind='debug')
    assert ledger.snapshot()['global']['committed'] == {'requests': total, 'tokens': 10 * total}


@pytest.mark.parametrize('index,cap', [(0, 20000), (1, 40000)])
def test_unit_conservative_token_exhaustion_does_not_borrow(ledger, roster, index, cap):
    run = begin_at(ledger, roster, index)
    attempt(ledger, run, tokens=cap - 8000, input_estimate=cap - 8000 - 8192)
    with pytest.raises(CellBudgetExhausted):
        attempt(ledger, run)
    assert not ledger.stopped
    ledger.finish_trial(run, 'terminated')
    with pytest.raises(ComparisonBudgetError, match='frozen_schedule_predecessor_incomplete'):
        ledger.begin_trial(roster[index + 1])
    assert ledger.stopped
    assert ledger.snapshot()['global']['committed']['requests'] == 1


def test_global_token_ceiling_stops_without_sending_extra_request(ledger, roster):
    ledger.start_clock()
    charged = 0
    sent = 0
    for run in roster:
        ledger.begin_trial(run)
        tokens = run.limits['max_tokens'] - 1000
        if charged + tokens > 600000:
            with pytest.raises(GlobalBudgetExhausted):
                attempt(ledger, run, tokens=tokens, input_estimate=tokens - 8192)
            break
        attempt(ledger, run, tokens=tokens, input_estimate=tokens - 8192)
        charged += tokens
        sent += 1
        ledger.finish_trial(run, 'completed')
        if run.kind == 'fresh_seed_pool':
            ledger.seal_seed_pool(run, 'd' * 64)
    assert ledger.stopped
    assert ledger.snapshot()['global']['committed'] == {'requests': sent, 'tokens': charged}


def test_exact_full_roster_ceiling_is_174_requests(ledger, roster):
    ledger.start_clock()
    for run in roster:
        ledger.begin_trial(run)
        for _ in range(run.main_target):
            attempt(ledger, run)
        for _ in range(2):
            attempt(ledger, run, kind='debug')
        ledger.finish_trial(run, 'completed')
        if run.kind == 'fresh_seed_pool':
            ledger.seal_seed_pool(run, 'd' * 64)
    state = ledger.snapshot()
    assert state['global']['committed'] == {'requests': 174, 'tokens': 1740}
    assert state['global']['remaining']['requests'] == 0
    assert sum(r['kind'] == 'main' for r in state['requests']) == 114
    assert sum(r['kind'] == 'debug' for r in state['requests']) == 60
    with pytest.raises(ComparisonBudgetError):
        ledger.begin_trial(roster[0])
    assert ledger.snapshot()['global']['committed']['requests'] == 174


def test_reading_shared_pool_contract_does_not_double_charge(ledger, roster):
    run = begin(ledger, roster)
    for _ in range(2):
        attempt(ledger, run)
    ledger.finish_trial(run, 'completed')
    before = ledger.snapshot()['global']['committed']
    ledger.seal_seed_pool(run, 'd' * 64)
    for continuation in roster[1:10]:
        assert continuation.pool_id == run.trial_id
        RunSpec(continuation, 'b' * 64).to_dict()
    assert ledger.snapshot()['global']['committed'] == before
    assert all(ledger.snapshot()['cells'][r.trial_id]['committed']['requests'] == 0
               for r in roster[1:10])


def test_unknown_usage_retains_full_reservation_and_stops(ledger, roster):
    run = begin(ledger, roster)
    guarded_comparison_attempt(ledger, run, 'main', lambda _: 'missing usage', None,
            estimated_input_tokens=100, usage_fn=lambda _: None)
    assert ledger.stopped
    assert ledger.snapshot()['global']['committed'] == {'requests': 1, 'tokens': 8292}
    with pytest.raises(ComparisonLedgerStopped):
        attempt(ledger, run)


def test_failed_observer_is_charged_and_transport_never_runs(ledger, roster):
    run = begin(ledger, roster)
    calls = []
    def observer(_):
        raise RuntimeError('synthetic observer failure')
    with pytest.raises(RuntimeError):
        guarded_comparison_attempt(ledger, run, 'main', lambda _: calls.append(True), None,
            estimated_input_tokens=100, usage_fn=lambda _: ProviderUsage(1, 1), on_reserve=observer)
    assert calls == []
    assert ledger.stopped
    assert ledger.snapshot()['global']['committed']['tokens'] == 8292


def test_duplicate_provider_id_cannot_double_credit(ledger, roster):
    run = begin(ledger, roster)
    for index in range(2):
        context = pytest.raises(ComparisonBudgetError) if index else __import__('contextlib').nullcontext()
        with context:
            guarded_comparison_attempt(ledger, run, 'main', lambda _: 'response', None,
                estimated_input_tokens=100, usage_fn=lambda _: ProviderUsage(1, 1),
                provider_request_id_fn=lambda _: 'same-provider-id')
    assert ledger.stopped
    assert ledger.snapshot()['global']['committed'] == {'requests': 2, 'tokens': 8294}


def test_restart_with_open_reservation_retains_cost_and_cannot_resume(ledger, roster, now):
    run = begin(ledger, roster)
    ledger.reserve(run, 'main', estimated_input_tokens=100)
    ledger.close()
    with CalibrationLedger.open(ledger.path, roster=roster, manifest_identity='b' * 64, now_fn=lambda: now[0]) as reopened:
        assert reopened.stopped
        assert reopened.started_at == 1000
        assert reopened.snapshot()['global']['committed']['tokens'] == 8292


def test_completed_trial_cannot_rerun(ledger, roster):
    run = begin(ledger, roster)
    attempt(ledger, run)
    ledger.finish_trial(run, 'completed')
    with pytest.raises(ComparisonBudgetError, match='trial_not_fresh'):
        ledger.begin_trial(run)
    assert ledger.stopped


def test_exclusive_process_lock_and_no_create_reset(ledger, roster):
    with pytest.raises(ComparisonLedgerLocked):
        CalibrationLedger.open(ledger.path, roster=roster, manifest_identity='b' * 64)
    ledger.close()
    with pytest.raises(ComparisonBudgetError, match='already_exists'):
        CalibrationLedger.create(ledger.path, roster=roster, manifest_identity='b' * 64)


@pytest.mark.parametrize('elapsed', [900, 14400, -1])
def test_deadline_or_backwards_clock_refuses_before_transport(ledger, roster, now, elapsed):
    run = begin(ledger, roster)
    now[0] += elapsed
    with pytest.raises(GlobalBudgetExhausted):
        attempt(ledger, run)
    assert ledger.snapshot()['global']['committed']['requests'] == 0


@pytest.mark.parametrize('mutation', ['limits', 'manifest', 'usage', 'old_schema', 'unit_limits'])
def test_corrupt_or_foreign_ledger_is_not_rewritten(ledger, roster, now, mutation):
    run = begin(ledger, roster)
    attempt(ledger, run)
    ledger.finish_trial(run, 'completed')
    ledger.close()
    state = json.loads(ledger.path.read_text())
    if mutation == 'limits':
        state['limits']['global']['max_tokens'] += 1
    elif mutation == 'manifest':
        state['manifest_identity'] = 'c' * 64
    elif mutation == 'old_schema':
        state['schema'] = 'calibration-budget-ledger.v1'
    elif mutation == 'unit_limits':
        state['trials'][run.identity()]['limits']['max_tokens'] = 40000
    else:
        state['requests'][0]['charged_tokens'] += 1
    ledger.path.write_text(json.dumps(state))
    before = ledger.path.read_bytes()
    with pytest.raises(ComparisonBudgetError):
        CalibrationLedger.open(ledger.path, roster=roster, manifest_identity='b' * 64, now_fn=lambda: now[0])
    assert ledger.path.read_bytes() == before


def test_shared_ledger_rejects_historic_calibration_trial(tmp_path):
    from research.calibration.specs import trials as old_trials
    with pytest.raises(ComparisonBudgetError, match='ledger_roster_invalid'):
        CalibrationLedger.create(tmp_path / 'foreign.json', roster=old_trials(ROOT, 'a' * 40),
                                 manifest_identity='b' * 64)
    assert not (tmp_path / 'foreign.json').exists()


@pytest.mark.parametrize('index,deadline', [(0, 1900), (1, 2800)])
def test_dynamic_unit_deadline(ledger, roster, index, deadline):
    begin_at(ledger, roster, index)
    assert ledger.deadline_epoch == deadline


@pytest.mark.parametrize('index', [1, 10, 29])
def test_frozen_schedule_cannot_skip_pending_predecessors(ledger, roster, index):
    ledger.start_clock()
    with pytest.raises(ComparisonBudgetError, match='frozen_schedule_predecessor_incomplete'):
        ledger.begin_trial(roster[index])
    assert ledger.stopped
    assert ledger.snapshot()['global']['committed']['requests'] == 0


def test_roster_order_cannot_be_changed(tmp_path, roster):
    with pytest.raises(ComparisonBudgetError, match='ledger_roster_order_invalid'):
        CalibrationLedger.create(tmp_path / 'wrong-order.json', roster=tuple(reversed(roster)),
                                 manifest_identity='b' * 64)


def test_terminated_pool_cannot_be_relabelled_completed(ledger, roster):
    run = begin(ledger, roster)
    ledger.terminate_cell(run.identity(), 'synthetic_cap_exhaustion')
    with pytest.raises(ComparisonBudgetError, match='terminated_trial_cannot_complete'):
        ledger.finish_trial(run, 'completed')
    with pytest.raises(ComparisonBudgetError, match='frozen_schedule_predecessor_incomplete'):
        ledger.begin_trial(roster[1])


def test_continuation_requires_ledger_pool_seal(ledger, roster):
    run = begin(ledger, roster)
    ledger.finish_trial(run, 'completed')
    with pytest.raises(ComparisonBudgetError, match='continuation_without_pool_seal'):
        ledger.begin_trial(roster[1])
    assert ledger.stopped


def test_seed_pool_seal_is_durable_immutable_and_uncharged(ledger, roster, now):
    run = begin(ledger, roster)
    for _ in range(2):
        attempt(ledger, run)
    ledger.finish_trial(run, 'completed')
    before = ledger.snapshot()['global']['committed']
    ledger.seal_seed_pool(run, 'd' * 64)
    with pytest.raises(ComparisonBudgetError, match='seed_pool_already_sealed'):
        ledger.seal_seed_pool(run, 'e' * 64)
    assert ledger.snapshot()['global']['committed'] == before
    ledger.close()
    with CalibrationLedger.open(ledger.path, roster=roster, manifest_identity='b' * 64,
                                now_fn=lambda: now[0]) as reopened:
        assert reopened.snapshot()['seed_pools'] == {
            run.trial_id: {'pool_run_identity': run.identity(), 'bundle_sha256': 'd' * 64}}
        assert reopened.snapshot()['global']['committed'] == before
        reopened.begin_trial(roster[1])


def test_seed_pool_seal_requires_completed_pool_and_valid_digest(ledger, roster):
    run = begin(ledger, roster)
    with pytest.raises(ComparisonBudgetError, match='seed_pool_seal_run_invalid'):
        ledger.seal_seed_pool(run, 'd' * 64)
    ledger.finish_trial(run, 'completed')
    for digest in (None, 'd' * 63, 'D' * 64, 64):
        with pytest.raises(ComparisonBudgetError, match='seed_pool_seal_digest_invalid'):
            ledger.seal_seed_pool(run, digest)
    ledger.seal_seed_pool(run, 'd' * 64)
    ledger.begin_trial(roster[1])
    ledger.finish_trial(roster[1], 'completed')
    with pytest.raises(ComparisonBudgetError, match='seed_pool_seal_run_invalid'):
        ledger.seal_seed_pool(roster[1], 'd' * 64)


@pytest.mark.parametrize('mutation', ['unknown_pool', 'wrong_identity', 'bad_digest', 'status', 'missing'])
def test_corrupt_pool_seal_is_rejected_without_rewrite(ledger, roster, now, mutation):
    run = begin(ledger, roster)
    ledger.finish_trial(run, 'completed')
    ledger.seal_seed_pool(run, 'd' * 64)
    ledger.begin_trial(roster[1])
    ledger.finish_trial(roster[1], 'completed')
    ledger.close()
    state = json.loads(ledger.path.read_text())
    seal = state['seed_pools'][run.trial_id]
    if mutation == 'unknown_pool':
        state['seed_pools']['unknown'] = state['seed_pools'].pop(run.trial_id)
    elif mutation == 'wrong_identity':
        seal['pool_run_identity'] = roster[1].identity()
    elif mutation == 'bad_digest':
        seal['bundle_sha256'] = 'invalid'
    elif mutation == 'status':
        state['trials'][run.identity()]['status'] = 'failed'
    else:
        state['seed_pools'].clear()
    ledger.path.write_text(json.dumps(state))
    before = ledger.path.read_bytes()
    with pytest.raises(ComparisonBudgetError):
        CalibrationLedger.open(ledger.path, roster=roster, manifest_identity='b' * 64,
                               now_fn=lambda: now[0])
    assert ledger.path.read_bytes() == before
