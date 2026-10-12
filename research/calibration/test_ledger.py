"""Synthetic regression tests only: no credentials, model, or sandbox access."""
import json
from pathlib import Path

import pytest

from research.calibration.ledger import CalibrationLedger
from research.calibration.specs import CalibrationError, RunSpec, protocol, selected_plan, trials
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


def attempt(ledger, run, *, tokens=10, kind='main', input_estimate=100):
    return guarded_comparison_attempt(ledger, run, kind, lambda _: 'response', None,
            estimated_input_tokens=input_estimate,
            usage_fn=lambda _: ProviderUsage(tokens, 0), debug_limit=2 if kind == 'debug' else None)


def test_selected_roster_and_limits_are_independent_of_old_comparison(roster):
    assert len(roster) == 27
    assert len({trial.identity() for trial in roster}) == 27
    assert {trial.task for trial in roster} == {'spambase'}
    assert protocol()['global_limits'] == {'max_requests': 216, 'max_tokens': 600000, 'max_elapsed_seconds': 14400}
    spec = RunSpec(roster[0], 'b' * 64)
    assert 'calibration-round2' in spec.run_dir
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
    assert snapshot['cells'][run.trial_id]['remaining']['tokens'] == 39950


def test_six_main_two_debug_ceiling(ledger, roster):
    run = begin(ledger, roster)
    for _ in range(6):
        attempt(ledger, run)
    with pytest.raises(InnerRequestLimitReached):
        attempt(ledger, run)
    for _ in range(2):
        attempt(ledger, run, kind='debug')
    with pytest.raises(InnerRequestLimitReached):
        attempt(ledger, run, kind='debug')
    assert ledger.snapshot()['global']['committed'] == {'requests': 8, 'tokens': 80}


def test_trial_conservative_token_exhaustion_does_not_borrow(ledger, roster):
    run = begin(ledger, roster)
    attempt(ledger, run, tokens=32000, input_estimate=23808)
    with pytest.raises(CellBudgetExhausted):
        attempt(ledger, run)
    assert not ledger.stopped
    ledger.finish_trial(run, 'terminated')
    ledger.begin_trial(roster[1])
    attempt(ledger, roster[1])
    assert ledger.snapshot()['global']['committed']['requests'] == 2


def test_global_token_ceiling_stops_without_sending_extra_request(ledger, roster):
    ledger.start_clock()
    for run in roster[:15]:
        ledger.begin_trial(run)
        attempt(ledger, run, tokens=39000, input_estimate=30808)
        ledger.finish_trial(run, 'completed')
    ledger.begin_trial(roster[15])
    with pytest.raises(GlobalBudgetExhausted):
        attempt(ledger, roster[15], tokens=39000, input_estimate=30808)
    assert ledger.stopped
    assert ledger.snapshot()['global']['committed'] == {'requests': 15, 'tokens': 585000}


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


@pytest.mark.parametrize('elapsed', [1800, 14400, -1])
def test_deadline_or_backwards_clock_refuses_before_transport(ledger, roster, now, elapsed):
    run = begin(ledger, roster)
    now[0] += elapsed
    with pytest.raises(GlobalBudgetExhausted):
        attempt(ledger, run)
    assert ledger.snapshot()['global']['committed']['requests'] == 0


@pytest.mark.parametrize('mutation', ['limits', 'manifest', 'usage'])
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
    else:
        state['requests'][0]['charged_tokens'] += 1
    ledger.path.write_text(json.dumps(state))
    before = ledger.path.read_bytes()
    with pytest.raises(ComparisonBudgetError):
        CalibrationLedger.open(ledger.path, roster=roster, manifest_identity='b' * 64, now_fn=lambda: now[0])
    assert ledger.path.read_bytes() == before
