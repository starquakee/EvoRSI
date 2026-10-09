"""Supervisor regression cases: no model requests, disposable ledgers only."""
from pathlib import Path
import pytest
from research.contracts.budget import (
    BudgetLedger, BudgetLimits, BudgetError, CancellationToken, guarded_attempt,
)

def test_restart_with_unresolved_usage_stops(tmp_path: Path) -> None:
    path = tmp_path / 'ledger.json'
    with BudgetLedger.open(path) as ledger:
        ledger.reserve(10, 20)
    with BudgetLedger.open(path) as ledger:
        with pytest.raises(BudgetError):
            ledger.reserve(10, 20)

def test_usage_parser_exception_stops(tmp_path: Path) -> None:
    def malformed(response: object) -> None:
        raise ValueError('invalid usage')
    with BudgetLedger.open(tmp_path / 'ledger.json') as ledger:
        with pytest.raises(ValueError):
            guarded_attempt(ledger, CancellationToken(), lambda request: {}, None,
                            estimated_input_tokens=10, max_output_tokens=20,
                            usage_fn=malformed)
        with pytest.raises(BudgetError):
            ledger.reserve(10, 20)

def test_restart_cannot_expand_limits(tmp_path: Path) -> None:
    path = tmp_path / 'ledger.json'
    with BudgetLedger.open(path, limits=BudgetLimits(100, 2, 60)):
        pass
    try:
        with BudgetLedger.open(path, limits=BudgetLimits(200, 4, 120)) as ledger:
            remaining = ledger.snapshot()['remaining']
            assert remaining['tokens'] <= 100
            assert remaining['requests'] <= 2
            assert remaining['elapsed_seconds'] <= 60
    except BudgetError:
        pass

def test_snapshot_does_not_mutate_live_accounting(tmp_path: Path) -> None:
    with BudgetLedger.open(tmp_path / 'ledger.json') as ledger:
        ledger.reserve(10, 20)
        snapshot = ledger.snapshot()
        snapshot['open_reservations'][0]['tokens'] = 0
        assert ledger.snapshot()['open_reservations'][0]['tokens'] == 30

def test_closed_ledger_cannot_write_without_lock(tmp_path: Path) -> None:
    ledger = BudgetLedger.open(tmp_path / 'ledger.json')
    ledger.close()
    with pytest.raises(BudgetError):
        ledger.reserve(10, 20)
