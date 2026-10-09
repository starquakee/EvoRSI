"""Malformed usage and persisted accounting must never buy budget."""
import json
import pytest
from research.contracts.budget import (
    BudgetLedger, BudgetLimits, BudgetError, ModelUsage, CancellationToken,
    guarded_attempt,
)

@pytest.mark.parametrize("value", [True, 1.5, float("nan"), float("inf"), -1])
def test_invalid_counts(value, tmp_path):
    with pytest.raises(ValueError):
        ModelUsage(value, 1)
    with pytest.raises(ValueError):
        BudgetLimits(value, 30, 5400)
    with BudgetLedger.open(tmp_path / "budget.json") as ledger:
        with pytest.raises(ValueError):
            ledger.reserve(value, 10)

@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1])
def test_invalid_elapsed_limits(value):
    with pytest.raises(ValueError):
        BudgetLimits(200000, 30, value)

@pytest.mark.parametrize("field,value", [
    ("tokens", -1000000), ("requests", -10), ("tokens", True),
    ("requests", 1.5),
])
def test_corrupt_committed_fails_closed(field, value, tmp_path):
    path = tmp_path / "budget.json"
    with BudgetLedger.open(path):
        pass
    state = json.loads(path.read_text())
    state["committed"][field] = value
    path.write_text(json.dumps(state))
    with pytest.raises(BudgetError, match="corrupt_ledger"):
        BudgetLedger.open(path)

@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_corrupt_anchor_fails_closed(value, tmp_path):
    path = tmp_path / "budget.json"
    with BudgetLedger.open(path):
        pass
    state = json.loads(path.read_text())
    state["started_at"] = value
    path.write_text(json.dumps(state))
    with pytest.raises(BudgetError, match="corrupt_ledger"):
        BudgetLedger.open(path)

def test_lowered_limits_survive_reopen(tmp_path):
    path = tmp_path / "budget.json"
    with BudgetLedger.open(path):
        pass
    with BudgetLedger.open(path, limits=BudgetLimits(100, 2, 60)):
        pass
    with BudgetLedger.open(path) as ledger:
        assert ledger.snapshot()["remaining"]["tokens"] == 100
    with pytest.raises(BudgetError, match="limits_cannot_expand"):
        BudgetLedger.open(path, limits=BudgetLimits(200, 4, 120))

def test_malformed_usage_retains_and_stops(tmp_path):
    with BudgetLedger.open(tmp_path / "budget.json") as ledger:
        with pytest.raises(ValueError, match="invalid_usage_payload"):
            guarded_attempt(
                ledger, CancellationToken(), lambda request: {}, {},
                estimated_input_tokens=100, max_output_tokens=50,
                usage_fn=lambda response: {"tokens": 1},
            )
        snapshot = ledger.snapshot()
        assert snapshot["stopped"] is not None
        assert snapshot["committed"] == {"requests": 1, "tokens": 150}

def test_clock_rollback_prevents_new_call(tmp_path):
    clock = [100.0]
    with BudgetLedger.open(tmp_path / "budget.json", now_fn=lambda: clock[0]) as ledger:
        clock[0] = 99.0
        with pytest.raises(BudgetError, match="clock"):
            ledger.reserve(1, 1)
