"""Original acceptance deadline persists across child-process reopen."""
from research.contracts.budget import BudgetLedger, BudgetLimits


def test_reopen_preserves_original_deadline_and_lower_limit_only_shortens(tmp_path):
    clock = [1000.0]
    path = tmp_path / 'ledger.json'
    with BudgetLedger.open(path, limits=BudgetLimits(max_elapsed_seconds=60), now_fn=lambda: clock[0]) as ledger:
        assert ledger.deadline_epoch == 1060.0
    clock[0] = 1020.0
    with BudgetLedger.open(path, now_fn=lambda: clock[0]) as ledger:
        assert ledger.deadline_epoch == 1060.0
        assert ledger.snapshot()['remaining']['elapsed_seconds'] == 40.0
    with BudgetLedger.open(path, limits=BudgetLimits(max_elapsed_seconds=40), now_fn=lambda: clock[0]) as ledger:
        assert ledger.deadline_epoch == 1040.0
    clock[0] = 1030.0
    with BudgetLedger.open(path, now_fn=lambda: clock[0]) as ledger:
        assert ledger.deadline_epoch == 1040.0
        assert ledger.snapshot()['remaining']['elapsed_seconds'] == 10.0
