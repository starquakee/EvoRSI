"""A durable final-phase event cannot be undone by dropping its state field."""
import json
from pathlib import Path
import pytest
from research.comparison.budget import ComparisonBudgetError, ComparisonLedger
from research.comparison.specs import iter_cells
from research.tests.test_comparison_specs import COMMIT, make_bindings


@pytest.mark.parametrize("tamper", ["set_none", "remove_field"])
def test_final_phase_cannot_be_rolled_back_while_transition_event_remains(tmp_path: Path, tamper: str) -> None:
    cells = tuple(iter_cells(make_bindings(), COMMIT))
    path = tmp_path / "ledger.json"
    with ComparisonLedger.open(path, cells=cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        ledger.enter_final_phase("a" * 64)
        assert ledger.final_phase is not None
    state = json.loads(path.read_text())
    assert any(event["type"] == "final_phase" for event in state["events"])
    if tamper == "set_none":
        state["final_phase"] = None
    else:
        del state["final_phase"]
    path.write_text(json.dumps(state))
    # Either reopening refuses the inconsistent record, or the reservation
    # remains closed based on its durable transition. Never resume search.
    with pytest.raises(ComparisonBudgetError):
        with ComparisonLedger.open(path, cells=cells, now_fn=lambda: 1000.0) as reopened:
            reopened.reserve(cells[0].inner_run(0), "main", estimated_input_tokens=1)
