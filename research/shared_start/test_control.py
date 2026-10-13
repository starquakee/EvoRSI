from types import SimpleNamespace

import pytest

from research.shared_start.control import SharedStartBudgetControl
from research.comparison.budget import InnerRequestLimitReached, GlobalBudgetExhausted


@pytest.mark.parametrize('target', [2, 4])
def test_floor_counts_only_new_mains(tmp_path, target):
    state = {'cells': {'unit': {'remaining': {'requests': target + 2}}},
             'global': {'remaining': {'requests': target + 2}}}
    run = SimpleNamespace(main_target=target, cell=SimpleNamespace(cell_key=lambda: 'unit'),
                          run_key=lambda: 'unit')
    ledger = SimpleNamespace(snapshot=lambda: state)
    control = SharedStartBudgetControl(ledger, run, debug_budget=2,
                                      control_log_path=tmp_path / 'log')
    assert control.main_needed() == target
    for _ in range(target):
        control.require_main('improve')
        state['cells']['unit']['remaining']['requests'] -= 1
        state['global']['remaining']['requests'] -= 1
    assert control.main_needed() == 0
    with pytest.raises(InnerRequestLimitReached):
        control.require_main('improve')
    control.note_debug_attempt()
    control.note_debug_attempt()
    assert control.debug_denial_reason().startswith('inner_debug_budget_exhausted')


def test_debug_cannot_consume_reserved_continuation_mains(tmp_path):
    state = {'cells': {'unit': {'remaining': {'requests': 4}}},
             'global': {'remaining': {'requests': 4}}}
    run = SimpleNamespace(main_target=4, cell=SimpleNamespace(cell_key=lambda: 'unit'),
                          run_key=lambda: 'unit')
    control = SharedStartBudgetControl(SimpleNamespace(snapshot=lambda: state), run,
        debug_budget=2, control_log_path=tmp_path / 'log')
    assert control.debug_denial_reason().startswith('cell_budget_insufficient')
    state['global']['remaining']['requests'] = 3
    with pytest.raises(GlobalBudgetExhausted):
        control.require_main('crossover')
