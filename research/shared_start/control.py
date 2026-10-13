"""Per-unit request floor; pool initialization and continuation are separate units."""
from research.comparison.control import ComparisonMainBudgetControl, run_comparison_search
from research.comparison.budget import InnerRequestLimitReached


class SharedStartBudgetControl(ComparisonMainBudgetControl):
    def __init__(self, ledger, run, *, debug_budget, control_log_path):
        super().__init__(ledger, run, debug_budget=debug_budget,
                         control_log_path=control_log_path)
        self._main_target = run.main_target
        if self._main_target not in (2, 4):
            raise ValueError('shared_start_main_target_invalid')

    def main_needed(self):
        return self._main_target - self._main_done

    def require_main(self, operator):
        if self.main_needed() <= 0:
            raise InnerRequestLimitReached('shared_start_main_limit')
        return super().require_main(operator)
