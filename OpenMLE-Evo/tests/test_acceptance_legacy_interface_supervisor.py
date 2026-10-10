"""Acceptance instrumentation must preserve existing generic task/hook APIs."""
import operator_trace_harness as h

class LegacyTask(h.FakeTask):
    # Intentionally no **kwargs: this is the existing task protocol.
    def step_task(self, state, code):
        return super().step_task(state, code)

def test_uninstrumented_search_preserves_two_argument_task_and_zero_argument_hook(tmp_path, monkeypatch):
    solver, llms = h.build_solver(tmp_path, monkeypatch)
    hook_calls = []
    solver.before_operator = lambda: hook_calls.append(True)
    def evaluate(code, index):
        return h.failure_result("synthetic execution failure") if index == 1 else h.default_evaluator(code, index)
    task = LegacyTask(evaluate)
    solver.search(task, {})
    assert task.eval_count >= 7
    assert llms["debug"].call_tracker >= 1
    assert len(hook_calls) == sum(llms[name].call_tracker for name in ("draft", "debug", "improve", "crossover"))
