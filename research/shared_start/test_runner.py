"""Real Evo assembly with fake generation/evaluation; never calls a provider."""
from pathlib import Path
from types import SimpleNamespace
import os

import pytest

from research.shared_start.control import SharedStartBudgetControl
from research.shared_start.seeds import initialize_continuation, restore_journal, validate_seed_nodes
from research.shared_start.specs import RunSpec, trials


@pytest.mark.parametrize('seed_scores', [(.9, .8), (.8, .9), (.8, .8)])
@pytest.mark.parametrize('config_id', ['c00', 'c04', 'c08'])
def test_real_evo_repaired_pool_then_four_new_main_calls(tmp_path, monkeypatch, seed_scores, config_id):
    pytest.importorskip('dojo')
    from dojo.solvers.evo import evo
    from dojo.core.interpreters.base import ExecutionResult
    from dojo.utils.logger import config_logger
    from research.comparison.runner import build_comparison_solver

    config_logger(None)
    repo = Path(os.environ.get('SHARED_START_TEST_REPO', Path(__file__).resolve().parents[2]))
    roster = trials(repo, 'a' * 40)
    pool_trial = next(t for t in roster if t.kind == 'fresh_seed_pool')
    cont_trial = next(t for t in roster if t.kind == 'continuation' and t.pool_id == pool_trial.pool_id
                      and t.record['config_id'] == config_id)

    def solver_for(trial, directory):
        base = RunSpec(trial, 'b' * 64)
        spec = SimpleNamespace(run=trial, decoded=base.decoded, run_dir=str(directory),
            job_timeout_seconds=base.job_timeout_seconds, validate=base.validate)
        return build_comparison_solver(spec, repo_root=repo)

    generated = []
    def fake_generation(*args, **kwargs):
        generated.append(len(generated) + 1)
        return 'fake plan', 'x = ' + str(generated[-1]), {'usage': {'total_tokens': 1}}
    monkeypatch.setattr(evo, 'execute_op_plan_code', fake_generation)

    class FakeTask:
        count = 0
        def step_task(self, state, code, **kwargs):
            self.count += 1
            bad = self.count == 1
            execution = ExecutionResult.get_empty()
            execution.exit_code = 1 if bad else 0
            execution.exec_time = .01
            score = seed_scores[self.count - 2] if self.count in (2, 3) else .8
            return state, {
                evo.EXECUTION_OUTPUT: execution,
                evo.VALIDATION_FITNESS: None if bad else score,
                evo.VALID_SOLUTION: not bad,
                evo.AUX_EVAL_INFO: {'job_id': 'offline-' + str(self.count),
                    'status': 'failed' if bad else 'completed', 'score': None if bad else score},
            }

    pool = solver_for(pool_trial, tmp_path / 'pool')
    pool.cfg.num_generations = 1
    pool_trace = []
    pool.operator_tracer = pool_trace.append
    pool.search(FakeTask(), {})
    nodes = pool.journal.node_list()
    terminals = validate_seed_nodes(nodes, pool_trace)
    assert len(generated) == 3
    assert [e['operator'] for e in pool_trace] == ['draft', 'debug', 'draft']
    assert len(terminals) == 2

    cont_dir = tmp_path / 'continuation'
    cont_dir.mkdir(exist_ok=True)
    continuation = solver_for(cont_trial, cont_dir)
    original_seed_method = continuation._seed_solution_database_from_journal
    initialize_continuation(continuation, {'nodes': nodes, 'prediction_artifacts': {},
        'receipt': {'terminal_node_ids': terminals},
        'directory': tmp_path / 'pool'}, cont_dir)
    assert continuation.state.current_generation == 1
    assert continuation.journal.nodes[2].parents[0] is continuation.journal.nodes[1]
    assert continuation.journal.nodes[2].metric.info == pool.journal.nodes[2].metric.info
    snapshot = {'cells': {cont_trial.cell_key(): {'remaining': {'requests': 6}}},
                'global': {'remaining': {'requests': 6}}}
    control = SharedStartBudgetControl(SimpleNamespace(snapshot=lambda: snapshot), cont_trial,
        debug_budget=2, control_log_path=tmp_path / 'budget.jsonl')
    continuation.budget_control = control
    cont_trace = []
    def traced(event):
        cont_trace.append(event)
        snapshot['cells'][cont_trial.cell_key()]['remaining']['requests'] -= 1
        snapshot['global']['remaining']['requests'] -= 1
        if event['operator'] == 'debug':
            control.note_debug_attempt()
    continuation.operator_tracer = traced
    continuation.search(FakeTask(), {})
    mains = [e for e in cont_trace if e['operator'] != 'debug']
    assert len(mains) == 4
    assert [e['generation_id'] for e in mains] == [1, 1, 2, 2]
    assert all(e['operator'] in ('improve', 'crossover') for e in mains)
    assert len(generated) == 8  # 3 original pool calls + 4 mains + 1 new repair
    assert continuation.state.current_generation == 3
    assert control.main_needed() == 0
    assert continuation._seed_solution_database_from_journal == original_seed_method
    selection = mains[0]['selection']
    assert set(c['node_id'] for c in selection['candidates']) == set(terminals)
    assert set(mains[0]['parent_node_ids']) <= set(terminals)
    restored_rows = continuation.journal.node_list()[:len(nodes)]
    for before, after in zip(nodes, restored_rows):
        assert {k: v for k, v in before.items() if k != 'children'} == {
            k: v for k, v in after.items() if k != 'children'}
