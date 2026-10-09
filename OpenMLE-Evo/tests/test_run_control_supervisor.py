"""Actual Evo + GenericLLM + provider adapter, fake final HTTP transport."""
import json
from pathlib import Path

import pytest
import operator_trace_harness as h
from test_budget_transport_wiring import FakeResponse, LITELLM_MODULE
from dojo.solvers.evo.evo import Evolutionary
from dojo.utils.logger import config_logger
from research.contracts.budget import BudgetError, BudgetLedger, CancellationToken, Cancelled
from research.search.budget_transport import TransportGuardConfig
from research.search.run_control import run_guarded_search


def test_native_operator_stop_cancels_live_jobs_and_persists_checkpoint(tmp_path, monkeypatch):
    config_logger(None)
    calls=[]
    def completion(**kwargs):
        calls.append(kwargs)
        return FakeResponse('```python\nimport pandas as pd\nSCORE = 0.5\n```',
                            {'prompt_tokens': 10, 'completion_tokens': 20})
    monkeypatch.setattr(f'{LITELLM_MODULE}.completion_fn', completion)
    cfg=h.make_solver_config(tmp_path/'solver')
    solver=Evolutionary(cfg, task_info=dict(h.TASK_INFO))
    class StopTask(h.FakeTask):
        def step_task(self, state, code):
            state,result=super().step_task(state,code)
            self.stop_requested=self.eval_count>=3
            return state,result
    task=StopTask()
    cleaned=[]
    def cancel():
        cleaned.append(True)
        return [{'job_id':'offline-only', 'cleanup_verified':True}]
    control=tmp_path/'control.json'
    with BudgetLedger.open(tmp_path/'budget.json') as ledger:
        with pytest.raises(Cancelled, match='task_stop_requested'):
            run_guarded_search(solver, task, {}, ledger=ledger, token=CancellationToken(),
                               transport_config=TransportGuardConfig(100,100,1),
                               cancel_live_jobs=cancel, checkpoint_path=control)
        assert len(calls)==3
        assert cleaned==[True]
        checkpoint=json.loads(control.read_text())
        assert checkpoint['journal_steps']==4
        assert checkpoint['budget']['committed']['requests']==3
        assert checkpoint['cleanup_error'] is None
    solver2=Evolutionary(cfg, task_info=dict(h.TASK_INFO))
    solver2.load_checkpoint()
    with BudgetLedger.open(tmp_path/'budget.json') as ledger:
        with pytest.raises(Cancelled):
            run_guarded_search(solver2,h.FakeTask(),{},ledger=ledger,
                               token=CancellationToken.from_dict(checkpoint['token']),
                               transport_config=TransportGuardConfig(100,100,1),
                               cancel_live_jobs=lambda:[],checkpoint_path=control)
        assert len(calls)==3
        assert ledger.snapshot()['committed']['requests']==3


def test_unverified_remote_cleanup_stops_run(tmp_path,monkeypatch):
    config_logger(None)
    monkeypatch.setattr(f'{LITELLM_MODULE}.completion_fn',lambda **kwargs:
        FakeResponse('```python\nimport pandas as pd\nSCORE = 0.5\n```',
                     {'prompt_tokens':10,'completion_tokens':20}))
    solver=Evolutionary(h.make_solver_config(tmp_path/'solver'),task_info=dict(h.TASK_INFO))
    with BudgetLedger.open(tmp_path/'budget.json') as ledger:
        with pytest.raises(BudgetError,match='sandbox_cleanup_unproven'):
            run_guarded_search(solver,h.FakeTask(),{},ledger=ledger,token=CancellationToken(),
                               transport_config=TransportGuardConfig(100,100,1),
                               cancel_live_jobs=lambda:[{'job_id':'offline-only','cleanup_verified':False}],
                               checkpoint_path=tmp_path/'control.json')
        assert ledger.stopped
        assert json.loads((tmp_path/'control.json').read_text())['cleanup_error']=='sandbox_cleanup_unproven'
