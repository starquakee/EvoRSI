"""Exercise real backends and Evo parsing with fake provider responses."""
import json
import time
from pathlib import Path

import pytest
import operator_trace_harness as h
from test_budget_transport_wiring import make_client, FakeResponse, LITELLM_MODULE
from dojo.core.tasks.constants import AUX_EVAL_INFO, VALIDATION_FITNESS
from research.contracts.budget import BudgetLedger, CancellationToken, Cancelled
from research.search.budget_transport import TransportGuardConfig, install_budget_guard


def test_stream_is_consumed_before_usage_settlement(tmp_path, monkeypatch):
    with BudgetLedger.open(tmp_path/'ledger.json') as ledger:
        events = []
        class Stream:
            def __iter__(self):
                for payload in [
                    {'choices': [{'delta': {'content': 'hello'}}]},
                    {'usage': {'prompt_tokens': 10, 'completion_tokens': 2}, 'choices': []},
                ]:
                    assert len(ledger.snapshot()['open_reservations']) == 1
                    events.append('chunk')
                    yield payload
            def close(self):
                events.append('closed')
        monkeypatch.setattr(f'{LITELLM_MODULE}.completion_fn', lambda **kwargs: Stream())
        client = make_client()
        install_budget_guard(client, ledger, CancellationToken(), TransportGuardConfig(100, 100, 1))
        output, usage = client.query(messages=[{'role': 'user', 'content': 'hi'}], stream=True)
        assert output == 'hello'
        assert events == ['chunk', 'chunk', 'closed']
        assert ledger.snapshot()['committed'] == {'requests': 1, 'tokens': 12}
        assert usage['usage_provenance'] == 'provider'


def test_endless_stream_hits_total_deadline_and_closes(tmp_path, monkeypatch):
    closed = []
    class Stream:
        def __iter__(self):
            while True:
                time.sleep(0.005)
                yield {'choices': [{'delta': {'content': 'x'}}]}
        def close(self):
            closed.append(True)
    monkeypatch.setattr(f'{LITELLM_MODULE}.completion_fn', lambda **kwargs: Stream())
    with BudgetLedger.open(tmp_path/'ledger.json') as ledger:
        client = make_client()
        install_budget_guard(client, ledger, CancellationToken(), TransportGuardConfig(100, 100, 0.03))
        with pytest.raises(Cancelled):
            client.query(messages=[{'role': 'user', 'content': 'hi'}], stream=True)
        for _ in range(50):
            if closed:
                break
            time.sleep(0.01)
        assert closed
        assert ledger.stopped
        assert ledger.snapshot()['committed']['requests'] == 1


def test_backend_receives_output_and_socket_limits(tmp_path, monkeypatch):
    observed = []
    def completion(**kwargs):
        observed.append(kwargs)
        return FakeResponse('ok', {'prompt_tokens': 10, 'completion_tokens': 2})
    monkeypatch.setattr(f'{LITELLM_MODULE}.completion_fn', completion)
    with BudgetLedger.open(tmp_path/'ledger.json') as ledger:
        client = make_client()
        install_budget_guard(client, ledger, CancellationToken(), TransportGuardConfig(100, 80, 0.5))
        client.query(messages=[{'role': 'user', 'content': 'hi'}], max_tokens=32768)
        assert observed[0]['max_tokens'] == 80
        assert 0 < observed[0]['request_timeout'] <= 0.5
        assert observed[0]['num_retries'] == observed[0]['max_retries'] == 0


@pytest.mark.parametrize('score,status', [(None, ''), (float('nan'), 'success'), (True, 'success'), (0.9, 'cancelled')])
def test_unscored_or_failed_evaluation_never_calls_analyze(tmp_path, monkeypatch, score, status):
    solver, llms = h.build_solver(tmp_path, monkeypatch)
    solver.create_root_node()
    node = solver._draft()
    result = h.success_result(0.5)
    result[VALIDATION_FITNESS] = score
    result[AUX_EVAL_INFO]['status'] = status
    solver.parse_eval_result(node, result)
    assert llms['analyze'].call_tracker == 0
    assert node.is_buggy
