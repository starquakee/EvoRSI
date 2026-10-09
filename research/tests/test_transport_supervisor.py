"""Provider usage and hard-deadline regressions; no network/model calls."""
import threading
import time

import pytest

from research.contracts.budget import BudgetError, BudgetLedger, Cancelled, CancellationToken
from research.search.budget_transport import TransportGuardConfig, extract_provider_usage, make_request_guard


@pytest.mark.parametrize('usage', [
    {'prompt_tokens': -0.5, 'completion_tokens': 2},
    {'prompt_tokens': 1.5, 'completion_tokens': 2},
    {'prompt_tokens': 1.0, 'completion_tokens': 2},
    {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 90},
    {'prompt_tokens': 1, 'completion_tokens': 2, 'usage_provenance': 'estimated'},
])
def test_malformed_or_estimated_usage_never_becomes_actual(usage):
    with pytest.raises(BudgetError, match='usage_unavailable'):
        extract_provider_usage({'usage': usage})


def test_hard_deadline_releases_controller_and_retains_unknown_cost(tmp_path):
    release = threading.Event()
    started = threading.Event()
    with BudgetLedger.open(tmp_path/'ledger.json') as ledger:
        token = CancellationToken()
        guard = make_request_guard(ledger, token, TransportGuardConfig(100, 50, 0.03))
        def blocked():
            started.set()
            release.wait(2)
            return {'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}
        before = time.monotonic()
        try:
            with pytest.raises(Cancelled):
                guard(blocked)
            assert started.is_set()
            assert time.monotonic() - before < 0.8
            assert ledger.snapshot()['committed'] == {'requests': 1, 'tokens': 150}
            assert ledger.stopped
            with pytest.raises(Cancelled):
                guard(blocked)
        finally:
            release.set()


def test_large_prompt_expands_reservation_and_caps_actual_output(tmp_path):
    with BudgetLedger.open(tmp_path/'ledger.json') as ledger:
        guard = make_request_guard(ledger, CancellationToken(), TransportGuardConfig(10, 50, 0.5))
        kwargs = {'max_tokens': 99999}
        seen = {}
        def send():
            seen.update(ledger.snapshot()['open_reservations'][0])
            assert kwargs['max_tokens'] == 50
            assert 0 < kwargs['request_timeout'] <= 0.5
            return {'usage': {'prompt_tokens': 200, 'completion_tokens': 20}}
        guard.run(send, messages=[{'role': 'user', 'content': '中'*500}], model_kwargs=kwargs)
        assert seen['tokens'] > 1500
        assert ledger.snapshot()['committed']['tokens'] == 220


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True])
def test_nonfinite_search_input_rejected(value):
    from research.search.search_vector import decode_retained
    with pytest.raises(ValueError):
        decode_retained([value, 1, 1, 1, 1])
