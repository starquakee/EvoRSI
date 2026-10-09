"""Offline contract tests for research/search/budget_transport.py (US-007).

All ledgers are temporary (tmp_path) with test-sized BudgetLimits; transports
are injected fakes. No model calls, no Evo imports.
"""
from __future__ import annotations

import pytest

from research.contracts.budget import (
    BudgetError,
    BudgetLedger,
    BudgetLimits,
    Cancelled,
    CancellationToken,
    LedgerStopped,
    ModelUsage,
)
from research.search.budget_transport import (
    TransportGuardConfig,
    extract_provider_usage,
    install_budget_guard,
    make_request_guard,
)

LIMITS = BudgetLimits(max_tokens=100_000, max_requests=50, max_elapsed_seconds=3600.0)
CONFIG = TransportGuardConfig(estimated_input_tokens=100, max_output_tokens=50)


def _ledger(tmp_path):
    return BudgetLedger.open(tmp_path / "budget.json", limits=LIMITS)


class TestTransportGuardConfig:
    def test_valid(self):
        cfg = TransportGuardConfig(estimated_input_tokens=0, max_output_tokens=1)
        assert cfg.request_deadline_seconds is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"estimated_input_tokens": -1, "max_output_tokens": 10},
            {"estimated_input_tokens": 1.5, "max_output_tokens": 10},
            {"estimated_input_tokens": True, "max_output_tokens": 10},
            {"estimated_input_tokens": 0, "max_output_tokens": 0},
            {"estimated_input_tokens": 0, "max_output_tokens": -1},
            {"estimated_input_tokens": 0, "max_output_tokens": 1.5},
            {"estimated_input_tokens": 0, "max_output_tokens": 1,
             "request_deadline_seconds": 0.0},
            {"estimated_input_tokens": 0, "max_output_tokens": 1,
             "request_deadline_seconds": -5.0},
            {"estimated_input_tokens": 0, "max_output_tokens": 1,
             "request_deadline_seconds": float("inf")},
            {"estimated_input_tokens": 0, "max_output_tokens": 1,
             "request_deadline_seconds": float("nan")},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            TransportGuardConfig(**kwargs)


class _ToDictResponse:
    """litellm-style response: .to_dict() carries the usage mapping."""

    def __init__(self, usage):
        self._usage = usage

    def to_dict(self):
        return {"id": "chatcmpl-1", "usage": self._usage}


class _AttrUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _AttrResponse:
    def __init__(self, usage):
        self.usage = usage


class TestExtractProviderUsage:
    def test_to_dict_with_usage(self):
        response = _ToDictResponse({"prompt_tokens": 11, "completion_tokens": 7})
        usage = extract_provider_usage(response)
        assert usage == ModelUsage(input_tokens=11, output_tokens=7)

    def test_dict_with_usage(self):
        usage = extract_provider_usage(
            {"usage": {"prompt_tokens": 3, "completion_tokens": 4}}
        )
        assert (usage.input_tokens, usage.output_tokens) == (3, 4)

    def test_bare_usage_dict(self):
        usage = extract_provider_usage({"prompt_tokens": 5, "completion_tokens": 6})
        assert (usage.input_tokens, usage.output_tokens) == (5, 6)

    def test_object_with_usage_attributes(self):
        response = _AttrResponse(_AttrUsage(8, 2))
        usage = extract_provider_usage(response)
        assert (usage.input_tokens, usage.output_tokens) == (8, 2)

    def test_object_with_mapping_usage_attribute(self):
        response = _AttrResponse({"prompt_tokens": 1, "completion_tokens": 9})
        usage = extract_provider_usage(response)
        assert (usage.input_tokens, usage.output_tokens) == (1, 9)

    def test_zero_usage_is_valid(self):
        usage = extract_provider_usage({"prompt_tokens": 0, "completion_tokens": 0})
        assert usage.total == 0

    @pytest.mark.parametrize(
        "response",
        [
            {},  # no usage at all
            {"id": "x"},  # mapping without usage keys
            {"usage": {}},  # both fields missing
            {"usage": {"prompt_tokens": 3}},  # one field missing
            {"usage": {"completion_tokens": 3}},  # other field missing
            {"usage": {"prompt_tokens": "abc", "completion_tokens": 1}},  # non-numeric
            {"usage": {"prompt_tokens": None, "completion_tokens": 1}},  # None
            {"usage": {"prompt_tokens": -1, "completion_tokens": 1}},  # negative
            {"usage": {"prompt_tokens": 1, "completion_tokens": -2}},  # negative
            {"usage": {"prompt_tokens": float("inf"), "completion_tokens": 1}},
            {"usage": {"prompt_tokens": 1, "completion_tokens": float("nan")}},
            {"usage": {"prompt_tokens": True, "completion_tokens": 1}},  # bool
            {"usage": None},  # explicit None usage
            _AttrResponse(None),  # None .usage attribute
            _AttrResponse(_AttrUsage(1, None)),  # attribute None field
            _ToDictResponse({}),  # to_dict without usage fields
            object(),  # no usage anywhere
        ]
        + [[], 42, "usage"],  # non-mapping scalars
    )
    def test_unavailable_usage_fails_closed(self, response):
        with pytest.raises(BudgetError, match="usage_unavailable"):
            extract_provider_usage(response)


class _SpyTransport:
    def __init__(self, ledger, response):
        self._ledger = ledger
        self._response = response
        self.reserved_at_call = None
        self.calls = 0

    def __call__(self):
        self.calls += 1
        snapshot = self._ledger.snapshot()
        self.reserved_at_call = len(snapshot["open_reservations"])
        return self._response


class TestMakeRequestGuard:
    def test_reserve_before_transport_and_commit_provider_usage(self, tmp_path):
        with _ledger(tmp_path) as ledger:
            guard = make_request_guard(ledger, CancellationToken(), CONFIG)
            transport = _SpyTransport(
                ledger, {"usage": {"prompt_tokens": 12, "completion_tokens": 8}}
            )
            response = guard(transport)
            assert response == {"usage": {"prompt_tokens": 12, "completion_tokens": 8}}
            # the reservation existed BEFORE the transport ran
            assert transport.reserved_at_call == 1
            snapshot = ledger.snapshot()
            assert snapshot["committed"] == {"requests": 1, "tokens": 20}
            assert snapshot["open_reservations"] == []
            assert snapshot["stopped"] is None

    def test_usage_failure_retains_reservation_and_stops(self, tmp_path):
        with _ledger(tmp_path) as ledger:
            guard = make_request_guard(ledger, CancellationToken(), CONFIG)
            with pytest.raises(BudgetError, match="usage_unavailable"):
                guard(lambda: {"no": "usage"})
            snapshot = ledger.snapshot()
            # reservation retained as permanently charged: est 100 + max 50
            assert snapshot["committed"] == {"requests": 1, "tokens": 150}
            assert snapshot["open_reservations"] == []
            assert snapshot["stopped"] is not None
            with pytest.raises(LedgerStopped):
                ledger.reserve(1, 1)

    def test_transport_exception_retains_and_stops_then_reraises(self, tmp_path):
        with _ledger(tmp_path) as ledger:
            guard = make_request_guard(ledger, CancellationToken(), CONFIG)

            def boom():
                raise RuntimeError("provider down")

            with pytest.raises(RuntimeError, match="provider down"):
                guard(boom)
            snapshot = ledger.snapshot()
            assert snapshot["committed"] == {"requests": 1, "tokens": 150}
            assert snapshot["stopped"] is not None
            with pytest.raises(LedgerStopped):
                guard(lambda: {"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    def test_cancelled_token_propagates_without_transport(self, tmp_path):
        with _ledger(tmp_path) as ledger:
            token = CancellationToken()
            token.stop("operator_halt")
            guard = make_request_guard(ledger, token, CONFIG)
            transport = _SpyTransport(ledger, {"usage": {"prompt_tokens": 1, "completion_tokens": 1}})
            with pytest.raises(Cancelled):
                guard(transport)
            assert transport.calls == 0
            assert ledger.snapshot()["committed"] == {"requests": 0, "tokens": 0}

    def test_deadline_token_cancels(self, tmp_path):
        with _ledger(tmp_path) as ledger:
            token = CancellationToken(deadline_epoch=1.0)  # long past
            guard = make_request_guard(ledger, token, CONFIG)
            with pytest.raises(Cancelled, match="deadline_exceeded"):
                guard(lambda: {"usage": {"prompt_tokens": 1, "completion_tokens": 1}})


class _FakeSdk:
    def __init__(self):
        self.max_retries = 2


class _FakeClient:
    def __init__(self, with_sdk=True):
        if with_sdk:
            self.client = _FakeSdk()


class TestInstallBudgetGuard:
    def test_installs_guard_and_zeroes_sdk_retries(self, tmp_path):
        with _ledger(tmp_path) as ledger:
            client = _FakeClient()
            returned = install_budget_guard(client, ledger, CancellationToken(), CONFIG)
            assert returned is client
            assert callable(client.request_guard)
            assert client.client.max_retries == 0

    def test_guard_invokes_thunk_exactly_once(self, tmp_path):
        with _ledger(tmp_path) as ledger:
            client = _FakeClient()
            install_budget_guard(client, ledger, CancellationToken(), CONFIG)
            calls = []

            def thunk():
                calls.append(1)
                return {"usage": {"prompt_tokens": 2, "completion_tokens": 3}}

            client.request_guard(thunk)
            client.request_guard(thunk)
            assert calls == [1, 1]
            assert ledger.snapshot()["committed"] == {"requests": 2, "tokens": 10}

    def test_second_install_fails_closed(self, tmp_path):
        with _ledger(tmp_path) as ledger:
            client = _FakeClient()
            install_budget_guard(client, ledger, CancellationToken(), CONFIG)
            with pytest.raises(BudgetError, match="guard_already_installed"):
                install_budget_guard(client, ledger, CancellationToken(), CONFIG)

    def test_no_nesting_guard_on_guarded_client(self, tmp_path):
        # Installing on a client whose request_guard is already a guard must
        # fail closed exactly like a second install: guards never nest.
        with _ledger(tmp_path) as ledger:
            client = _FakeClient()
            client.request_guard = make_request_guard(
                ledger, CancellationToken(), CONFIG
            )
            with pytest.raises(BudgetError, match="guard_already_installed"):
                install_budget_guard(client, ledger, CancellationToken(), CONFIG)

    def test_client_without_sdk_is_fine(self, tmp_path):
        with _ledger(tmp_path) as ledger:
            client = _FakeClient(with_sdk=False)
            install_budget_guard(client, ledger, CancellationToken(), CONFIG)
            assert callable(client.request_guard)
