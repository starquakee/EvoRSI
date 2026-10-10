"""Acceptance transport-profile enforcement tests (offline, fake provider).

Every actual operator request must carry the exact reviewed profile (model,
stream, max_tokens, temperature, top_p, reasoning_effort); off-profile
requests fail closed BEFORE any reservation, and a provider-side parameter
rejection stops the ledger with the reservation retained — no hidden retry
or fallback.
"""
from __future__ import annotations

import pytest

from research.contracts.budget import (
    BudgetError,
    BudgetLedger,
    BudgetLimits,
    CancellationToken,
)
from research.search.acceptance_inner import acceptance_request_profile
from research.search.budget_transport import RequestGuard, TransportGuardConfig


def _limits():
    return BudgetLimits(max_tokens=100_000, max_requests=10, max_elapsed_seconds=600.0)


def _guard(ledger, profile=None):
    return RequestGuard(
        ledger,
        CancellationToken(),
        TransportGuardConfig(
            estimated_input_tokens=100,
            max_output_tokens=8192,
            required_request_profile=acceptance_request_profile() if profile is None else profile,
        ),
    )


def test_on_profile_request_passes_and_records_usage(tmp_path):
    calls = []

    def provider():
        calls.append(True)
        return {"usage": {"prompt_tokens": 5, "completion_tokens": 7}}

    with BudgetLedger.open(tmp_path / "ledger.json", limits=_limits()) as ledger:
        _guard(ledger).run(
            provider,
            messages=[{"role": "user", "content": "x"}],
            model_kwargs=dict(acceptance_request_profile()),
        )
        snapshot = ledger.snapshot()
    assert calls == [True]
    assert snapshot["committed"]["tokens"] == 12
    assert snapshot["stopped"] is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("model", "openai/k3-256k"),
        ("model", "k3"),
        ("stream", False),
        ("max_tokens", 4096),
        ("temperature", 0.6),
        ("top_p", 0.5),
        ("reasoning_effort", "high"),
        ("reasoning_effort", None),
    ],
)
def test_off_profile_request_fails_before_reservation(tmp_path, key, value):
    calls = []

    def provider():  # pragma: no cover - must never run
        calls.append(True)
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    kwargs = dict(acceptance_request_profile())
    if value is None:
        kwargs.pop(key)
    else:
        kwargs[key] = value
    with BudgetLedger.open(tmp_path / "ledger.json", limits=_limits()) as ledger:
        with pytest.raises(BudgetError, match=f"request_profile_mismatch:{key}"):
            _guard(ledger).run(
                provider,
                messages=[{"role": "user", "content": "x"}],
                model_kwargs=kwargs,
            )
        snapshot = ledger.snapshot()
    assert calls == []
    assert snapshot["committed"]["requests"] == 0
    assert snapshot["open_reservations"] == []
    assert snapshot["stopped"] is None  # config error, not an unknown-usage event


def test_provider_parameter_rejection_stops_without_retry(tmp_path):
    """A provider rejection (e.g. HTTP 400 for a parameter) is an unknown
    usage outcome: exactly ONE attempt, reservation retained, ledger stopped,
    no hidden fallback or retry."""
    calls = []

    class SimulatedBadRequest(Exception):
        pass

    def provider():
        calls.append(True)
        raise SimulatedBadRequest("400: unsupported request parameter")

    with BudgetLedger.open(tmp_path / "ledger.json", limits=_limits()) as ledger:
        with pytest.raises(SimulatedBadRequest):
            _guard(ledger).run(
                provider,
                messages=[{"role": "user", "content": "x"}],
                model_kwargs=dict(acceptance_request_profile()),
            )
        snapshot = ledger.snapshot()
    assert len(calls) == 1
    assert snapshot["stopped"] is not None
    # The unknown-usage reservation is retained as permanently charged.
    assert snapshot["committed"]["requests"] == 1
    assert snapshot["committed"]["tokens"] > 0
    assert snapshot["open_reservations"] == []
    with pytest.raises(BudgetError):
        ledger.reserve(1, 1)  # no further request is possible


def test_profile_pins_match_reviewed_acceptance_values():
    profile = acceptance_request_profile()
    assert profile == {
        "model": "openai/k3",
        "stream": True,
        "max_tokens": 8192,
        "temperature": 1.0,
        "top_p": 0.95,
        "reasoning_effort": "low",
    }
