"""Independent real comparison guard regressions; disposable offline ledgers."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pytest
from research.comparison.budget import ComparisonBudgetError, ComparisonLedger
from research.comparison.protocol import ModelProfile
from research.comparison.transport import ComparisonRequestGuard, comparison_request_profile, install_comparison_guard
from research.contracts.budget import CancellationToken
from research.tests.test_comparison_budget_supervisor import roster

MESSAGES = [{"role": "system", "content": "offline-only"}, {"role": "user", "content": "fixture"}]

def profile() -> dict[str, Any]:
    return {**comparison_request_profile(), "base_url": ModelProfile().base_url,
            "allowed_openai_params": ["reasoning_effort"], "max_retries": 0, "num_retries": 0}

def response() -> dict[str, Any]:
    return {"usage": {"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26,
                       "usage_provenance": "provider", "provider_request_id": "provider-offline-1"}}

def test_bound_run_method_and_single_operator_sink(tmp_path: Path) -> None:
    cells = roster()
    observed: list[Any] = []
    with ComparisonLedger.open(tmp_path/"ledger.json", cells=cells) as ledger:
        ledger.start_clock()
        client = SimpleNamespace(request_guard=None)
        def sink(*args: Any, **kwargs: Any) -> None:
            observed.append((args, kwargs))
        install_comparison_guard(client, ledger, CancellationToken(), cells[0].inner_run(0),
                                 operator="draft", debug_budget=2, attempt_sink=sink)
        assert callable(client.request_guard.run), "run spec must not shadow the bound run() method"
        client.request_guard.run(response, messages=MESSAGES, model_kwargs=profile())
        assert len(observed) == 1, "operator sink must call the original observer exactly once"
        assert observed[0][1] == {"operator": "draft"}
        assert ledger.snapshot()["global"]["committed"] == {"requests": 1, "tokens": 26}

@pytest.mark.parametrize("field,bad", [
    ("base_url", "https://example.invalid"),
    ("allowed_openai_params", []),
    ("max_retries", 1), ("num_retries", 1),
    ("stream", 1), ("temperature", True), ("max_tokens", 8192.0),
    ("max_completion_tokens", 1), ("n", True),
])
def test_full_profile_drift_refused_before_transport(tmp_path: Path, field: str, bad: Any) -> None:
    cells = roster()
    called: list[bool] = []
    def transport() -> dict[str, Any]:
        called.append(True)
        return response()
    with ComparisonLedger.open(tmp_path/"ledger.json", cells=cells) as ledger:
        ledger.start_clock()
        guard = ComparisonRequestGuard(ledger, CancellationToken(), cells[0].inner_run(0), kind="main")
        params = profile()
        params[field] = bad
        with pytest.raises(ComparisonBudgetError):
            guard.run(transport, messages=MESSAGES, model_kwargs=params)
        assert not called
        assert ledger.snapshot()["global"]["committed"]["requests"] == 0
        assert not ledger.snapshot()["reservations"]

@pytest.mark.parametrize("bad", ["", 17])
def test_malformed_reported_provider_id_stops_after_attempt(tmp_path: Path, bad: Any) -> None:
    cells = roster()
    with ComparisonLedger.open(tmp_path/"ledger.json", cells=cells) as ledger:
        ledger.start_clock()
        guard = ComparisonRequestGuard(ledger, CancellationToken(), cells[0].inner_run(0), kind="main")
        reply = response()
        reply["usage"]["provider_request_id"] = bad
        with pytest.raises(ComparisonBudgetError):
            guard.run(lambda: reply, messages=MESSAGES, model_kwargs=profile())
        assert ledger.stopped, "malformed provider identity must not be silently erased to None"


def test_late_provider_completion_cannot_reopen_budget(tmp_path: Path) -> None:
    import threading
    from research.contracts.budget import Cancelled
    cells = roster()
    release, finished = threading.Event(), threading.Event()
    with ComparisonLedger.open(tmp_path/"ledger.json", cells=cells) as ledger:
        ledger.start_clock()
        token = CancellationToken(deadline_epoch=ledger.deadline_epoch)
        guard = ComparisonRequestGuard(ledger, token, cells[0].inner_run(0),
                                       kind="main", request_deadline_seconds=0.02)
        def delayed() -> dict[str, Any]:
            release.wait(2)
            finished.set()
            return response()
        try:
            with pytest.raises(Cancelled):
                guard.run(delayed, messages=MESSAGES, model_kwargs=profile())
            before = ledger.snapshot()
            assert before["stopped"] is not None
            assert len(before["requests"]) == 1
            assert before["requests"][0]["usage_known"] is False
            assert before["requests"][0]["charged_tokens"] == before["requests"][0]["reserved_tokens"]
        finally:
            release.set()
            assert finished.wait(2)
        assert ledger.snapshot()["global"]["committed"] == before["global"]["committed"]
        next_calls: list[bool] = []
        def next_request() -> dict[str, Any]:
            next_calls.append(True)
            return response()
        with pytest.raises((Cancelled, ComparisonBudgetError)):
            guard.run(next_request, messages=MESSAGES, model_kwargs=profile())
        assert next_calls == []
