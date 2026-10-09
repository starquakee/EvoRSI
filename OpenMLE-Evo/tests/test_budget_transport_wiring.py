"""US-007 budget-transport wiring tests (offline; litellm completion faked).

Uses the REAL ``LiteLLMClient`` with ``completion_fn`` monkeypatched, the
REAL ``research.contracts.budget.BudgetLedger`` on a tmp path and the REAL
``research.search.budget_transport`` guard interface. No model/API calls.

Interface deviation documented here (deliberate, guard NOT weakened):
``guarded_attempt`` treats a transport exception as unknown usage -> the
reservation is retained AND the ledger stops (fail closed). The
function-calling fallback is a second guarded attempt, so under an installed
guard the fallback reservation is denied with ``LedgerStopped``. The spec
expectation "fallback executes -> ledger shows 2 requests / 2 reservations"
is unreachable without weakening that contract, so the guarded fallback test
asserts the fail-closed behavior instead; the unguarded fallback test proves
the fallback mechanism itself still works exactly as before.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import litellm
import pytest

import operator_trace_harness as h
from dojo.core.solvers.llm_helpers.backends.lite_llm import LiteLLMClient
from dojo.solvers.evo.evo import Evolutionary
from dojo.solvers.evo.operator_trace import OperatorTraceWriter, validate_operator_event
from dojo.utils.logger import config_logger
from research.contracts.budget import (
    BudgetError,
    BudgetLedger,
    BudgetLimits,
    CancellationToken,
    LedgerStopped,
)
from research.search.budget_transport import (
    TransportGuardConfig,
    install_budget_guard,
)

LITELLM_MODULE = "dojo.core.solvers.llm_helpers.backends.lite_llm"
LIMITS = BudgetLimits(
    max_tokens=1_000_000, max_requests=10_000, max_elapsed_seconds=86400
)
GUARD_CONFIG = TransportGuardConfig(estimated_input_tokens=100, max_output_tokens=100)


class FakeResponse:
    """Minimal litellm-style response: choices[0].message.content + to_dict."""

    def __init__(self, content: str, usage: dict | None):
        self._content = content
        self._usage = usage
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]

    def to_dict(self):
        payload = {}
        if self._usage is not None:
            payload["usage"] = dict(self._usage)
        return payload


def make_client() -> LiteLLMClient:
    return LiteLLMClient(
        types.SimpleNamespace(
            model_id="fake-model",
            base_url="http://example.invalid",
            api_key="fake-key",
            use_azure_client=False,
            provider="openai",
        )
    )


def make_completion(calls, script):
    """Build a fake completion_fn; script(calls, kwargs) -> FakeResponse."""
    def fake_completion(*args, **kwargs):
        calls.append(kwargs)
        return script(calls, kwargs)

    return fake_completion


def good_script(calls, kwargs):
    return FakeResponse(
        f"ok-{len(calls)}", {"prompt_tokens": 10, "completion_tokens": 20}
    )


def open_ledger(tmp_path) -> BudgetLedger:
    return BudgetLedger.open(Path(tmp_path) / "ledger.json", limits=LIMITS)


def test_guard_reserves_before_transport_and_commits_provider_usage(
    tmp_path, monkeypatch
):
    ledger = open_ledger(tmp_path)
    calls: list = []

    order: list[str] = []

    def ordered_script(calls, kwargs):
        order.append("transport")
        return good_script(calls, kwargs)

    monkeypatch.setattr(
        f"{LITELLM_MODULE}.completion_fn", make_completion(calls, ordered_script)
    )

    original_reserve = ledger.reserve
    original_commit = ledger.commit

    def spy_reserve(*args, **kwargs):
        order.append("reserve")
        return original_reserve(*args, **kwargs)

    def spy_commit(*args, **kwargs):
        order.append("commit")
        return original_commit(*args, **kwargs)

    ledger.reserve = spy_reserve
    ledger.commit = spy_commit

    client = make_client()
    install_budget_guard(client, ledger, CancellationToken(), GUARD_CONFIG)

    output, usage_stats = client.query(messages=[{"role": "user", "content": "hi"}])

    assert order == ["reserve", "transport", "commit"]
    assert len(calls) == 1
    snapshot = ledger.snapshot()
    assert snapshot["committed"] == {"requests": 1, "tokens": 30}
    assert snapshot["open_reservations"] == []
    assert usage_stats["usage_provenance"] == "provider"
    assert usage_stats["prompt_tokens"] == 10
    assert usage_stats["completion_tokens"] == 20
    # Provider did not report a total, so it is computed from the components.
    assert usage_stats["total_tokens"] == 30
    ledger.close()


def test_provider_total_tokens_not_overwritten(tmp_path, monkeypatch):
    calls: list = []

    def script(calls, kwargs):
        return FakeResponse(
            "ok",
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 999},
        )

    monkeypatch.setattr(f"{LITELLM_MODULE}.completion_fn", make_completion(calls, script))
    client = make_client()
    _, usage_stats = client.query(messages=[{"role": "user", "content": "hi"}])
    assert usage_stats["total_tokens"] == 999
    assert usage_stats["usage_provenance"] == "provider"


def test_missing_usage_retains_reservation_and_stops_ledger(tmp_path, monkeypatch):
    ledger = open_ledger(tmp_path)
    calls: list = []

    def no_usage_script(calls, kwargs):
        return FakeResponse(f"ok-{len(calls)}", None)

    monkeypatch.setattr(
        f"{LITELLM_MODULE}.completion_fn", make_completion(calls, no_usage_script)
    )
    client = make_client()
    install_budget_guard(client, ledger, CancellationToken(), GUARD_CONFIG)

    with pytest.raises(BudgetError, match="usage_unavailable"):
        client.query(messages=[{"role": "user", "content": "hi"}])

    snapshot = ledger.snapshot()
    # Dynamic byte-based input reservation is retained, including framing.
    assert snapshot["committed"]["requests"] == 1
    assert snapshot["committed"]["tokens"] >= 200
    assert snapshot["open_reservations"] == []
    assert snapshot["stopped"] is not None
    assert snapshot["stopped"]["reason"].startswith("unknown_usage")

    with pytest.raises(LedgerStopped):
        client.query(messages=[{"role": "user", "content": "again"}])
    assert len(calls) == 1
    ledger.close()

    # The same response WITHOUT a guard is fine and marked estimated.
    unguarded = make_client()
    _, usage_stats = unguarded.query(messages=[{"role": "user", "content": "hi there"}])
    assert usage_stats["usage_provenance"] == "estimated"
    assert usage_stats["prompt_tokens"] > 0
    assert usage_stats["completion_tokens"] > 0


def test_hidden_retries_disabled_under_guard(tmp_path, monkeypatch):
    ledger = open_ledger(tmp_path)
    calls: list = []

    def failing_script(calls, kwargs):
        raise litellm.RateLimitError(
            message="rate limited", llm_provider="openai", model="fake"
        )

    monkeypatch.setattr(
        f"{LITELLM_MODULE}.completion_fn", make_completion(calls, failing_script)
    )
    client = make_client()
    install_budget_guard(client, ledger, CancellationToken(), GUARD_CONFIG)

    with pytest.raises(litellm.RateLimitError):
        client.query(messages=[{"role": "user", "content": "hi"}])

    # Exactly one invocation: no litellm-internal retry happened under guard.
    assert len(calls) == 1
    assert calls[0]["max_retries"] == 0
    assert calls[0]["num_retries"] == 0
    # Transport failure => unknown usage => retained + stopped.
    assert ledger.snapshot()["stopped"] is not None
    ledger.close()


def test_caller_supplied_retries_respected_when_unguarded(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(f"{LITELLM_MODULE}.completion_fn", make_completion(calls, good_script))
    client = make_client()

    client.query(messages=[{"role": "user", "content": "hi"}], num_retries=7)
    assert calls[-1]["num_retries"] == 7
    # Caller pinned only num_retries: no default max_retries is injected.
    assert "max_retries" not in calls[-1]

    # Default behavior unchanged when the caller passes nothing.
    client.query(messages=[{"role": "user", "content": "hi"}])
    assert calls[-1]["num_retries"] == 10
    assert calls[-1]["max_retries"] == 10


FUNC_SCHEMA = json.dumps(
    {"type": "object", "properties": {"x": {"type": "string"}}}
)


def test_function_calling_fallback_unguarded(tmp_path, monkeypatch):
    calls: list = []

    def script(calls, kwargs):
        if len(calls) == 1:
            raise litellm.BadRequestError(
                message="this model does not support function calling",
                model="fake",
                llm_provider="openai",
            )
        return FakeResponse("plain text output", {"prompt_tokens": 3, "completion_tokens": 4})

    monkeypatch.setattr(f"{LITELLM_MODULE}.completion_fn", make_completion(calls, script))
    client = make_client()
    output, usage_stats = client.query(
        messages=[{"role": "user", "content": "hi"}],
        json_schema=FUNC_SCHEMA,
        function_name="f",
        function_description="desc",
    )
    assert len(calls) == 2
    assert "functions" in calls[0]
    assert "functions" not in calls[1]
    assert "function_call" not in calls[1]
    assert output == "plain text output"


def test_function_calling_fallback_fails_closed_under_guard(tmp_path, monkeypatch):
    """See module docstring: the first attempt's transport exception retains
    its reservation and stops the ledger (unknown usage), so the fallback's
    own guarded reservation is denied with LedgerStopped."""
    ledger = open_ledger(tmp_path)
    calls: list = []

    def script(calls, kwargs):
        if len(calls) == 1:
            raise litellm.BadRequestError(
                message="this model does not support function calling",
                model="fake",
                llm_provider="openai",
            )
        return FakeResponse("plain text output", {"prompt_tokens": 3, "completion_tokens": 4})

    monkeypatch.setattr(f"{LITELLM_MODULE}.completion_fn", make_completion(calls, script))
    client = make_client()
    install_budget_guard(client, ledger, CancellationToken(), GUARD_CONFIG)

    with pytest.raises(LedgerStopped):
        client.query(
            messages=[{"role": "user", "content": "hi"}],
            json_schema=FUNC_SCHEMA,
            function_name="f",
            function_description="desc",
        )
    assert len(calls) == 1
    snapshot = ledger.snapshot()
    assert snapshot["committed"]["requests"] == 1
    assert snapshot["stopped"] is not None
    ledger.close()


def test_install_budget_guard_twice_refused(tmp_path):
    ledger = open_ledger(tmp_path)
    client = make_client()
    install_budget_guard(client, ledger, CancellationToken(), GUARD_CONFIG)
    with pytest.raises(BudgetError, match="guard_already_installed"):
        install_budget_guard(client, ledger, CancellationToken(), GUARD_CONFIG)
    ledger.close()


def test_solver_integration_guarded_clients_and_trace(tmp_path, monkeypatch):
    """Full integration: REAL Evolutionary + REAL GenericLLM + REAL
    LiteLLMClient (completion monkeypatched with a scripted deterministic
    fake returning operator-parseable content WITH provider usage), budget
    guards installed on every operator client, real BudgetLedger(tmp) and
    OperatorTraceWriter. ALL FOUR operators (draft/improve/debug/crossover)
    go through the real guarded client — nothing is wired around it."""
    config_logger(None)
    cfg = h.make_solver_config(
        Path(tmp_path) / "checkpoint",
        overrides={
            "num_generations": 3,
            "individuals_per_generation": 2,
            "num_generations_till_crossover": 1,
            "crossover_prob": 0.5,
            "max_debug_depth": 1,
            "step_limit": 12,
        },
    )
    solver = Evolutionary(cfg, task_info=dict(h.TASK_INFO))

    calls: list = []

    def script(calls, kwargs):
        marker = f"call-{len(calls)}"
        code = (
            "import pandas as pd\n"
            "SCORE = 0.5\n"
            f"MARKER = \"{marker}\"\n"
        )
        content = f"Plan {marker}.\n```python\n{code}```\n"
        return FakeResponse(content, {"prompt_tokens": 10, "completion_tokens": 20})

    monkeypatch.setattr(f"{LITELLM_MODULE}.completion_fn", make_completion(calls, script))

    operator_llms = {
        "draft": solver.draft_fn.args[0],
        "improve": solver.improve_fn.args[0],
        "debug": solver.debug_fn.args[0],
        "crossover": solver.crossover_fn.args[0],
        "analyze": solver.analyze_fn.args[0],
    }
    ledger = open_ledger(tmp_path)
    token = CancellationToken()
    for llm in operator_llms.values():
        install_budget_guard(llm.client, ledger, token, GUARD_CONFIG)

    trace_path = Path(tmp_path) / "trace.jsonl"
    solver.operator_tracer = OperatorTraceWriter(trace_path)
    task = h.FakeTask()
    solver.search(task, {})

    non_root = [n for n in solver.journal.nodes if not solver.journal.is_root_node(n)]
    events = h.read_trace(trace_path)

    # Ledger request count == actual fake-completion invocations == sum of
    # operator LLM calls (analyze excluded: deterministic metadata path).
    operator_calls = sum(
        llm.call_tracker for name, llm in operator_llms.items() if name != "analyze"
    )
    snapshot = ledger.snapshot()
    assert snapshot["committed"]["requests"] == len(calls) == operator_calls
    assert operator_llms["analyze"].call_tracker == 0
    assert snapshot["committed"]["tokens"] == 30 * operator_calls
    assert snapshot["stopped"] is None

    # Trace: one valid event per created node, stable config hash.
    assert len(events) == len(non_root)
    for event in events:
        validate_operator_event(event)
    assert len({e["config"]["config_sha256"] for e in events}) == 1

    # Every candidate score came from eval_result metadata; operator usage
    # stats record provider provenance (never estimates).
    usage_entries = [
        metrics["usage"]
        for node in non_root
        for metrics in node.operators_metrics
        if isinstance(metrics, dict) and isinstance(metrics.get("usage"), dict)
    ]
    assert usage_entries
    assert all(u.get("usage_provenance") == "provider" for u in usage_entries)
    ledger.close()
