"""One reservation per provider attempt, including stream consumption."""
from __future__ import annotations

import contextvars
import json
import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from research.contracts.budget import (
    BudgetError, BudgetLedger, CancellationToken, Cancelled, ModelUsage,
    guarded_attempt,
)

Thunk = Callable[[], Any]


@dataclass(frozen=True)
class TransportGuardConfig:
    estimated_input_tokens: int
    max_output_tokens: int
    request_deadline_seconds: float | None = None
    # Optional observer invoked with (reservation, messages, model_kwargs)
    # immediately after each reservation, so acceptance evidence can link
    # every provider attempt to its ledger reservation. Never fed secrets:
    # callers must redact credential fields from model_kwargs first.
    attempt_sink: Any = None

    def __post_init__(self) -> None:
        for name, minimum in (("estimated_input_tokens", 0), ("max_output_tokens", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                label = "nonnegative" if minimum == 0 else "positive"
                raise ValueError(f"{name}_must_be_{label}_int")
        value = self.request_deadline_seconds
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0
        ):
            raise ValueError("request_deadline_seconds_must_be_positive_finite")


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BudgetError("usage_unavailable")
    return value


def _field(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def extract_provider_usage(response: Any) -> ModelUsage:
    """Accept only integral, internally consistent, provider-reported counts."""
    candidate = response
    if not isinstance(candidate, Mapping):
        convert = getattr(candidate, "to_dict", None)
        if callable(convert):
            candidate = convert()
    usage = _field(candidate, "usage")
    if usage is None and isinstance(candidate, Mapping) and (
        "prompt_tokens" in candidate or "completion_tokens" in candidate
    ):
        usage = candidate
    if usage is None:
        usage = getattr(response, "usage", None)
    if usage is None:
        raise BudgetError("usage_unavailable")
    for obj in (candidate, usage):
        if _field(obj, "usage_provenance", "provider") != "provider":
            raise BudgetError("usage_unavailable")
    prompt = _count(_field(usage, "prompt_tokens"))
    completion = _count(_field(usage, "completion_tokens"))
    total = _field(usage, "total_tokens")
    if total is not None and _count(total) != prompt + completion:
        raise BudgetError("usage_unavailable")
    return ModelUsage(input_tokens=prompt, output_tokens=completion)


def conservative_input_tokens(messages: Any, model_kwargs: Mapping[str, Any]) -> int:
    """Reserve UTF-8 bytes plus framing, never a whitespace token estimate.

    This acceptance transport is text-only. Multimodal URL contents cannot
    be bounded from the request bytes, so they are rejected before a call.
    Only public request fields enter the estimate; credentials never do.
    """
    if not isinstance(messages, list) or any(
        not isinstance(m, Mapping) or not isinstance(m.get("content"), str)
        for m in messages
    ):
        raise BudgetError("unsupported_nontext_request")
    public = {"messages": messages}
    for key in ("tools", "functions", "response_format", "function_call", "tool_choice"):
        if key in model_kwargs:
            public[key] = model_kwargs[key]
    return len(json.dumps(public, ensure_ascii=False).encode("utf-8")) + 256 + 64 * len(messages)


class RequestGuard:
    """The complete provider attempt runs under the ledger's serial mutex.

    A hard controller deadline covers the whole request (also an endless
    stream), while socket timeouts bound blocked reads. If a provider fails
    to terminate promptly, its daemon thread can only finish that already
    reserved attempt: the ledger stops with the full reservation retained,
    and the runner is free to cancel sandbox jobs and checkpoint immediately.
    No retry or next request can start after this unknown-usage outcome.
    """

    def __init__(self, ledger: BudgetLedger, token: CancellationToken, config: TransportGuardConfig):
        self.ledger, self.token, self.config = ledger, token, config
        self._deadline: float | None = None

    def __call__(self, thunk: Thunk) -> Any:
        return self.run(thunk)

    def check_deadline(self) -> None:
        self.token.check()
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise Cancelled("request_deadline_exceeded")

    def run(self, thunk: Thunk, *, messages: Any = None,
            model_kwargs: dict[str, Any] | None = None,
            timeout_key: str = "request_timeout") -> Any:
        self.token.check()
        estimate = self.config.estimated_input_tokens
        if messages is not None:
            estimate = max(estimate, conservative_input_tokens(messages, model_kwargs or {}))
        if model_kwargs is not None:
            if model_kwargs.get("n", 1) != 1 or model_kwargs.get("best_of", 1) != 1:
                raise BudgetError("multiple_completions_forbidden")
            output_key = "max_completion_tokens" if "max_completion_tokens" in model_kwargs else "max_tokens"
            requested = model_kwargs.get(output_key, self.config.max_output_tokens)
            if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
                raise BudgetError("invalid_output_token_limit")
            model_kwargs[output_key] = min(requested, self.config.max_output_tokens)
            if output_key != "max_tokens":
                model_kwargs.pop("max_tokens", None)

        reservation_box: dict[str, Any] = {}

        def attempt(_request: Any) -> Any:
            self.token.check()
            remaining = float(self.ledger.snapshot()["remaining"]["elapsed_seconds"])
            if self.token.deadline_epoch is not None:
                remaining = min(remaining, self.token.deadline_epoch - time.time())
            if self.config.request_deadline_seconds is not None:
                remaining = min(remaining, self.config.request_deadline_seconds)
            if remaining <= 0:
                raise Cancelled("request_deadline_exceeded")
            self._deadline = time.monotonic() + remaining
            if model_kwargs is not None:
                model_kwargs[timeout_key] = remaining
            if self.config.attempt_sink is not None:
                # The observer sees the SAME bounded request the provider
                # gets (timeout injected above); a failing observer aborts
                # before the provider runs (unknown usage -> retain + stop).
                self.config.attempt_sink(
                    reservation_box["reservation"], messages, model_kwargs
                )
            finished = threading.Event()
            outcome: dict[str, Any] = {}

            def execute() -> None:
                try:
                    outcome["response"] = thunk()
                except BaseException as exc:
                    outcome["error"] = exc
                finally:
                    finished.set()

            context = contextvars.copy_context()
            thread = threading.Thread(target=lambda: context.run(execute), daemon=True,
                                      name="budgeted-provider-attempt")
            thread.start()
            # Check cancellation during a blocked provider call as well.
            while not finished.wait(min(0.05, max(0.0, self._deadline - time.monotonic()))):
                try:
                    self.check_deadline()
                except Cancelled:
                    self.token.stop("request_deadline_exceeded")
                    raise
            if "error" in outcome:
                error = outcome["error"]
                if isinstance(error, Exception):
                    raise error
                raise BudgetError("provider_attempt_interrupted") from error
            return outcome["response"]

        on_reserve = None
        if self.config.attempt_sink is not None:
            def on_reserve(reservation: Any) -> None:
                reservation_box["reservation"] = reservation

        return guarded_attempt(
            self.ledger, self.token, attempt, None,
            estimated_input_tokens=estimate,
            max_output_tokens=self.config.max_output_tokens,
            usage_fn=extract_provider_usage,
            on_reserve=on_reserve,
        )


def make_request_guard(ledger: BudgetLedger, token: CancellationToken,
                       config: TransportGuardConfig) -> RequestGuard:
    return RequestGuard(ledger, token, config)


def install_budget_guard(client: Any, ledger: BudgetLedger, token: CancellationToken,
                         config: TransportGuardConfig, *, operator: str | None = None) -> Any:
    if getattr(client, "request_guard", None) is not None:
        raise BudgetError("guard_already_installed")
    if operator is not None and config.attempt_sink is not None:
        sink = config.attempt_sink

        def operator_sink(reservation: Any, messages: Any, model_kwargs: Any) -> None:
            sink(reservation, messages, model_kwargs, operator=operator)

        config = replace(config, attempt_sink=operator_sink)
    client.request_guard = make_request_guard(ledger, token, config)
    sdk = getattr(client, "client", None)
    if sdk is not None and hasattr(sdk, "max_retries"):
        sdk.max_retries = 0
    return client
