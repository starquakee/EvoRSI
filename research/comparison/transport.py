"""Retry-free guarded provider transport for the comparison runner (EXP-005).

Mirrors the acceptance RequestGuard pattern but the ONLY accounting layer is
the comparison ledger (EXP-004): every operator attempt is exactly one
``guarded_comparison_attempt`` — reserve (atomic global+cell+inner debit of
the conservative full-prompt estimate plus the pinned 8192 output) -> one
transport call under a hard controller deadline (covering full stream
consumption) -> settle with provider usage. A transport/usage/observer
failure retains the full reservation and stops the ledger globally inside
``guarded_comparison_attempt``; no retry or next request can silently
follow an uncertain attempt.

The pinned K3 profile (ModelProfile) is enforced on EVERY attempt BEFORE
any reservation: an off-profile request fails closed and never reaches the
provider. Debug attempts carry the run-level total debug budget (0 or the
pinned inner cap 2) as the ledger ``debug_limit`` — distinct from theta's
per-candidate repair-chain depth.
"""
from __future__ import annotations

import contextvars
import math
import threading
import time
from typing import Any, Callable, Mapping

from research.comparison.budget import (
    ComparisonBudgetError,
    ComparisonLedger,
    ComparisonLedgerStopped,
    ComparisonReservation,
    guarded_comparison_attempt,
)
from research.comparison.protocol import ModelProfile
from research.comparison.specs import InnerRunSpec
from research.contracts.budget import (
    BudgetError,
    CancellationToken,
    Cancelled,
)
from research.search.budget_transport import (
    conservative_input_tokens,
    extract_provider_usage,
)

#: Operators that produce MAIN candidates (draft/improve/crossover).
MAIN_OPERATORS = ("draft", "improve", "crossover")
#: The debug operator spends the optional run-level debug budget.
DEBUG_OPERATORS = ("debug",)
#: Auxiliary operators (analyze/rich memory) are never budgeted in the
#: comparison: with experience enabled and prompt memory pinned off they
#: must never fire; if one does, the guard refuses BEFORE any reservation.
UNBUDGETED_OPERATORS = ("analyze", "rich_memory_summary")

_REQUEST_DEADLINE_SECONDS = 600.0


def comparison_request_profile() -> dict[str, Any]:
    """The exact transport kwargs every comparison operator request carries.

    Any deviation is refused before reservation (fail closed, no off-profile
    request and no silent parameter drift from the approved K3 profile).
    Retry pins are explicit: hidden SDK/litellm retries would bypass the
    per-attempt reservation.
    """
    profile = ModelProfile()
    return {
        "model": f"openai/{profile.model_id}",
        "base_url": profile.base_url,
        "stream": profile.stream,
        "max_tokens": profile.max_output_tokens,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "reasoning_effort": profile.reasoning_effort,
        "allowed_openai_params": list(profile.allowed_openai_params),
        "max_retries": profile.transport_retries,
        "num_retries": profile.transport_retries,
    }


def _profile_value_matches(expected: Any, value: Any) -> bool:
    """Type-strict pin comparison: True != 1 and 8192.0 != 8192 here."""
    if isinstance(expected, bool):
        return value is expected
    if isinstance(expected, int):
        return type(value) is int and value == expected
    if isinstance(expected, float):
        return type(value) in (int, float) and float(value) == expected
    if isinstance(expected, list):
        return isinstance(value, list) and value == expected
    return type(value) is str and value == expected


def _provider_request_id(response: Any) -> str | None:
    """Provider request identity from a guarded response mapping.

    The streaming transport returns ``{"usage": ...}``. An explicitly
    reported but malformed/ambiguous identity (empty, non-string, or the
    backend's conflict marker) is a hard failure — never silently erased
    to None; a genuinely absent identity stays None (the durable LOCAL
    identity is the ledger reservation id; local UUIDs are never reported
    as provider identities).
    """
    if isinstance(response, Mapping):
        usage = response.get("usage")
        if isinstance(usage, Mapping):
            if usage.get("provider_request_id_invalid") is True:
                raise ComparisonBudgetError("provider_request_id_invalid")
            if "provider_request_id" in usage:
                value = usage["provider_request_id"]
                if not isinstance(value, str) or not value:
                    raise ComparisonBudgetError("provider_request_id_invalid")
                return value
    return None


class ComparisonRequestGuard:
    """One provider attempt under the comparison ledger, hard deadline.

    The whole attempt (reservation, provider call including full stream
    consumption, settlement) is serialized by the ledger's attempt mutex
    inside ``guarded_comparison_attempt``. The controller deadline is the
    minimum of the per-request timeout and the ORIGINAL persisted absolute
    ledger deadline; cancellation or usage uncertainty settles the attempt
    as unknown usage (retained in full + global stop), never a free retry.
    """

    def __init__(
        self,
        ledger: ComparisonLedger,
        token: CancellationToken,
        run: InnerRunSpec,
        *,
        kind: str,
        debug_budget: int | None = None,
        request_deadline_seconds: float = _REQUEST_DEADLINE_SECONDS,
        attempt_sink: Callable[..., None] | None = None,
    ):
        if kind not in ("main", "debug"):
            raise ComparisonBudgetError("request_kind_invalid")
        if kind == "debug" and (
            type(debug_budget) is not int or not 0 <= debug_budget <= 2
        ):
            raise ComparisonBudgetError("debug_budget_invalid")
        if (
            isinstance(request_deadline_seconds, bool)
            or not isinstance(request_deadline_seconds, (int, float))
            or not math.isfinite(float(request_deadline_seconds))
            or float(request_deadline_seconds) <= 0
        ):
            raise ComparisonBudgetError("request_deadline_invalid")
        self.ledger = ledger
        self.token = token
        self.run_spec = run
        self.kind = kind
        self.debug_budget = debug_budget
        self.request_deadline_seconds = float(request_deadline_seconds)
        self.attempt_sink = attempt_sink
        self._deadline: float | None = None

    def __call__(self, thunk: Callable[[], Any]) -> Any:
        return self.run(thunk)

    def check_deadline(self) -> None:
        """Polled by the client while consuming the stream."""
        self.token.check()
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise Cancelled("request_deadline_exceeded")

    def run(
        self,
        thunk: Callable[[], Any],
        *,
        messages: Any = None,
        model_kwargs: dict[str, Any] | None = None,
        timeout_key: str = "request_timeout",
    ) -> Any:
        self.token.check()
        if self.ledger.stopped:
            stopped = self.ledger.snapshot()["stopped"]
            raise ComparisonLedgerStopped(f"stopped:{stopped['reason']}")
        for key, expected in comparison_request_profile().items():
            if not _profile_value_matches(expected, (model_kwargs or {}).get(key)):
                # Off-profile requests fail closed BEFORE any reservation.
                raise ComparisonBudgetError(f"request_profile_mismatch:{key}")
        if model_kwargs is not None:
            for multi_key in ("n", "best_of"):
                multi = model_kwargs.get(multi_key, 1)
                if type(multi) is not int or multi != 1:
                    raise ComparisonBudgetError("multiple_completions_forbidden")
            if "max_completion_tokens" in model_kwargs:
                # A second, conflicting output-limit key would bypass the
                # pinned max_tokens reservation basis.
                if not _profile_value_matches(
                    ModelProfile().max_output_tokens,
                    model_kwargs["max_completion_tokens"],
                ):
                    raise ComparisonBudgetError("max_completion_tokens_conflict")
        # The conservative reservation is driven by the EXACT outgoing
        # system/messages/tools (UTF-8 bytes + framing), never an average.
        estimate = conservative_input_tokens(messages, model_kwargs or {})
        if estimate < 1:
            raise ComparisonBudgetError("invalid_estimate")

        reservation_box: dict[str, ComparisonReservation] = {}

        def on_reserve(reservation: ComparisonReservation) -> None:
            reservation_box["reservation"] = reservation

        def attempt(_request: Any) -> Any:
            self.token.check()
            deadline_epoch = self.ledger.deadline_epoch
            remaining = self.request_deadline_seconds
            if deadline_epoch is not None:
                remaining = min(remaining, deadline_epoch - time.time())
            if remaining <= 0:
                raise Cancelled("request_deadline_exceeded")
            self._deadline = time.monotonic() + remaining
            if model_kwargs is not None:
                model_kwargs[timeout_key] = remaining
            if self.attempt_sink is not None:
                # The observer sees the SAME bounded request the provider
                # gets; a failing observer aborts before the provider runs
                # (guarded_comparison_attempt settles unknown usage + stop).
                self.attempt_sink(
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
            thread = threading.Thread(
                target=lambda: context.run(execute),
                daemon=True,
                name="comparison-provider-attempt",
            )
            thread.start()
            while not finished.wait(
                min(0.05, max(0.0, self._deadline - time.monotonic()))
            ):
                try:
                    self.check_deadline()
                except Cancelled:
                    self.token.stop("request_deadline_exceeded")
                    raise
            if "error" in outcome:
                error = outcome["error"]
                if isinstance(error, Exception):
                    raise error
                raise ComparisonBudgetError("provider_attempt_interrupted") from error
            return outcome["response"]

        return guarded_comparison_attempt(
            self.ledger,
            self.run_spec,
            self.kind,
            attempt,
            None,
            estimated_input_tokens=estimate,
            usage_fn=extract_provider_usage,
            debug_limit=(self.debug_budget if self.kind == "debug" else None),
            on_reserve=on_reserve,
            provider_request_id_fn=_provider_request_id,
        )


def install_comparison_guard(
    client: Any,
    ledger: ComparisonLedger,
    token: CancellationToken,
    run: InnerRunSpec,
    *,
    operator: str,
    debug_budget: int,
    attempt_sink: Callable[..., None] | None = None,
) -> Any:
    """Install the comparison guard on one operator's LLM client.

    ``operator`` selects the request kind: draft/improve/crossover are MAIN,
    debug spends the run-level debug budget, and the unbudgeted auxiliary
    operators are refused outright (before any reservation) so no
    unguarded analysis/memory request can fire.
    """
    if getattr(client, "request_guard", None) is not None:
        raise ComparisonBudgetError("guard_already_installed")
    if operator in MAIN_OPERATORS:
        kind: str | None = "main"
    elif operator in DEBUG_OPERATORS:
        kind = "debug"
    elif operator in UNBUDGETED_OPERATORS:
        kind = None
    else:
        raise ComparisonBudgetError(f"operator_unknown:{operator}")

    sink = attempt_sink
    if sink is not None:
        base_sink = sink

        def operator_sink(reservation: Any, messages: Any, model_kwargs: Any) -> None:
            base_sink(reservation, messages, model_kwargs, operator=operator)

        sink = operator_sink

    if kind is None:
        class _RefusedGuard:
            def __call__(self, thunk: Callable[[], Any]) -> Any:
                raise ComparisonBudgetError(f"operator_not_budgeted:{operator}")

            def run(self, thunk: Callable[[], Any], **kwargs: Any) -> Any:
                raise ComparisonBudgetError(f"operator_not_budgeted:{operator}")

            def check_deadline(self) -> None:
                token.check()

        client.request_guard = _RefusedGuard()
    else:
        client.request_guard = ComparisonRequestGuard(
            ledger,
            token,
            run,
            kind=kind,
            debug_budget=(debug_budget if kind == "debug" else None),
            attempt_sink=sink,
        )
    sdk = getattr(client, "client", None)
    if sdk is not None and hasattr(sdk, "max_retries"):
        sdk.max_retries = 0
    return client
