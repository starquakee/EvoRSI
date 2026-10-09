"""Persistent serial budget ledger and cancellation hooks (US-003).

The ledger enforces the PRD model budget BEFORE every model attempt,
including retries: 200000 tokens, 30 requests, 5400 elapsed seconds,
whichever is exhausted first.

- Reservation first: every attempt reserves a conservative input estimate
  plus the max output tokens before the transport is invoked; the
  reservation occupies token AND request slots until settled.
- Unknown usage fails closed: if a call returns without usage data, or
  the transport or usage parser raises (the request may have reached the
  model), the reservation is retained as permanently charged and the
  ledger stops. The same applies to a reservation left open across a
  restart: reopening a ledger with unresolved reservations stops it.
- Never reset: the ledger is a JSON file. Reopening it continues the
  committed totals, the original started_at (elapsed anchor), the
  persisted limits and any stopped state. There is no reset API; a
  restart cannot buy budget, and reopening with larger limits than the
  persisted ones is refused (limits_cannot_expand).
- Serial access: an exclusive flock on a sidecar lock file refuses a
  second concurrent opener (LedgerLocked); a threading.Lock serializes
  in-process mutations. Once close() releases the lock, every further
  operation on that instance raises BudgetError (ledger_closed).
- Snapshots are deep copies: mutating a returned snapshot can never
  alter live accounting.

CancellationToken carries an ABSOLUTE deadline and a persistent stopped
flag, both surviving checkpoint serialize/restore, so resume can never
extend a deadline or revive a stopped run.

No real model calls happen here: transports are injected callables.
"""
from __future__ import annotations

import copy
import fcntl
import math
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .persist import PersistError, atomic_write_json, read_json_object

DEFAULT_MAX_TOKENS = 200_000
DEFAULT_MAX_REQUESTS = 30
DEFAULT_MAX_ELAPSED_SECONDS = 5_400.0


def _count(value: Any, *, positive: bool = False) -> bool:
    return type(value) is int and value >= (1 if positive else 0)


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


class BudgetError(RuntimeError):
    """Base class for ledger failures."""


class BudgetExhausted(BudgetError):
    """A reservation would exceed token/request/elapsed limits."""


class LedgerStopped(BudgetError):
    """The ledger was stopped; no new calls are permitted after stop."""


class LedgerLocked(BudgetError):
    """Another process already holds this ledger (serial access)."""


class Cancelled(RuntimeError):
    """Cancellation token deadline reached or stop requested."""


@dataclass(frozen=True)
class BudgetLimits:
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_requests: int = DEFAULT_MAX_REQUESTS
    max_elapsed_seconds: float = DEFAULT_MAX_ELAPSED_SECONDS

    def __post_init__(self) -> None:
        if not _count(self.max_tokens, positive=True) or not _count(self.max_requests, positive=True):
            raise ValueError("limits_require_positive_integers")
        if not _finite(self.max_elapsed_seconds, positive=True):
            raise ValueError("elapsed_limit_requires_positive_finite_number")


@dataclass(frozen=True)
class ModelUsage:
    """Actual usage reported by a model response."""

    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if not _count(self.input_tokens) or not _count(self.output_tokens):
            raise ValueError("usage_requires_nonnegative_integers")

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class Reservation:
    """An open budget reservation: conservative input estimate + max output."""

    reservation_id: str
    tokens: int
    created_at: float


class CancellationToken:
    """Deadline + stop hook with checkpoint-safe serialization.

    The deadline is stored as an absolute epoch and the stopped flag is
    persisted, so serialize/restore (checkpoint resume) can never extend
    the deadline or revive a stopped run.
    """

    def __init__(
        self,
        *,
        deadline_epoch: float | None = None,
        stopped_reason: str | None = None,
        stopped_at: float | None = None,
    ):
        if deadline_epoch is not None and not _finite(deadline_epoch):
            raise ValueError("invalid_deadline")
        if stopped_at is not None and not _finite(stopped_at):
            raise ValueError("invalid_stop_time")
        self.deadline_epoch = deadline_epoch
        self.stopped_reason = stopped_reason
        self.stopped_at = stopped_at

    @classmethod
    def after(cls, seconds: float, *, now: float | None = None) -> "CancellationToken":
        """Token with an absolute deadline `seconds` from `now`."""
        if not _finite(seconds, positive=True):
            raise ValueError("deadline_must_be_positive_finite")
        anchor = time.time() if now is None else now
        return cls(deadline_epoch=anchor + seconds)

    def check(self, *, now: float | None = None) -> None:
        """Raise Cancelled if stopped or past the absolute deadline."""
        if self.stopped_reason is not None:
            raise Cancelled(f"stopped:{self.stopped_reason}")
        if self.deadline_epoch is not None:
            current = time.time() if now is None else now
            if current >= self.deadline_epoch:
                raise Cancelled("deadline_exceeded")

    def stop(self, reason: str, *, now: float | None = None) -> None:
        if self.stopped_reason is None:
            self.stopped_reason = reason
            self.stopped_at = time.time() if now is None else now

    def to_dict(self) -> dict[str, Any]:
        return {
            "deadline_epoch": self.deadline_epoch,
            "stopped_reason": self.stopped_reason,
            "stopped_at": self.stopped_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CancellationToken":
        deadline = data.get("deadline_epoch")
        return cls(
            deadline_epoch=None if deadline is None else float(deadline),
            stopped_reason=(
                None if data.get("stopped_reason") is None
                else str(data["stopped_reason"])
            ),
            stopped_at=(
                None if data.get("stopped_at") is None
                else float(data["stopped_at"])
            ),
        )


class BudgetLedger:
    """Persistent serial budget ledger; open() never resets existing state."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        path: Path,
        limits: BudgetLimits,
        now_fn: Callable[[], float],
        state: dict[str, Any],
        lock_handle: Any,
    ):
        self._path = Path(path)
        self._limits = limits
        self._now_fn = now_fn
        self._state = state
        self._lock_handle = lock_handle
        self._mutex = threading.Lock()
        self._attempt_mutex = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------ open
    @classmethod
    def open(
        cls,
        path: Path,
        *,
        limits: BudgetLimits | None = None,
        now_fn: Callable[[], float] | None = None,
    ) -> "BudgetLedger":
        """Open (or create) the ledger at path. Existing state is continued,
        never reset: committed totals, started_at, persisted limits and the
        stopped flag survive. Reopening with larger limits than persisted is
        refused; reopening with unresolved reservations stops the ledger —
        their usage is unknown, which is a stop condition."""
        path = Path(path)
        clock = time.time if now_fn is None else now_fn
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = open(path.with_name(path.name + ".lock"), "a+b")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock_handle.close()
            raise LedgerLocked(f"ledger_in_use:{path.name}") from exc
        try:
            if path.exists():
                state = cls._load(path)
                persisted = BudgetLimits(
                    max_tokens=int(state["limits"]["max_tokens"]),
                    max_requests=int(state["limits"]["max_requests"]),
                    max_elapsed_seconds=float(
                        state["limits"]["max_elapsed_seconds"]
                    ),
                )
                if limits is not None and (
                    limits.max_tokens > persisted.max_tokens
                    or limits.max_requests > persisted.max_requests
                    or limits.max_elapsed_seconds
                    > persisted.max_elapsed_seconds
                ):
                    raise BudgetError("limits_cannot_expand")
                effective_limits = persisted if limits is None else limits
                if limits is not None:
                    state["limits"] = {
                        "max_tokens": limits.max_tokens,
                        "max_requests": limits.max_requests,
                        "max_elapsed_seconds": limits.max_elapsed_seconds,
                    }
                    atomic_write_json(path, state)
                if state["stopped"] is None and state["reservations"]:
                    # A reservation survived a restart without being settled:
                    # its usage is unknowable, so retain it and stop.
                    stopped = {"reason": "unresolved_reservations_on_restart",
                               "at": float(clock())}
                    state["stopped"] = stopped
                    state["events"].append(
                        {"type": "stop", "at": stopped["at"],
                         "reason": stopped["reason"]}
                    )
                    atomic_write_json(path, state)
            else:
                effective_limits = limits if limits is not None else BudgetLimits()
                state = {
                    "schema_version": cls.SCHEMA_VERSION,
                    "started_at": clock(),
                    "stopped": None,
                    "limits": {
                        "max_tokens": effective_limits.max_tokens,
                        "max_requests": effective_limits.max_requests,
                        "max_elapsed_seconds": (
                            effective_limits.max_elapsed_seconds
                        ),
                    },
                    "committed": {"requests": 0, "tokens": 0},
                    "reservations": [],
                    "events": [],
                }
                atomic_write_json(path, state)
        except Exception:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
            raise
        return cls(path, effective_limits, clock, state, lock_handle)

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        try:
            state = read_json_object(path)
        except PersistError as exc:
            raise BudgetError(f"corrupt_ledger:{exc}") from exc
        if state.get("schema_version") != BudgetLedger.SCHEMA_VERSION:
            raise BudgetError("unsupported_schema_version")
        committed = state.get("committed")
        reservations = state.get("reservations")
        events = state.get("events")
        limits = state.get("limits")
        if (
            not _finite(state.get("started_at"))
            or not isinstance(committed, dict)
            or not _count(committed.get("requests"))
            or not _count(committed.get("tokens"))
            or not isinstance(reservations, list)
            or not isinstance(events, list)
            or "stopped" not in state
            or not (state["stopped"] is None or isinstance(state["stopped"], dict))
            or not isinstance(limits, dict)
            or not _count(limits.get("max_tokens"), positive=True)
            or not _count(limits.get("max_requests"), positive=True)
            or not _finite(limits.get("max_elapsed_seconds"), positive=True)
        ):
            raise BudgetError("corrupt_ledger:schema")
        seen = set()
        for reservation in reservations:
            if (not isinstance(reservation, dict)
                    or not isinstance(reservation.get("id"), str)
                    or not reservation["id"]
                    or reservation["id"] in seen
                    or not _count(reservation.get("tokens"), positive=True)
                    or not _finite(reservation.get("created_at"))):
                raise BudgetError("corrupt_ledger:reservation")
            seen.add(reservation["id"])
        stopped = state["stopped"]
        if stopped is not None and (
            not isinstance(stopped.get("reason"), str)
            or not stopped["reason"] or not _finite(stopped.get("at"))
        ):
            raise BudgetError("corrupt_ledger:stopped")
        return state

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        if not self._closed:
            self._closed = True
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()

    def __enter__(self) -> "BudgetLedger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --------------------------------------------------------------- helpers
    def _now(self) -> float:
        current = self._now_fn()
        if not _finite(current) or current < self.started_at:
            raise BudgetError("invalid_or_rolled_back_clock")
        return float(current)

    def _require_open(self) -> None:
        if self._closed:
            raise BudgetError("ledger_closed")

    def _persist_locked(self) -> None:
        atomic_write_json(self._path, self._state)

    def _event_locked(self, kind: str, **fields: Any) -> None:
        self._state["events"].append({"type": kind, "at": self._now(), **fields})

    @property
    def started_at(self) -> float:
        return float(self._state["started_at"])

    @property
    def stopped(self) -> bool:
        return self._state["stopped"] is not None

    @property
    def deadline_epoch(self) -> float:
        """Absolute run deadline from the PERSISTED limits and started_at.

        The snapshot intentionally exposes only remaining/elapsed; this is
        the safe way to anchor a CancellationToken to the original deadline
        across process boundaries (never ``time.time() + remaining``, which
        would silently extend the budget when a child restarts).
        """
        return self.started_at + self._limits.max_elapsed_seconds

    def snapshot(self) -> dict[str, Any]:
        """Current accounting view (committed + open reservations + remaining).

        The returned dict is a deep copy: mutating it never affects live
        accounting.
        """
        self._require_open()
        with self._mutex:
            now = self._now()
            committed = self._state["committed"]
            open_reservations = copy.deepcopy(self._state["reservations"])
            open_tokens = sum(int(r["tokens"]) for r in open_reservations)
            elapsed = max(0.0, now - self.started_at)
            used_tokens = int(committed["tokens"]) + open_tokens
            used_requests = int(committed["requests"]) + len(open_reservations)
            return {
                "stopped": copy.deepcopy(self._state["stopped"]),
                "started_at": self.started_at,
                "elapsed_seconds": elapsed,
                "committed": dict(committed),
                "open_reservations": open_reservations,
                "remaining": {
                    "tokens": self._limits.max_tokens - used_tokens,
                    "requests": self._limits.max_requests - used_requests,
                    "elapsed_seconds": (
                        self._limits.max_elapsed_seconds - elapsed
                    ),
                },
            }

    # -------------------------------------------------------------- mutation
    def reserve(
        self,
        estimated_input_tokens: int,
        max_output_tokens: int,
    ) -> Reservation:
        """Reserve budget BEFORE a model attempt (call or retry).

        Raises BudgetError if the ledger is closed, BudgetExhausted when
        tokens, requests or elapsed time would exceed the limits, and
        LedgerStopped after stop.
        """
        self._require_open()
        if not _count(estimated_input_tokens) or not _count(max_output_tokens, positive=True):
            raise ValueError("invalid_estimate")
        with self._mutex:
            now = self._now()
            if now - self.started_at >= self._limits.max_elapsed_seconds:
                self._event_locked("reserve_denied", reason="elapsed")
                self._persist_locked()
                raise BudgetExhausted("elapsed_seconds_exhausted")
            stopped = self._state["stopped"]
            if stopped is not None:
                raise LedgerStopped(f"stopped:{stopped['reason']}")
            needed = estimated_input_tokens + max_output_tokens
            committed = self._state["committed"]
            reservations = self._state["reservations"]
            open_tokens = sum(int(r["tokens"]) for r in reservations)
            if int(committed["requests"]) + len(reservations) + 1 > (
                self._limits.max_requests
            ):
                self._event_locked("reserve_denied", reason="requests")
                self._persist_locked()
                raise BudgetExhausted("requests_exhausted")
            if int(committed["tokens"]) + open_tokens + needed > (
                self._limits.max_tokens
            ):
                self._event_locked("reserve_denied", reason="tokens")
                self._persist_locked()
                raise BudgetExhausted("tokens_exhausted")
            reservation = Reservation(
                reservation_id=uuid.uuid4().hex,
                tokens=needed,
                created_at=now,
            )
            reservations.append(
                {
                    "id": reservation.reservation_id,
                    "tokens": reservation.tokens,
                    "created_at": reservation.created_at,
                }
            )
            self._event_locked(
                "reserve",
                reservation_id=reservation.reservation_id,
                tokens=reservation.tokens,
            )
            self._persist_locked()
            return reservation

    def commit(self, reservation_id: str, usage: ModelUsage | None) -> None:
        """Settle a reservation with actual usage.

        usage=None (unknown usage: unreported, transport failure or usage
        parser failure) retains the reservation as permanently charged AND
        stops the ledger — an unknown token balance is a PRD stop condition.
        """
        self._require_open()
        with self._mutex:
            reservations = self._state["reservations"]
            index = next(
                (i for i, r in enumerate(reservations) if r["id"] == reservation_id),
                None,
            )
            if index is None:
                raise BudgetError(f"unknown_reservation:{reservation_id}")
            open_res = reservations.pop(index)
            committed = self._state["committed"]
            committed["requests"] = int(committed["requests"]) + 1
            if usage is None:
                committed["tokens"] = int(committed["tokens"]) + int(
                    open_res["tokens"]
                )
                self._event_locked(
                    "retain",
                    reservation_id=reservation_id,
                    tokens=int(open_res["tokens"]),
                )
                self._stop_locked(f"unknown_usage:{reservation_id}")
            else:
                committed["tokens"] = int(committed["tokens"]) + usage.total
                self._event_locked(
                    "commit",
                    reservation_id=reservation_id,
                    tokens=usage.total,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
                if usage.total > int(open_res["tokens"]):
                    self._stop_locked("usage_exceeded_reservation")
            self._persist_locked()

    def stop(self, reason: str) -> None:
        """Stop the ledger; no new reservations afterwards. First stop wins."""
        self._require_open()
        with self._mutex:
            self._stop_locked(reason)
            self._persist_locked()

    def _stop_locked(self, reason: str) -> None:
        if self._state["stopped"] is None:
            stopped = {"reason": reason, "at": self._now()}
            self._state["stopped"] = stopped
            self._state["events"].append(
                {"type": "stop", "at": stopped["at"], "reason": reason}
            )


# ------------------------------------------------------------ guarded calls
Transport = Callable[[Any], Any]
UsageFn = Callable[[Any], ModelUsage | None]
RetryPredicate = Callable[[Any], bool]


def guarded_attempt(
    ledger: BudgetLedger,
    token: CancellationToken,
    transport: Transport,
    request: Any,
    *,
    estimated_input_tokens: int,
    max_output_tokens: int,
    usage_fn: UsageFn,
    on_reserve: Callable[[Reservation], None] | None = None,
) -> Any:
    """One model attempt under the full contract:

    cancellation check -> reserve (before the call) -> transport -> settle.
    A transport exception OR a usage-parser exception means unknown usage:
    the reservation is retained and the ledger stops, then the exception
    propagates. ``on_reserve`` (optional) is invoked with the reservation
    right after it is taken, so callers can link the attempt to the ledger.
    An observer failure is treated like a transport failure: the reservation
    is retained, the ledger stops, and the provider is never called.
    """
    with ledger._attempt_mutex:
        token.check()
        reservation = ledger.reserve(estimated_input_tokens, max_output_tokens)
        try:
            if on_reserve is not None:
                on_reserve(reservation)
            response = transport(request)
        except Exception:
            ledger.commit(reservation.reservation_id, None)
            raise
        try:
            usage = usage_fn(response)
            if usage is not None and not isinstance(usage, ModelUsage):
                raise ValueError("invalid_usage_payload")
        except Exception:
            ledger.commit(reservation.reservation_id, None)
            raise
        ledger.commit(reservation.reservation_id, usage)
        return response


def call_with_retries(
    ledger: BudgetLedger,
    token: CancellationToken,
    transport: Transport,
    request: Any,
    *,
    max_attempts: int,
    estimated_input_tokens: int,
    max_output_tokens: int,
    usage_fn: UsageFn,
    should_retry: RetryPredicate,
    on_reserve: Callable[[Reservation], None] | None = None,
) -> Any:
    """Retry loop where EVERY attempt (including each retry) reserves first.

    Retries are decided from the response (e.g. a rate-limit reply with
    reported usage). Transport exceptions fail closed in guarded_attempt
    and are never retried. Ledger stop/exhaustion and cancellation
    propagate immediately.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts_must_be_positive")
    for attempt_no in range(1, max_attempts + 1):
        response = guarded_attempt(
            ledger,
            token,
            transport,
            request,
            estimated_input_tokens=estimated_input_tokens,
            max_output_tokens=max_output_tokens,
            usage_fn=usage_fn,
            on_reserve=on_reserve,
        )
        if attempt_no == max_attempts or not should_retry(response):
            return response
    raise AssertionError("unreachable")  # pragma: no cover
