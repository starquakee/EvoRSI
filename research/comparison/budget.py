"""Atomic global and per-cell budget accounting for the comparison (EXP-004).

Controller-side only (like data.py): not re-exported by the package
``__init__`` so the protocol/spec contract surface stays minimal.

One persistent ledger owns BOTH the global quota (1728 requests /
5400000 tokens / 86400 seconds) and every per-cell quota (64 requests /
200000 tokens, no borrowing) for the frozen 27-cell comparison:

- Reservation first: before every outbound model attempt (main or debug,
  including any failure path) the ledger atomically debits global AND cell
  availability AND the inner-run shape (max 6 main + a run-level TOTAL
  debug budget of at most 2) in one locked, atomically persisted step. The
  conservative reservation covers the full prompt estimate (system +
  messages + framing, UTF-8 bytes — reuse of the pinned
  ``conservative_input_tokens`` estimator) plus the pinned 8192 output
  allowance; ``max_output_tokens`` must equal the pinned profile value.
- Roster-bound: the ledger is created for exactly the 27 approved
  CellSpecs (216 InnerRunSpecs). Unknown cells or inner runs are rejected;
  quota is never silently allocated. The persisted experiment identity
  binds protocol identity, source commit and every cell/run identity (and
  each cell_key/run_key pairing is re-verified on open); reopening with a
  different roster or tampered identity metadata stops the ledger
  GLOBALLY and persistently (``experiment_identity_mismatch`` /
  ``experiment_metadata_mismatch``) before refusing.
- Fail closed on unknown usage: a transport or usage-parser failure
  (``commit(..., None)``) retains the full reservation as permanently
  charged and stops the ledger globally. Reopening with an unresolved
  reservation stops globally — its usage is unknowable. Over-reservation
  usage, corrupt/nonfinite accounting, duplicate identities and explicit
  cleanup/source failure stops behave the same way. A global stop is the
  truthful INCOMPLETE record; caps never auto-expand (limits are the
  frozen protocol pins, there is no limits parameter).
- Truthful cell termination: exhausting a cell cap terminates only that
  cell (recorded with reason) and raises CellBudgetExhausted; other cells
  continue. Global exhaustion stops the whole ledger.
- Clock: ``started_at`` is set exactly once by ``start_clock()`` at the
  supervised live start, persists across resume and is never reset
  (``clock_already_started``). Reservations refuse before the clock
  starts. The deadline (started_at + 86400s) survives reopen unchanged.

Process safety mirrors the US-003 ledger: an exclusive non-blocking flock
on a sidecar lock file refuses concurrent openers (ComparisonLedgerLocked),
a threading.Lock serializes in-process mutations, and every mutation is
durably written via atomic tmp-file + os.replace + fsync.

No real model calls happen here: transports are injected callables and the
profile pins transport_retries=0, so ``guarded_comparison_attempt`` wraps
exactly ONE attempt. The comparison ledger must be the ONLY budget layer
around its transport — do not additionally wrap the same call in the
US-009 RequestGuard/BudgetLedger (that would double charge).
"""
from __future__ import annotations

import copy
import fcntl
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from research.comparison.protocol import (
    CONFIGS_PER_CELL,
    NUM_CELLS,
    ComparisonSpecError,
    ModelProfile,
    canonical_protocol,
    identity_of,
)
from research.comparison.specs import CellSpec, InnerRunSpec, iter_inner_runs
from research.contracts.budget import ModelUsage
from research.contracts.persist import (
    PersistError,
    atomic_write_json,
    read_json_object,
)
from research.search.budget_transport import conservative_input_tokens

COMPARISON_LEDGER_SCHEMA = "istratde-comparison-ledger.v1"

_REQUEST_KINDS = ("main", "debug")


class ComparisonBudgetError(RuntimeError):
    """Base class for comparison ledger failures."""


class ComparisonLedgerLocked(ComparisonBudgetError):
    """Another process already holds this ledger (serial access)."""


class ComparisonLedgerStopped(ComparisonBudgetError):
    """The ledger is globally stopped; no new reservations."""


class GlobalBudgetExhausted(ComparisonBudgetError):
    """A global cap (requests/tokens/elapsed) is exhausted; ledger stopped."""


class CellBudgetExhausted(ComparisonBudgetError):
    """One cell's quota is exhausted; the cell is terminated truthfully."""


class InnerRequestLimitReached(ComparisonBudgetError):
    """Inner-run shape cap (6 main / run-level debug budget) reached; no mutation."""


class ComparisonFinalPhaseActive(ComparisonBudgetError):
    """The irreversible final phase is active; model reservations refuse.

    This is NOT a global failure: the ledger keeps its original clock and
    accounting so trusted final CPU scoring can still run. It only means no
    new model request may ever be reserved again.
    """


@dataclass(frozen=True)
class ProviderUsage:
    """Actual provider-reported usage for one settled request."""

    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if not _strict_int(self.input_tokens) or not _strict_int(
            self.output_tokens
        ):
            raise ComparisonBudgetError("usage_requires_nonnegative_integers")

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ComparisonReservation:
    """One open reservation: durable request identity + conservative tokens."""

    request_id: str
    run_identity: str
    cell_identity: str
    kind: str
    tokens: int
    created_at: float


def _strict_int(value: Any, *, positive: bool = False) -> bool:
    return type(value) is int and value >= (1 if positive else 0)


def _finite_nonneg(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _validate_cells(cells: Sequence[CellSpec]) -> tuple[CellSpec, ...]:
    """The approved roster: exactly 27 unique cells, one source commit."""
    if not isinstance(cells, (tuple, list)) or len(cells) != NUM_CELLS:
        raise ComparisonSpecError("ledger_cells_invalid")
    for cell in cells:
        if not isinstance(cell, CellSpec):
            raise ComparisonSpecError("ledger_cells_invalid")
    identities = [cell.identity() for cell in cells]
    if len(set(identities)) != NUM_CELLS:
        raise ComparisonSpecError("ledger_cells_duplicate")
    keys = [cell.cell_key() for cell in cells]
    if len(set(keys)) != NUM_CELLS:
        raise ComparisonSpecError("ledger_cells_duplicate")
    if len({cell.source_commit for cell in cells}) != 1:
        raise ComparisonSpecError("source_commit_inconsistent")
    return tuple(cells)


def comparison_experiment_identity(cells: Sequence[CellSpec]) -> str:
    """Content identity binding protocol, source commit and all cell/run ids."""
    roster = _validate_cells(cells)
    runs = iter_inner_runs(roster)
    return identity_of(
        {
            "schema": COMPARISON_LEDGER_SCHEMA,
            "cells": sorted(cell.identity() for cell in roster),
            "runs": sorted(run.identity() for run in runs),
        }
    )


def _pinned_limits() -> dict[str, Any]:
    proto = canonical_protocol()
    return {
        "global": proto.global_budget.to_dict(),
        "cell": proto.cell_budget.to_dict(),
        "inner": proto.inner_budget.to_dict(),
    }


def _atomic_write_durable(path: Path, payload: Any) -> None:
    """atomic_write_json PLUS a parent-directory fsync after the rename.

    The shared helper fsyncs the temp file before os.replace; the
    comparison ledger additionally syncs the containing directory so the
    rename itself is durable across an abrupt process/OS exit. The
    historical contracts helper is intentionally left unchanged.
    """
    atomic_write_json(path, payload)
    dir_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


class ComparisonLedger:
    """Persistent atomic global+cell budget ledger; open() never resets."""

    def __init__(
        self,
        path: Path,
        cells_by_identity: Mapping[str, CellSpec],
        runs_by_identity: Mapping[str, InnerRunSpec],
        now_fn: Callable[[], float],
        state: dict[str, Any],
        lock_handle: Any,
    ):
        self._path = Path(path)
        self._cells = dict(cells_by_identity)
        self._runs = dict(runs_by_identity)
        self._limits = _pinned_limits()
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
        cells: Sequence[CellSpec],
        now_fn: Callable[[], float] | None = None,
    ) -> "ComparisonLedger":
        """Open (or create) the comparison ledger at path.

        Existing state is continued, never reset: committed totals,
        started_at, the stopped flag and unresolved reservations survive.
        Reopening with a different experiment roster (source/protocol/cell/
        run identities) is refused without mutating the file; reopening
        with unresolved reservations stops the ledger globally.
        """
        roster = _validate_cells(cells)
        runs = iter_inner_runs(roster)
        cells_by_identity = {cell.identity(): cell for cell in roster}
        runs_by_identity = {run.identity(): run for run in runs}
        experiment = {
            "identity": comparison_experiment_identity(roster),
            "protocol_identity": roster[0].protocol_identity,
            "source_commit": roster[0].source_commit,
            "cells": sorted(cells_by_identity),
            "runs": sorted(runs_by_identity),
        }
        path = Path(path)
        clock = time.time if now_fn is None else now_fn
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = open(path.with_name(path.name + ".lock"), "a+b")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock_handle.close()
            raise ComparisonLedgerLocked(f"ledger_in_use:{path.name}") from exc
        try:
            if path.exists():
                state = cls._load(path)
                mismatch: str | None = None
                if state["experiment"] != experiment:
                    mismatch = "experiment_identity_mismatch"
                else:
                    # The identity hash matches; still verify every piece of
                    # identity metadata against the roster so a tampered
                    # cell_key/run_key/cell pairing cannot slip through.
                    for identity, cell in cells_by_identity.items():
                        if state["cells"][identity]["cell_key"] != cell.cell_key():
                            mismatch = "experiment_metadata_mismatch"
                    for identity, run in runs_by_identity.items():
                        run_state = state["runs"][identity]
                        if (
                            run_state["run_key"] != run.run_key()
                            or run_state["cell_identity"] != run.cell.identity()
                        ):
                            mismatch = "experiment_metadata_mismatch"
                if mismatch is not None:
                    # Source/identity mismatch is a GLOBAL stop condition and
                    # the stop must persist: a later correct reopen finds the
                    # ledger stopped, never silently usable.
                    if state["stopped"] is None:
                        stopped = {"reason": mismatch, "at": float(clock())}
                        state["stopped"] = stopped
                        state["events"].append(
                            {
                                "type": "stop",
                                "at": stopped["at"],
                                "reason": mismatch,
                            }
                        )
                        _atomic_write_durable(path, state)
                    raise ComparisonBudgetError(mismatch)
                if state["stopped"] is None and state["reservations"]:
                    stopped = {
                        "reason": "unresolved_reservations_on_restart",
                        "at": float(clock()),
                    }
                    state["stopped"] = stopped
                    state["events"].append(
                        {
                            "type": "stop",
                            "at": stopped["at"],
                            "reason": stopped["reason"],
                        }
                    )
                    _atomic_write_durable(path, state)
            else:
                state = {
                    "schema": COMPARISON_LEDGER_SCHEMA,
                    "experiment": experiment,
                    "limits": _pinned_limits(),
                    "started_at": None,
                    "stopped": None,
                    "final_phase": None,
                    "global_committed": {"requests": 0, "tokens": 0},
                    "cells": {
                        identity: {
                            "cell_key": cell.cell_key(),
                            "committed": {"requests": 0, "tokens": 0},
                            "terminated": None,
                        }
                        for identity, cell in cells_by_identity.items()
                    },
                    "runs": {
                        identity: {
                            "run_key": run.run_key(),
                            "cell_identity": run.cell.identity(),
                            "main_requests": 0,
                            "debug_requests": 0,
                            "debug_limit": None,
                        }
                        for identity, run in runs_by_identity.items()
                    },
                    "reservations": [],
                    "requests": [],
                    "events": [],
                }
                _atomic_write_durable(path, state)
        except Exception:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
            raise
        return cls(
            path, cells_by_identity, runs_by_identity, clock, state, lock_handle
        )

    # ------------------------------------------------------------- load gate
    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        """Read and FULLY validate persisted state; corruption fails closed."""
        try:
            state = read_json_object(path)
        except PersistError as exc:
            raise ComparisonBudgetError(f"corrupt_ledger:{exc}") from exc
        if state.get("schema") != COMPARISON_LEDGER_SCHEMA:
            raise ComparisonBudgetError("corrupt_ledger:schema")
        experiment = state.get("experiment")
        if (
            not isinstance(experiment, dict)
            or not isinstance(experiment.get("identity"), str)
            or len(experiment["identity"]) != 64
            or not isinstance(experiment.get("protocol_identity"), str)
            or not isinstance(experiment.get("source_commit"), str)
            or not isinstance(experiment.get("cells"), list)
            or len(experiment["cells"]) != NUM_CELLS
            or len(set(experiment["cells"])) != NUM_CELLS
            or any(not isinstance(c, str) for c in experiment["cells"])
            or not isinstance(experiment.get("runs"), list)
            or len(experiment["runs"]) != NUM_CELLS * CONFIGS_PER_CELL
            or len(set(experiment["runs"])) != NUM_CELLS * CONFIGS_PER_CELL
            or any(not isinstance(r, str) for r in experiment["runs"])
        ):
            raise ComparisonBudgetError("corrupt_ledger:experiment")
        if state.get("limits") != _pinned_limits():
            raise ComparisonBudgetError("corrupt_ledger:limits")
        started_at = state.get("started_at")
        if started_at is not None and not _finite_nonneg(started_at):
            raise ComparisonBudgetError("corrupt_ledger:started_at")
        stopped = state.get("stopped")
        if stopped is not None and (
            not isinstance(stopped, dict)
            or not isinstance(stopped.get("reason"), str)
            or not stopped["reason"]
            or not _finite_nonneg(stopped.get("at"))
        ):
            raise ComparisonBudgetError("corrupt_ledger:stopped")
        if "final_phase" not in state:
            # The phase field is part of the ledger schema from creation; a
            # missing key is tampering, not a pre-final ledger.
            raise ComparisonBudgetError("corrupt_ledger:final_phase")
        final_phase = state["final_phase"]
        if final_phase is not None:
            barrier_identity = (
                final_phase.get("barrier_identity")
                if isinstance(final_phase, dict)
                else None
            )
            if (
                not isinstance(barrier_identity, str)
                or len(barrier_identity) != 64
                or any(c not in "0123456789abcdef" for c in barrier_identity)
                or not _finite_nonneg(final_phase.get("at"))
                or started_at is None
                or (
                    _finite_nonneg(final_phase.get("at"))
                    and float(final_phase["at"]) < float(started_at)
                )
            ):
                raise ComparisonBudgetError("corrupt_ledger:final_phase")
        limits = _pinned_limits()
        g_max_req = int(limits["global"]["max_requests"])
        g_max_tok = int(limits["global"]["max_tokens"])
        c_max_req = int(limits["cell"]["max_requests"])
        c_max_tok = int(limits["cell"]["max_tokens"])
        i_max_main = int(limits["inner"]["max_main_requests"])
        i_max_debug = int(limits["inner"]["max_debug_requests"])

        global_committed = state.get("global_committed")
        if (
            not isinstance(global_committed, dict)
            or not _strict_int(global_committed.get("requests"))
            or not _strict_int(global_committed.get("tokens"))
        ):
            raise ComparisonBudgetError("corrupt_ledger:global_committed")
        # Totals beyond a cap are valid ONLY as truthfully stopped overage
        # evidence (the overage commit itself stops the ledger); the same
        # numbers without a persisted stop are inconsistent.
        if stopped is None and (
            global_committed["requests"] > g_max_req
            or global_committed["tokens"] > g_max_tok
        ):
            raise ComparisonBudgetError("corrupt_ledger:inconsistent")

        cells = state.get("cells")
        if not isinstance(cells, dict) or set(cells) != set(experiment["cells"]):
            raise ComparisonBudgetError("corrupt_ledger:cells")
        for cell_id, cell_state in cells.items():
            committed = cell_state.get("committed") if isinstance(
                cell_state, dict
            ) else None
            terminated = (
                cell_state.get("terminated") if isinstance(cell_state, dict)
                else None
            )
            if (
                not isinstance(cell_state.get("cell_key"), str)
                or not isinstance(committed, dict)
                or not _strict_int(committed.get("requests"))
                or not _strict_int(committed.get("tokens"))
                or not (terminated is None or isinstance(terminated, dict))
                or (
                    isinstance(terminated, dict)
                    and (
                        not isinstance(terminated.get("reason"), str)
                        or not terminated["reason"]
                        or not _finite_nonneg(terminated.get("at"))
                    )
                )
            ):
                raise ComparisonBudgetError("corrupt_ledger:cell")
        if stopped is None and any(
            cell_state["committed"]["requests"] > c_max_req
            or cell_state["committed"]["tokens"] > c_max_tok
            for cell_state in cells.values()
        ):
            raise ComparisonBudgetError("corrupt_ledger:inconsistent")

        runs = state.get("runs")
        if not isinstance(runs, dict) or set(runs) != set(experiment["runs"]):
            raise ComparisonBudgetError("corrupt_ledger:runs")
        for run_id, run_state in runs.items():
            if not isinstance(run_state, dict):
                raise ComparisonBudgetError("corrupt_ledger:run")
            debug_limit = run_state.get("debug_limit")
            if (
                not isinstance(run_state.get("run_key"), str)
                or run_state.get("cell_identity") not in cells
                or not _strict_int(run_state.get("main_requests"))
                or not _strict_int(run_state.get("debug_requests"))
                or run_state["main_requests"] > i_max_main
                or run_state["debug_requests"] > i_max_debug
                or not (
                    debug_limit is None
                    or (
                        _strict_int(debug_limit) and debug_limit <= i_max_debug
                    )
                )
                or (
                    _strict_int(debug_limit)
                    and run_state["debug_requests"] > debug_limit
                )
            ):
                raise ComparisonBudgetError("corrupt_ledger:run")

        reservations = state.get("reservations")
        requests = state.get("requests")
        events = state.get("events")
        if (
            not isinstance(reservations, list)
            or not isinstance(requests, list)
            or not isinstance(events, list)
        ):
            raise ComparisonBudgetError("corrupt_ledger:schema")
        reservation_ids: set[str] = set()
        for res in reservations:
            if (
                not isinstance(res, dict)
                or not isinstance(res.get("id"), str)
                or not res["id"]
                or res["id"] in reservation_ids
                or res.get("run_identity") not in runs
                or res.get("cell_identity") not in cells
                or res.get("kind") not in _REQUEST_KINDS
                or runs[res["run_identity"]]["cell_identity"]
                != res["cell_identity"]
                or not _strict_int(res.get("tokens"), positive=True)
                or not _finite_nonneg(res.get("created_at"))
            ):
                raise ComparisonBudgetError("corrupt_ledger:reservation")
            reservation_ids.add(res["id"])

        request_ids: set[str] = set()
        provider_ids: set[str] = set()
        for rec in requests:
            if not isinstance(rec, dict):
                raise ComparisonBudgetError("corrupt_ledger:request")
            provider_id = rec.get("provider_request_id")
            if (
                not isinstance(rec.get("request_id"), str)
                or not rec["request_id"]
                or rec["request_id"] in request_ids
                or rec["request_id"] in reservation_ids
                or rec.get("run_identity") not in runs
                or rec.get("cell_identity") not in cells
                or runs[rec["run_identity"]]["cell_identity"]
                != rec["cell_identity"]
                or rec.get("kind") not in _REQUEST_KINDS
                or not _strict_int(rec.get("reserved_tokens"), positive=True)
                # charged may be 0 (a provider may legitimately report zero
                # usage) and MAY exceed caps/reservation: an over-reservation
                # commit is recorded truthfully and stops the ledger, so the
                # reopened state is stopped, never "corrupt".
                or not _strict_int(rec.get("charged_tokens"))
                or type(rec.get("usage_known")) is not bool
                or not _finite_nonneg(rec.get("reserved_at"))
                or not _finite_nonneg(rec.get("committed_at"))
                or not (
                    provider_id is None
                    or (isinstance(provider_id, str) and provider_id)
                )
            ):
                raise ComparisonBudgetError("corrupt_ledger:request")
            if rec["usage_known"]:
                if (
                    not _strict_int(rec.get("input_tokens"))
                    or not _strict_int(rec.get("output_tokens"))
                    or rec["input_tokens"] + rec["output_tokens"]
                    != rec["charged_tokens"]
                ):
                    raise ComparisonBudgetError("corrupt_ledger:request")
            elif rec["charged_tokens"] != rec["reserved_tokens"]:
                raise ComparisonBudgetError("corrupt_ledger:request")
            if isinstance(provider_id, str):
                if provider_id in provider_ids:
                    raise ComparisonBudgetError("corrupt_ledger:request")
                provider_ids.add(provider_id)
            request_ids.add(rec["request_id"])
        if stopped is None and any(
            (not rec["usage_known"])
            or rec["charged_tokens"] > rec["reserved_tokens"]
            for rec in requests
        ):
            # Unknown-usage and over-reservation records are terminal
            # evidence: they exist only alongside a persisted global stop.
            # The same records with the stop flag removed are tampering —
            # fail closed instead of silently resuming.
            raise ComparisonBudgetError(
                "corrupt_ledger:unstopped_terminal_record"
            )

        # Cross-consistency: settled records must reproduce every counter.
        def _settled(pred: Callable[[Mapping[str, Any]], bool]) -> list[Any]:
            return [rec for rec in requests if pred(rec)]

        if global_committed["requests"] != len(requests) or global_committed[
            "tokens"
        ] != sum(int(rec["charged_tokens"]) for rec in requests):
            raise ComparisonBudgetError("corrupt_ledger:inconsistent")
        for cell_id, cell_state in cells.items():
            cell_recs = _settled(lambda r: r["cell_identity"] == cell_id)
            if cell_state["committed"]["requests"] != len(
                cell_recs
            ) or cell_state["committed"]["tokens"] != sum(
                int(rec["charged_tokens"]) for rec in cell_recs
            ):
                raise ComparisonBudgetError("corrupt_ledger:inconsistent")
        for run_id, run_state in runs.items():
            run_recs = _settled(lambda r: r["run_identity"] == run_id)
            if run_state["main_requests"] != sum(
                1 for rec in run_recs if rec["kind"] == "main"
            ) or run_state["debug_requests"] != sum(
                1 for rec in run_recs if rec["kind"] == "debug"
            ):
                raise ComparisonBudgetError("corrupt_ledger:inconsistent")

        # Open reservations alone must fit inside every cap. Committed totals
        # are deliberately NOT part of these bounds: a truthfully recorded
        # over-reservation commit pushes them past the cap and stops the
        # ledger, and that stopped state must reopen as stopped, not corrupt.
        open_tokens = sum(int(res["tokens"]) for res in reservations)
        if len(reservations) > g_max_req or open_tokens > g_max_tok:
            raise ComparisonBudgetError("corrupt_ledger:inconsistent")
        for cell_id in cells:
            cell_open = [r for r in reservations if r["cell_identity"] == cell_id]
            if len(cell_open) > c_max_req or sum(
                int(r["tokens"]) for r in cell_open
            ) > c_max_tok:
                raise ComparisonBudgetError("corrupt_ledger:inconsistent")
        for run_id, run_state in runs.items():
            run_open = [r for r in reservations if r["run_identity"] == run_id]
            open_main = sum(1 for r in run_open if r["kind"] == "main")
            open_debug = sum(1 for r in run_open if r["kind"] == "debug")
            if run_state["main_requests"] + open_main > i_max_main or (
                run_state["debug_requests"] + open_debug > i_max_debug
            ):
                raise ComparisonBudgetError("corrupt_ledger:inconsistent")

        for event in events:
            if (
                not isinstance(event, dict)
                or not isinstance(event.get("type"), str)
                or not event["type"]
                or not _finite_nonneg(event.get("at"))
            ):
                raise ComparisonBudgetError("corrupt_ledger:event")
        # The durable final-phase transition is IRREVERSIBLE: the state field
        # and its transition event must agree exactly. A removed/reset field
        # with the event still present is a rollback attempt — refuse the
        # whole record rather than silently reopening search reservations.
        final_events = [event for event in events if event["type"] == "final_phase"]
        if final_phase is None:
            if final_events:
                raise ComparisonBudgetError("corrupt_ledger:final_phase")
        else:
            if len(final_events) != 1:
                raise ComparisonBudgetError("corrupt_ledger:final_phase")
            transition = final_events[0]
            if (
                transition.get("barrier_identity") != final_phase["barrier_identity"]
                or transition["at"] != final_phase["at"]
            ):
                raise ComparisonBudgetError("corrupt_ledger:final_phase")
        return state

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        if not self._closed:
            self._closed = True
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()

    @property
    def path(self) -> Path:
        """The ledger file path (orchestration lock handoff, EXP-006)."""
        return self._path

    @property
    def now_fn(self) -> Callable[[], float]:
        """The clock function bound at open (reopen must reuse it)."""
        return self._now_fn

    @property
    def closed(self) -> bool:
        """Whether this handle has been closed (orchestration handoff)."""
        return self._closed

    def __enter__(self) -> "ComparisonLedger":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --------------------------------------------------------------- helpers
    def _now(self) -> float:
        current = self._now_fn()
        started = self._state["started_at"]
        if (
            not _finite_nonneg(current)
            or (started is not None and current < started)
        ):
            raise ComparisonBudgetError("invalid_or_rolled_back_clock")
        return float(current)

    def _require_open(self) -> None:
        if self._closed:
            raise ComparisonBudgetError("ledger_closed")

    def _persist_locked(self) -> None:
        _atomic_write_durable(self._path, self._state)

    def _event_locked(self, event_type: str, **fields: Any) -> None:
        self._state["events"].append(
            {"type": event_type, "at": self._now(), **fields}
        )

    def _stop_locked(self, reason: str) -> None:
        """Global stop: the truthful INCOMPLETE record. First stop wins."""
        if self._state["stopped"] is None:
            stopped = {"reason": reason, "at": self._now()}
            self._state["stopped"] = stopped
            self._state["events"].append(
                {"type": "stop", "at": stopped["at"], "reason": reason}
            )

    def _terminate_cell_locked(self, cell_identity: str, reason: str) -> None:
        cell_state = self._state["cells"][cell_identity]
        if cell_state["terminated"] is None:
            terminated = {"reason": reason, "at": self._now()}
            cell_state["terminated"] = terminated
            self._state["events"].append(
                {
                    "type": "cell_terminated",
                    "at": terminated["at"],
                    "cell_key": cell_state["cell_key"],
                    "reason": reason,
                }
            )

    # -------------------------------------------------------------- identity
    @property
    def experiment_identity(self) -> str:
        return str(self._state["experiment"]["identity"])

    @property
    def started_at(self) -> float | None:
        started = self._state["started_at"]
        return None if started is None else float(started)

    @property
    def stopped(self) -> bool:
        return self._state["stopped"] is not None

    @property
    def final_phase(self) -> dict[str, Any] | None:
        """The durable final-phase transition record, or None.

        Once entered (after the FULL verified all-cell search barrier), the
        phase is irreversible and bound to that barrier's identity across
        reopen/resume; the original clock and accounting are preserved.
        """
        phase = self._state["final_phase"]
        return None if phase is None else dict(phase)

    @property
    def deadline_epoch(self) -> float | None:
        """Absolute experiment deadline; None until the clock starts.

        Never recomputed from ``time.time()``: it is anchored to the
        persisted started_at, so resume can never extend it.
        """
        started = self._state["started_at"]
        if started is None:
            return None
        return float(started) + float(
            self._limits["global"]["max_elapsed_seconds"]
        )

    # ------------------------------------------------------------------ clock
    def start_clock(self) -> None:
        """Arm the experiment clock exactly once (supervised live start)."""
        self._require_open()
        with self._mutex:
            if self._state["started_at"] is not None:
                raise ComparisonBudgetError("clock_already_started")
            if self._state["final_phase"] is not None:
                raise ComparisonBudgetError("final_phase_active")
            self._state["started_at"] = self._now()
            self._event_locked("start_clock")
            self._persist_locked()

    # ---------------------------------------------------------- final phase
    def enter_final_phase(self, barrier_identity: str) -> None:
        """Durably enter the irreversible final phase, bound to a barrier.

        Called ONLY after the FULL verified all-27-cell search barrier and
        BEFORE any final input is copied. Afterwards every model
        reservation/generation entry refuses (``reserve`` raises
        ComparisonFinalPhaseActive without charging or mutating anything) —
        including for an unused invalid/skipped config or an alternative
        fresh run directory. This is NOT a global stop: the ledger keeps
        its original clock, deadline and accounting so trusted final CPU
        scoring can still proceed. Re-entering with the SAME barrier is an
        idempotent no-op (crash-safe resume); a different barrier is an
        integrity failure.
        """
        self._require_open()
        if (
            not isinstance(barrier_identity, str)
            or len(barrier_identity) != 64
            or any(c not in "0123456789abcdef" for c in barrier_identity)
        ):
            raise ComparisonBudgetError("final_phase_barrier_invalid")
        with self._mutex:
            stopped = self._state["stopped"]
            if stopped is not None:
                raise ComparisonLedgerStopped(f"stopped:{stopped['reason']}")
            if self._state["started_at"] is None:
                raise ComparisonBudgetError("clock_not_started")
            existing = self._state["final_phase"]
            if existing is not None:
                if existing["barrier_identity"] == barrier_identity:
                    return  # idempotent resume of the same transition
                raise ComparisonBudgetError("final_phase_barrier_mismatch")
            phase = {"barrier_identity": barrier_identity, "at": self._now()}
            self._state["final_phase"] = phase
            # State and event describe one transition. Reading the clock
            # twice makes real clocks violate the loader's exact binding.
            self._state["events"].append({"type": "final_phase", **phase})
            self._persist_locked()

    # --------------------------------------------------------------- snapshot
    @staticmethod
    def _snapshot_from_state(
        state: dict[str, Any],
        limits: Mapping[str, Any],
        now_fn: Callable[[], float],
    ) -> dict[str, Any]:
        """Deep-copied accounting view derived from a VALIDATED state."""
        state = copy.deepcopy(state)
        g_limits = limits["global"]
        c_limits = limits["cell"]
        reservations = state["reservations"]
        open_tokens = sum(int(r["tokens"]) for r in reservations)
        started = state["started_at"]
        elapsed = None if started is None else max(0.0, now_fn() - started)
        cells_view = {}
        for cell_id, cell_state in state["cells"].items():
            cell_open = [r for r in reservations if r["cell_identity"] == cell_id]
            cell_open_tokens = sum(int(r["tokens"]) for r in cell_open)
            cells_view[cell_state["cell_key"]] = {
                "cell_identity": cell_id,
                "committed": dict(cell_state["committed"]),
                "open_reservations": len(cell_open),
                "open_tokens": cell_open_tokens,
                "terminated": copy.deepcopy(cell_state["terminated"]),
                "remaining": {
                    "requests": c_limits["max_requests"]
                    - cell_state["committed"]["requests"]
                    - len(cell_open),
                    "tokens": c_limits["max_tokens"]
                    - cell_state["committed"]["tokens"]
                    - cell_open_tokens,
                },
            }
        deadline = None
        if started is not None:
            deadline = float(started) + float(g_limits["max_elapsed_seconds"])
        return {
            "schema": state["schema"],
            "experiment_identity": state["experiment"]["identity"],
            "started_at": state["started_at"],
            "deadline_epoch": deadline,
            "elapsed_seconds": elapsed,
            "stopped": state["stopped"],
            "final_phase": copy.deepcopy(state["final_phase"]),
            "limits": state["limits"],
            "global": {
                "committed": dict(state["global_committed"]),
                "open_reservations": len(reservations),
                "open_tokens": open_tokens,
                "remaining": {
                    "requests": g_limits["max_requests"]
                    - state["global_committed"]["requests"]
                    - len(reservations),
                    "tokens": g_limits["max_tokens"]
                    - state["global_committed"]["tokens"]
                    - open_tokens,
                    "elapsed_seconds": (
                        None
                        if elapsed is None
                        else g_limits["max_elapsed_seconds"] - elapsed
                    ),
                },
            },
            "cells": cells_view,
            "reservations": reservations,
            "requests": state["requests"],
            "events": state["events"],
        }

    def snapshot(self) -> dict[str, Any]:
        """Deep-copied accounting view; mutating it never affects the ledger."""
        self._require_open()
        with self._mutex:
            self._now()  # rolled-back/invalid clocks fail closed here too
            return self._snapshot_from_state(self._state, self._limits, self._now)

    # -------------------------------------------------------------- mutation
    def reserve(
        self,
        run: InnerRunSpec,
        kind: str,
        *,
        estimated_input_tokens: int,
        max_output_tokens: int = ModelProfile().max_output_tokens,
        debug_limit: int | None = None,
    ) -> ComparisonReservation:
        """Atomically debit global AND cell quota before ONE model attempt.

        ``kind`` is "main" or "debug". Debug reservations require the
        run-level TOTAL debug budget ``debug_limit`` (0 disables debug for
        the run; otherwise the pinned inner cap 2). This is deliberately
        NOT theta's per-candidate repair-chain depth: theta depth 1 or 2
        both map to the same total budget of 2 debug requests per inner
        run (EXP-005 supervisor clarification). The first debug
        reservation of a run persists it and later mismatches fail
        closed. Raises CellBudgetExhausted for truthful per-cell
        termination, GlobalBudgetExhausted/ComparisonLedgerStopped for
        global stops, InnerRequestLimitReached for inner shape caps.
        """
        self._require_open()
        if not isinstance(run, InnerRunSpec):
            raise ComparisonBudgetError("inner_run_spec_invalid")
        if kind not in _REQUEST_KINDS:
            raise ComparisonBudgetError("request_kind_invalid")
        if not _strict_int(estimated_input_tokens, positive=True):
            raise ComparisonBudgetError("invalid_estimate")
        if (
            not _strict_int(max_output_tokens, positive=True)
            or max_output_tokens != ModelProfile().max_output_tokens
        ):
            raise ComparisonBudgetError("max_output_tokens_not_pinned")
        if kind == "debug":
            if debug_limit is None:
                raise ComparisonBudgetError("debug_limit_required")
            if not _strict_int(debug_limit) or debug_limit > int(
                self._limits["inner"]["max_debug_requests"]
            ):
                raise ComparisonBudgetError("debug_limit_out_of_range")
        run_identity = run.identity()
        cell_identity = run.cell.identity()
        with self._mutex:
            runs = self._state["runs"]
            if run_identity not in runs or (
                runs[run_identity]["cell_identity"] != cell_identity
            ):
                raise ComparisonBudgetError("unknown_inner_run")
            stopped = self._state["stopped"]
            if stopped is not None:
                raise ComparisonLedgerStopped(f"stopped:{stopped['reason']}")
            if self._state["final_phase"] is not None:
                # The final phase is irreversible: NO model reservation is
                # possible afterwards, for ANY run of the roster (including
                # unused invalid/skipped configs). No charge, no mutation,
                # and — deliberately — no global stop: trusted final CPU
                # scoring still runs under the original clock.
                raise ComparisonFinalPhaseActive("final_phase_active")
            started = self._state["started_at"]
            if started is None:
                raise ComparisonBudgetError("clock_not_started")
            now = self._now()
            if now - started >= float(
                self._limits["global"]["max_elapsed_seconds"]
            ):
                self._stop_locked("global_exhausted:elapsed_seconds")
                self._persist_locked()
                raise GlobalBudgetExhausted("elapsed_seconds_exhausted")
            cell_state = self._state["cells"][cell_identity]
            if cell_state["terminated"] is not None:
                raise CellBudgetExhausted(
                    f"cell_terminated:{cell_state['terminated']['reason']}"
                )

            run_state = runs[run_identity]
            reservations = self._state["reservations"]
            run_open = [
                r for r in reservations if r["run_identity"] == run_identity
            ]
            open_main = sum(1 for r in run_open if r["kind"] == "main")
            open_debug = sum(1 for r in run_open if r["kind"] == "debug")
            if kind == "main":
                if run_state["main_requests"] + open_main >= int(
                    self._limits["inner"]["max_main_requests"]
                ):
                    raise InnerRequestLimitReached(
                        "inner_main_requests_exhausted"
                    )
            else:
                assert debug_limit is not None  # validated above
                persisted_limit = run_state["debug_limit"]
                if persisted_limit is not None and persisted_limit != debug_limit:
                    raise ComparisonBudgetError("debug_limit_mismatch")
                if run_state["debug_requests"] + open_debug >= debug_limit:
                    raise InnerRequestLimitReached("inner_debug_limit_reached")

            needed = estimated_input_tokens + max_output_tokens
            global_committed = self._state["global_committed"]
            open_tokens = sum(int(r["tokens"]) for r in reservations)
            if global_committed["requests"] + len(reservations) + 1 > int(
                self._limits["global"]["max_requests"]
            ):
                self._stop_locked("global_exhausted:requests")
                self._persist_locked()
                raise GlobalBudgetExhausted("requests_exhausted")
            if global_committed["tokens"] + open_tokens + needed > int(
                self._limits["global"]["max_tokens"]
            ):
                self._stop_locked("global_exhausted:tokens")
                self._persist_locked()
                raise GlobalBudgetExhausted("tokens_exhausted")
            cell_open = [
                r for r in reservations if r["cell_identity"] == cell_identity
            ]
            cell_open_tokens = sum(int(r["tokens"]) for r in cell_open)
            if cell_state["committed"]["requests"] + len(cell_open) + 1 > int(
                self._limits["cell"]["max_requests"]
            ):
                self._terminate_cell_locked(
                    cell_identity, "cell_requests_exhausted"
                )
                self._persist_locked()
                raise CellBudgetExhausted("cell_requests_exhausted")
            if cell_state["committed"]["tokens"] + cell_open_tokens + needed > (
                int(self._limits["cell"]["max_tokens"])
            ):
                self._terminate_cell_locked(
                    cell_identity, "cell_tokens_exhausted"
                )
                self._persist_locked()
                raise CellBudgetExhausted("cell_tokens_exhausted")

            reservation = ComparisonReservation(
                request_id=uuid.uuid4().hex,
                run_identity=run_identity,
                cell_identity=cell_identity,
                kind=kind,
                tokens=needed,
                created_at=now,
            )
            if kind == "debug" and run_state["debug_limit"] is None:
                # Persist the run-level debug budget only WITH the reservation
                # it authorizes (one atomic write); a refused debug attempt
                # never mutates state.
                run_state["debug_limit"] = debug_limit
            reservations.append(
                {
                    "id": reservation.request_id,
                    "run_identity": run_identity,
                    "cell_identity": cell_identity,
                    "kind": kind,
                    "tokens": needed,
                    "created_at": now,
                }
            )
            self._event_locked(
                "reserve",
                request_id=reservation.request_id,
                run_key=run.run_key(),
                kind=kind,
                tokens=needed,
            )
            self._persist_locked()
            return reservation

    def commit(
        self,
        request_id: str,
        usage: ProviderUsage | ModelUsage | None,
        *,
        provider_request_id: str | None = None,
    ) -> None:
        """Settle one reservation exactly once with provider usage.

        usage=None (unknown usage: transport failure, missing/invalid
        provider usage) retains the full reservation as permanently charged
        AND stops the ledger globally. Actual usage charges input+output;
        usage above the reservation commits, then stops the ledger
        (``usage_exceeded_reservation``). Double or mismatched commits raise
        ``unknown_reservation``.
        """
        self._require_open()
        if not isinstance(request_id, str) or not request_id:
            raise ComparisonBudgetError("request_id_invalid")
        if provider_request_id is not None and (
            not isinstance(provider_request_id, str) or not provider_request_id
        ):
            raise ComparisonBudgetError("provider_request_id_invalid")
        if usage is not None and not isinstance(usage, (ProviderUsage, ModelUsage)):
            raise ComparisonBudgetError("usage_type_invalid")
        with self._mutex:
            reservations = self._state["reservations"]
            index = next(
                (i for i, r in enumerate(reservations) if r["id"] == request_id),
                None,
            )
            if index is None:
                raise ComparisonBudgetError(f"unknown_reservation:{request_id}")
            if provider_request_id is not None and any(
                rec["provider_request_id"] == provider_request_id
                for rec in self._state["requests"]
            ):
                raise ComparisonBudgetError("duplicate_provider_request_id")
            open_res = reservations.pop(index)
            now = self._now()
            charged = int(open_res["tokens"])
            input_tokens: int | None = None
            output_tokens: int | None = None
            if usage is not None:
                input_tokens = int(usage.input_tokens)
                output_tokens = int(usage.output_tokens)
                charged = input_tokens + output_tokens
            cell_identity = str(open_res["cell_identity"])
            run_identity = str(open_res["run_identity"])
            global_committed = self._state["global_committed"]
            global_committed["requests"] += 1
            global_committed["tokens"] += charged
            cell_committed = self._state["cells"][cell_identity]["committed"]
            cell_committed["requests"] += 1
            cell_committed["tokens"] += charged
            run_state = self._state["runs"][run_identity]
            if open_res["kind"] == "main":
                run_state["main_requests"] += 1
            else:
                run_state["debug_requests"] += 1
            self._state["requests"].append(
                {
                    "request_id": request_id,
                    "run_identity": run_identity,
                    "cell_identity": cell_identity,
                    "kind": open_res["kind"],
                    "reserved_tokens": int(open_res["tokens"]),
                    "charged_tokens": charged,
                    "usage_known": usage is not None,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "provider_request_id": provider_request_id,
                    "reserved_at": float(open_res["created_at"]),
                    "committed_at": now,
                }
            )
            self._event_locked(
                "commit",
                request_id=request_id,
                kind=open_res["kind"],
                reserved_tokens=int(open_res["tokens"]),
                charged_tokens=charged,
                usage_known=usage is not None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if usage is None:
                self._stop_locked(f"unknown_usage:{request_id}")
            elif charged > int(open_res["tokens"]):
                self._stop_locked("usage_exceeded_reservation")
            self._persist_locked()

    def terminate_cell(self, cell_identity: str, reason: str) -> None:
        """Truthfully terminate ONE cell (persisted; survives resume).

        Used when the run floor detects the cell can no longer fund its
        required main requests; other cells continue and the ledger is NOT
        globally stopped. Idempotent (first termination reason wins).
        """
        self._require_open()
        if not isinstance(reason, str) or not reason:
            raise ComparisonBudgetError("terminate_reason_invalid")
        with self._mutex:
            if cell_identity not in self._state["cells"]:
                raise ComparisonBudgetError("unknown_cell")
            self._terminate_cell_locked(cell_identity, reason)
            self._persist_locked()

    def stop(self, reason: str) -> None:
        """Explicit global stop (e.g. cleanup or source identity failure)."""
        self._require_open()
        if not isinstance(reason, str) or not reason:
            raise ComparisonBudgetError("stop_reason_invalid")
        with self._mutex:
            self._stop_locked(reason)
            self._persist_locked()


# ------------------------------------------------------------ guarded call
Transport = Callable[[Any], Any]
UsageFn = Callable[[Any], ProviderUsage | ModelUsage | None]


def guarded_comparison_attempt(
    ledger: ComparisonLedger,
    run: InnerRunSpec,
    kind: str,
    transport: Transport,
    request: Any,
    *,
    estimated_input_tokens: int,
    usage_fn: UsageFn,
    debug_limit: int | None = None,
    on_reserve: Callable[[ComparisonReservation], None] | None = None,
    provider_request_id_fn: Callable[[Any], str | None] | None = None,
) -> Any:
    """Exactly ONE provider attempt under the comparison ledger contract.

    reserve (atomically debiting global+cell+inner) -> transport -> settle.
    The pinned profile has transport_retries=0, so there is no retry loop.
    The WHOLE attempt (reservation, provider call, settlement) runs under
    the ledger's attempt mutex: the protocol requires serial model
    execution, not just serialized reservations. A transport exception, an
    observer exception, a usage-parser exception or an invalid provider
    request identity means unknown/untrustworthy accounting: the
    reservation is retained as permanently charged, the ledger stops
    globally and the exception propagates. ``usage_fn`` is
    interface-compatible with
    ``research.search.budget_transport.extract_provider_usage``.

    This ledger must be the ONLY budget layer around the transport: never
    nest the same call in the US-009 RequestGuard/BudgetLedger (double
    charging).
    """
    with ledger._attempt_mutex:
        reservation = ledger.reserve(
            run,
            kind,
            estimated_input_tokens=estimated_input_tokens,
            debug_limit=debug_limit,
        )
        try:
            if on_reserve is not None:
                on_reserve(reservation)
            response = transport(request)
        except Exception:
            ledger.commit(reservation.request_id, None)
            raise
        try:
            usage = usage_fn(response)
            if usage is not None and not isinstance(
                usage, (ProviderUsage, ModelUsage)
            ):
                raise ComparisonBudgetError("invalid_usage_payload")
            provider_request_id = (
                None if provider_request_id_fn is None
                else provider_request_id_fn(response)
            )
            if provider_request_id is not None and (
                not isinstance(provider_request_id, str)
                or not provider_request_id
            ):
                # The transport already ran; an unusable provider identity
                # is settled as unknown usage (retained + global stop).
                raise ComparisonBudgetError("provider_request_id_invalid")
        except Exception:
            ledger.commit(reservation.request_id, None)
            raise
        try:
            ledger.commit(
                reservation.request_id,
                usage,
                provider_request_id=provider_request_id,
            )
        except ComparisonBudgetError:
            # Post-transport accounting failure (e.g. a duplicate provider
            # request identity): the provider may have consumed the call,
            # so settle as unknown usage — retained in full + global stop.
            ledger.commit(reservation.request_id, None)
            raise
        return response


class ReadOnlyComparisonLedger:
    """Read-only validated view of a persisted comparison ledger.

    Used by offline report/audit surfaces (EXP-008): the SAME full _load
    validation and roster identity/metadata binding as open(), but NEVER
    takes the flock, NEVER creates the file or lock sidecar and NEVER
    persists a stop — an identity mismatch or corruption simply raises.
    Safe against a concurrently running experiment and unable to mutate the
    original ledger, deadline, winners or accounting.
    """

    def __init__(
        self,
        path: Path,
        state: dict[str, Any],
        now_fn: Callable[[], float],
    ):
        self._path = Path(path)
        self._state = state
        self._limits = _pinned_limits()
        self._now_fn = now_fn

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        cells: Sequence[CellSpec],
        now_fn: Callable[[], float] | None = None,
    ) -> "ReadOnlyComparisonLedger":
        """Validate the persisted ledger against the approved roster.

        Raises ComparisonBudgetError on corruption or experiment
        identity/metadata mismatch; never writes anything.
        """
        roster = _validate_cells(cells)
        runs = iter_inner_runs(roster)
        cells_by_identity = {cell.identity(): cell for cell in roster}
        path = Path(path)
        if path.is_symlink() or not path.exists():
            raise ComparisonBudgetError(f"ledger_missing:{path.name}")
        state = ComparisonLedger._load(path)
        experiment = {
            "identity": comparison_experiment_identity(roster),
            "protocol_identity": roster[0].protocol_identity,
            "source_commit": roster[0].source_commit,
            "cells": sorted(cells_by_identity),
            "runs": sorted(run.identity() for run in runs),
        }
        if state["experiment"] != experiment:
            raise ComparisonBudgetError("experiment_identity_mismatch")
        for identity, cell in cells_by_identity.items():
            if state["cells"][identity]["cell_key"] != cell.cell_key():
                raise ComparisonBudgetError("experiment_metadata_mismatch")
        runs_by_identity = {run.identity(): run for run in runs}
        for identity, run in runs_by_identity.items():
            run_state = state["runs"][identity]
            if (
                run_state["run_key"] != run.run_key()
                or run_state["cell_identity"] != run.cell.identity()
            ):
                raise ComparisonBudgetError("experiment_metadata_mismatch")
        return cls(path, state, time.time if now_fn is None else now_fn)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def experiment_identity(self) -> str:
        return str(self._state["experiment"]["identity"])

    @property
    def started_at(self) -> float | None:
        started = self._state["started_at"]
        return None if started is None else float(started)

    @property
    def stopped(self) -> bool:
        return self._state["stopped"] is not None

    @property
    def final_phase(self) -> dict[str, Any] | None:
        phase = self._state["final_phase"]
        return None if phase is None else dict(phase)

    @property
    def deadline_epoch(self) -> float | None:
        """Absolute ORIGINAL deadline anchored to persisted started_at."""
        started = self._state["started_at"]
        if started is None:
            return None
        return float(started) + float(
            self._limits["global"]["max_elapsed_seconds"]
        )

    def snapshot(self) -> dict[str, Any]:
        """The same accounting view shape as ComparisonLedger.snapshot()."""
        return ComparisonLedger._snapshot_from_state(
            self._state, self._limits, self._now_fn
        )


__all__ = [
    "COMPARISON_LEDGER_SCHEMA",
    "CellBudgetExhausted",
    "ComparisonBudgetError",
    "ComparisonFinalPhaseActive",
    "ComparisonLedger",
    "ComparisonLedgerLocked",
    "ComparisonLedgerStopped",
    "ComparisonReservation",
    "GlobalBudgetExhausted",
    "InnerRequestLimitReached",
    "ProviderUsage",
    "ReadOnlyComparisonLedger",
    "comparison_experiment_identity",
    "conservative_input_tokens",
    "guarded_comparison_attempt",
]
