"""Comparison run control: inner budget floor + guarded search loop (EXP-005).

Distinct from the US-009 ``run_guarded_search``, which stops its ledger on
EVERY exception. The comparison distinguishes truthfully:

- ``CellBudgetExhausted`` / ``InnerRequestLimitReached``: the CELL or the
  inner run ends; the ledger is NOT globally stopped and the incumbent found
  so far stays valid evidence (other cells continue).
- ``GlobalBudgetExhausted`` / ``ComparisonLedgerStopped`` / deadline /
  unknown-usage / unexpected errors / cleanup failure: the ledger is
  stopped GLOBALLY (the truthful INCOMPLETE record).

The inner debug cap skips further OPTIONAL debug: the pre-chain denial and
the mid-chain wrapper never crash the run when the run-level debug budget
(2 total) is consumed — the candidate stays honestly invalid and the search
continues with the actual main candidates.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from research.comparison.budget import (
    CellBudgetExhausted,
    ComparisonBudgetError,
    ComparisonLedger,
    ComparisonLedgerStopped,
    GlobalBudgetExhausted,
    InnerRequestLimitReached,
)
from research.comparison.protocol import INNER_GENERATIONS, MAIN_CANDIDATES_PER_GENERATION
from research.comparison.specs import InnerRunSpec
from research.comparison.transport import install_comparison_guard
from research.contracts.budget import (
    BudgetError,
    CancellationToken,
    Cancelled,
)
from research.contracts.persist import atomic_write_json

CONTROL_SCHEMA = "comparison-budget-control.v1"
SEARCH_CONTROL_SCHEMA = "comparison-search-control.v1"

_REASON_LIMIT = 200

_TOTAL_MAIN = INNER_GENERATIONS * MAIN_CANDIDATES_PER_GENERATION
_INNER_DEBUG_CAP = 2


def comparison_termination_reason(exc: BaseException) -> str:
    """Bounded machine-readable reason; provider internals never persist.

    Our own budget/cancellation exceptions carry curated strings; anything
    else keeps just the type name, so credential-bearing or provider text
    never enters checkpoints/results.
    """
    name = type(exc).__name__
    if isinstance(
        exc,
        (
            Cancelled,
            BudgetError,
            ComparisonBudgetError,
        ),
    ):
        detail = str(exc).splitlines()[0][:_REASON_LIMIT] if str(exc) else ""
        return f"{name}:{detail}" if detail else name
    return name


class ComparisonMainBudgetControl:
    """Operator-aware budget floor for ONE comparison inner run.

    - ``require_main`` runs BEFORE every main operator's provider call. The
      frozen arithmetic (8 configs x (6 main + 2 debug) == 64 == the cell
      cap) structurally guarantees the six mains always fit; a floor
      shortfall is therefore an inconsistency and fails closed: cell-scoped
      shortfall terminates ONLY that cell (persisted), a global shortfall
      stops the ledger.
    - ``debug_denial_reason`` runs BEFORE each debug repair chain: debug is
      optional and needs ONE extra request of slack beyond the remaining
      main calls, must not exceed the run-level total debug budget (2; 0
      when theta's chain depth is 0), and every skip is persisted with its
      machine-readable reason. Skipping never fabricates code, scores,
      operator traces or provider reservations.
    """

    def __init__(
        self,
        ledger: ComparisonLedger,
        run: InnerRunSpec,
        *,
        debug_budget: int,
        control_log_path: Path,
    ):
        if type(debug_budget) is not int or not 0 <= debug_budget <= _INNER_DEBUG_CAP:
            raise ComparisonBudgetError("debug_budget_invalid")
        self._ledger = ledger
        self._run = run
        self._debug_budget = debug_budget
        self._main_done = 0
        self._debug_done = 0
        self._path = Path(control_log_path)

    @property
    def debug_budget(self) -> int:
        return self._debug_budget

    def _record(self, event: str, operator: str, reason: str) -> None:
        entry = {
            "schema": CONTROL_SCHEMA,
            "event": event,
            "run_key": self._run.run_key(),
            "operator": operator,
            "reason": reason,
            "main_done": self._main_done,
            "debug_done": self._debug_done,
            "recorded_at": time.time(),
        }
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _remaining(self) -> tuple[int, int]:
        """(cell remaining requests, global remaining requests)."""
        snapshot = self._ledger.snapshot()
        cell_key = self._run.cell.cell_key()
        cell_remaining = int(
            snapshot["cells"][cell_key]["remaining"]["requests"]
        )
        global_remaining = int(snapshot["global"]["remaining"]["requests"])
        return cell_remaining, global_remaining

    def main_needed(self) -> int:
        return _TOTAL_MAIN - self._main_done

    def require_main(self, operator: str) -> None:
        """Floor check BEFORE a main operator call (draft/improve/crossover)."""
        needed = self.main_needed()
        cell_remaining, global_remaining = self._remaining()
        if cell_remaining < needed:
            reason = (
                f"cell_main_budget_insufficient:operator={operator},"
                f"cell_remaining={cell_remaining},needed_main={needed}"
            )
            self._record("main_budget_insufficient", operator, reason)
            self._ledger.terminate_cell(
                self._run.cell.identity(), "cell_main_budget_insufficient"
            )
            raise CellBudgetExhausted("cell_main_budget_insufficient")
        if global_remaining < needed:
            reason = (
                f"global_main_budget_insufficient:operator={operator},"
                f"global_remaining={global_remaining},needed_main={needed}"
            )
            self._record("main_budget_insufficient", operator, reason)
            raise GlobalBudgetExhausted("global_main_budget_insufficient")
        self._main_done += 1

    def note_debug_attempt(self) -> None:
        """Count one actually-reserved debug attempt (called by the runner
        attempt sink; the ledger remains the authoritative accounting)."""
        self._debug_done += 1

    def debug_denial_reason(self) -> str | None:
        """None when an optional debug chain may start; else the skip reason."""
        if self._debug_budget == 0:
            return "debug_disabled:theta_max_debug_depth_0"
        if self._debug_done >= self._debug_budget:
            return (
                f"inner_debug_budget_exhausted:used={self._debug_done},"
                f"limit={self._debug_budget}"
            )
        needed = self.main_needed()
        cell_remaining, global_remaining = self._remaining()
        if cell_remaining < needed + 1:
            return (
                f"cell_budget_insufficient_for_optional_debug:"
                f"cell_remaining={cell_remaining},needed_main={needed}"
            )
        if global_remaining < needed + 1:
            return (
                f"global_budget_insufficient_for_optional_debug:"
                f"global_remaining={global_remaining},needed_main={needed}"
            )
        return None

    def record_debug_skipped(self, reason: str) -> None:
        """Persist the skip decision (no provider reservation was made)."""
        self._record("debug_skipped", "debug", reason)

    def record_debug_chain_capped(self) -> None:
        """Persist a mid-chain stop at the run-level debug budget."""
        self._record(
            "debug_chain_capped", "debug", "inner_debug_budget_exhausted_mid_chain"
        )


def run_comparison_search(
    solver: Any,
    task: Any,
    state: Any,
    *,
    ledger: ComparisonLedger,
    run: InnerRunSpec,
    token: CancellationToken,
    budget_control: ComparisonMainBudgetControl,
    cancel_live_jobs: Callable[[], list[dict[str, Any]]],
    checkpoint_path: Path,
    attempt_sink: Callable[..., None] | None = None,
    prepare_operator: Callable[[], None] | None = None,
    cooperative_stop: Callable[[], str | None] | None = None,
) -> str:
    """Drive the REAL solver under the comparison ledger; return the
    machine-readable termination reason (``completed`` on success).

    Cell/inner budget exhaustion ends ONLY this run (ledger stays open for
    other cells); global caps, unknown usage, source/identity failures and
    cleanup failures stop the ledger globally. The checkpoint persists only
    accounting/cancellation state; the solver journal stays in its
    configured ignored run directory.
    """
    if getattr(solver, "before_operator", None) is not None:
        raise ComparisonBudgetError("search_control_already_installed")
    if getattr(solver, "budget_control", None) is not None:
        raise ComparisonBudgetError("search_control_already_installed")
    mode = str(getattr(solver.cfg, "execution_mode", "generation"))
    if mode != "generation":
        raise ComparisonBudgetError("comparison_requires_serial_generation_mode")
    for name in (
        "draft", "improve", "debug", "crossover", "analyze", "rich_memory_summary"
    ):
        operator = getattr(solver, name + "_fn", None)
        if operator is None:
            continue
        args = getattr(operator, "args", ())
        client = getattr(args[0], "client", None) if args else None
        if client is None:
            raise ComparisonBudgetError("operator_client_unavailable:" + name)
        install_comparison_guard(
            client,
            ledger,
            token,
            run,
            operator=name,
            debug_budget=budget_control.debug_budget,
            attempt_sink=attempt_sink,
        )

    cell_key = run.cell.cell_key()

    def before_operator() -> None:
        token.check()
        snapshot = ledger.snapshot()
        if snapshot["stopped"] is not None:
            raise ComparisonLedgerStopped(str(snapshot["stopped"]["reason"]))
        terminated = snapshot["cells"][cell_key]["terminated"]
        if terminated is not None:
            raise CellBudgetExhausted(f"cell_terminated:{terminated['reason']}")
        if getattr(task, "stop_requested", False):
            raise Cancelled("task_stop_requested")
        if cooperative_stop is not None:
            stop_reason = cooperative_stop()
            if stop_reason is not None:
                raise Cancelled(f"cooperative_stop:{stop_reason}")
        if solver.state.current_step >= solver.cfg.step_limit:
            raise Cancelled("solver_step_limit")
        # SDK debug logging can include request options; auth is
        # runtime-only and must not enter config/journal/log artifacts.
        for logger_name in ("openai", "httpx", "httpcore", "LiteLLM", "litellm"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        if prepare_operator is not None:
            prepare_operator()

    solver.before_operator = before_operator
    # The REAL solver consults this at every main operator (require_main)
    # and before each optional debug chain (debug_denial_reason).
    solver.budget_control = budget_control

    # Mid-chain protection: when the run-level total debug budget is
    # consumed INSIDE a repair chain (chain depth 2 with one slot already
    # spent), the reservation refusal truncates the chain truthfully — the
    # candidate stays invalid — instead of crashing the run.
    original_debug_cycle = solver.debug_cycle

    def debug_cycle_capped(state: Any, task_: Any, buggy_node: Any) -> Any:
        try:
            return original_debug_cycle(state, task_, buggy_node)
        except InnerRequestLimitReached:
            budget_control.record_debug_chain_capped()
            return state, [buggy_node], None

    solver.debug_cycle = debug_cycle_capped

    termination = "completed"
    cleanup: list[dict[str, Any]] = []
    cleanup_error: str | None = None
    try:
        before_operator()
        solver.search(task, state)
    except (CellBudgetExhausted, InnerRequestLimitReached) as exc:
        # Truthful CELL/RUN-level termination: the ledger is NOT stopped;
        # the incumbent found so far remains valid evidence.
        termination = comparison_termination_reason(exc)
    except (GlobalBudgetExhausted, ComparisonLedgerStopped) as exc:
        termination = comparison_termination_reason(exc)
        token.stop("search_stopped:" + termination)
        ledger.stop("search_stopped:" + termination)
    except BaseException as exc:
        # Unknown outcome: fail closed GLOBALLY (never silently continue).
        termination = comparison_termination_reason(exc)
        token.stop("search_stopped:" + termination)
        ledger.stop("search_stopped:" + termination)
    finally:
        try:
            cleanup = cancel_live_jobs()
            if any(item.get("cleanup_verified") is not True for item in cleanup):
                cleanup_error = "sandbox_cleanup_unproven"
        except Exception as exc:
            cleanup_error = "cancel_failed:" + type(exc).__name__
        if cleanup_error is not None:
            token.stop(cleanup_error)
            ledger.stop(cleanup_error)
        try:
            solver.save_checkpoint()
        finally:
            atomic_write_json(checkpoint_path, {
                "schema": SEARCH_CONTROL_SCHEMA,
                "run_key": run.run_key(),
                "termination": termination,
                "token": token.to_dict(),
                "budget": ledger.snapshot(),
                "cancellation": cleanup,
                "cleanup_error": cleanup_error,
                "inner_generation": solver.state.current_generation,
                "journal_steps": solver.state.current_step,
            })
        if cleanup_error is not None:
            raise ComparisonBudgetError(cleanup_error)
    return termination
