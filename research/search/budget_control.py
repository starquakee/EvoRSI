"""Reserve current+future MAIN candidate requests inside one inner run.

Debug is OPTIONAL (max_debug_depth is an upper bound, never a required
minimum). With a tight request budget the acceptance must therefore reserve
the requests for the remaining MAIN candidates (draft/improve/crossover of
this run) plus the FUTURE main candidates of all later outer configs, and
spend slack only on optional debug attempts:

- ``require_main`` runs BEFORE every main operator's provider call: the call
  is allowed exactly when ``remaining == needed``; if the required main calls
  themselves cannot fit, the whole run stops with the machine reason
  ``remaining_budget_insufficient`` (fail closed, no wasted calls).
- ``debug_denial_reason`` runs BEFORE any optional debug attempt: a debug
  needs ONE extra request of slack beyond the reserved main calls, otherwise
  it is SKIPPED — ``record_debug_skipped`` persists a separate
  budget-control.jsonl event with the reason and the remaining/needed
  numbers, the candidate stays honestly invalid (no fabricated code, score,
  operator trace or provider reservation), and the search continues with the
  actual main candidates.

Every check reads the shared ledger snapshot (committed + open
reservations), so decisions follow the same accounting as the budget guard.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from research.contracts.budget import BudgetError, BudgetLedger

CONTROL_SCHEMA = "budget-control.v1"


class MainBudgetControl:
    """Operator-aware budget floor for one inner run (acceptance only)."""

    def __init__(
        self,
        ledger: BudgetLedger,
        *,
        total_main: int,
        future_main: int,
        control_log_path: Path,
        run_id: str,
    ):
        if total_main < 1 or future_main < 0:
            raise ValueError("main_budget_control_counts_invalid")
        self._ledger = ledger
        self._total_main = total_main
        self._future_main = future_main
        self._main_done = 0
        self._path = Path(control_log_path)
        self._run_id = run_id

    def _remaining(self) -> int:
        return int(self._ledger.snapshot()["remaining"]["requests"])

    def main_needed(self) -> int:
        """Remaining main calls of THIS run + all future outer configs."""
        return (self._total_main - self._main_done) + self._future_main

    def _record(self, event: str, operator: str, reason: str,
                remaining: int, needed: int) -> None:
        entry = {
            "schema": CONTROL_SCHEMA,
            "event": event,
            "run_id": self._run_id,
            "operator": operator,
            "reason": reason,
            "remaining_requests": remaining,
            "needed_main_requests": needed,
            "recorded_at": time.time(),
        }
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def require_main(self, operator: str) -> None:
        """Floor check BEFORE a main operator call (draft/improve/crossover).

        Allowed exactly at equality (remaining == needed); stops the run when
        the required main calls cannot fit.
        """
        remaining = self._remaining()
        needed = self.main_needed()
        if remaining < needed:
            reason = (
                f"remaining_budget_insufficient:operator={operator},"
                f"remaining={remaining},needed_main={needed}"
            )
            self._record("main_budget_insufficient", operator, reason,
                         remaining, needed)
            raise BudgetError("remaining_budget_insufficient")
        self._main_done += 1

    def debug_denial_reason(self) -> str | None:
        """None when an optional debug attempt may proceed; otherwise the
        machine-readable skip reason (debug needs ONE slack request beyond
        the reserved main calls)."""
        remaining = self._remaining()
        needed = self.main_needed()
        if remaining < needed + 1:
            return (
                "remaining_budget_insufficient_for_optional_debug:"
                f"remaining={remaining},needed_main={needed}"
            )
        return None

    def record_debug_skipped(self, reason: str) -> None:
        """Persist the skip decision (no provider reservation was made)."""
        self._record("debug_skipped", "debug", reason,
                     self._remaining(), self.main_needed())
