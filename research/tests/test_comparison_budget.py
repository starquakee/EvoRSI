"""Offline contract tests for research/comparison/budget.py (EXP-004).

Covers the atomic global+per-cell ledger: roster binding, one-time clock,
conservative reservations (full prompt estimate + pinned 8192 output),
inner shape caps (6 main + theta-capped debug), exact quota boundaries,
truthful cell termination vs global stop, fail-closed unknown usage /
crash-reopen / over-reservation / corruption handling, duplicate and
mismatched commits, thread races, and the single-attempt guarded transport
interface (no double charging). All ledgers are tmp-path fixtures; no real
model calls, no live ledger.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from research.comparison.budget import (
    CellBudgetExhausted,
    ComparisonBudgetError,
    ComparisonFinalPhaseActive,
    ComparisonLedger,
    ComparisonLedgerLocked,
    ComparisonLedgerStopped,
    ComparisonReservation,
    GlobalBudgetExhausted,
    InnerRequestLimitReached,
    ProviderUsage,
    _pinned_limits,
    comparison_experiment_identity,
    conservative_input_tokens,
    guarded_comparison_attempt,
)
from research.comparison.protocol import (
    CONFIGS_PER_CELL,
    NUM_CELLS,
    TASKS,
    CellBudget,
    ModelProfile,
    canonical_protocol,
    split_seed,
)
from research.comparison.protocol import TASKS_BY_ID
from research.comparison.specs import (
    CellSpec,
    InnerRunSpec,
    TaskBinding,
    iter_cells,
)

COMMIT = "0123456789abcdef0123456789abcdef01234567"
OTHER_COMMIT = "ffffffffffffffffffffffffffffffffffffffff"
PINNED_OUTPUT = ModelProfile().max_output_tokens

_HASHES = {
    "train_csv_sha256": "a" * 64,
    "validation_features_sha256": "b" * 64,
    "validation_labels_sha256": "c" * 64,
    "final_features_sha256": "d" * 64,
    "final_labels_sha256": "e" * 64,
}


def make_binding(task_id: str) -> TaskBinding:
    pin = TASKS_BY_ID[task_id]
    return TaskBinding(
        task_id=task_id,
        split_seed=split_seed(pin),
        rows=pin.rows,
        features=pin.features,
        **_HASHES,
    )


def make_cells(commit: str = COMMIT) -> tuple[CellSpec, ...]:
    bindings = {task.task_id: make_binding(task.task_id) for task in TASKS}
    return iter_cells(bindings, commit)


CELLS = make_cells()
RUNS = {cell: [cell.inner_run(i) for i in range(CONFIGS_PER_CELL)] for cell in CELLS}


class Clock:
    def __init__(self, start: float = 10_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def open_ledger(tmp_path: Path, clock: Clock, commit: str = COMMIT) -> ComparisonLedger:
    cells = CELLS if commit == COMMIT else make_cells(commit)
    return ComparisonLedger.open(tmp_path / "ledger.json", cells=cells, now_fn=clock)


def started_ledger(tmp_path: Path, clock: Clock) -> ComparisonLedger:
    ledger = open_ledger(tmp_path, clock)
    ledger.start_clock()
    return ledger


def commit_main(
    ledger: ComparisonLedger, run: InnerRunSpec, est: int = 10, in_t: int = 6, out_t: int = 4
) -> str:
    res = ledger.reserve(run, "main", estimated_input_tokens=est)
    ledger.commit(res.request_id, ProviderUsage(in_t, out_t))
    return res.request_id


class TestLifecycle:
    def test_create_binds_experiment_and_limits(self, tmp_path: Path) -> None:
        clock = Clock()
        with open_ledger(tmp_path, clock) as ledger:
            assert ledger.experiment_identity == comparison_experiment_identity(CELLS)
            snap = ledger.snapshot()
            assert snap["started_at"] is None
            assert snap["deadline_epoch"] is None
            assert snap["limits"] == _pinned_limits()
            assert len(snap["cells"]) == NUM_CELLS
            assert snap["global"]["committed"] == {"requests": 0, "tokens": 0}

    def test_reserve_refused_before_clock_starts(self, tmp_path: Path) -> None:
        clock = Clock()
        with open_ledger(tmp_path, clock) as ledger:
            with pytest.raises(ComparisonBudgetError, match="clock_not_started"):
                ledger.reserve(RUNS[CELLS[0]][0], "main", estimated_input_tokens=10)

    def test_clock_starts_once_and_never_resets(self, tmp_path: Path) -> None:
        clock = Clock()
        ledger = started_ledger(tmp_path, clock)
        assert ledger.started_at == 10_000.0
        deadline = ledger.deadline_epoch
        assert deadline == 10_000.0 + 86_400.0
        with pytest.raises(ComparisonBudgetError, match="clock_already_started"):
            ledger.start_clock()
        ledger.close()
        clock.now += 500.0
        with open_ledger(tmp_path, clock) as reopened:
            assert reopened.started_at == 10_000.0  # immutable across resume
            assert reopened.deadline_epoch == deadline
            with pytest.raises(ComparisonBudgetError, match="clock_already_started"):
                reopened.start_clock()

    def test_second_opener_refused_while_locked(self, tmp_path: Path) -> None:
        clock = Clock()
        with open_ledger(tmp_path, clock):
            with pytest.raises(ComparisonLedgerLocked, match="ledger_in_use"):
                open_ledger(tmp_path, clock)
        with open_ledger(tmp_path, clock):  # lock released by close
            pass

    def test_identity_mismatch_persists_global_stop(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock):
            pass
        with pytest.raises(ComparisonBudgetError, match="experiment_identity_mismatch"):
            open_ledger(tmp_path, clock, commit=OTHER_COMMIT)
        # The source mismatch stop is persisted: even a correct reopen finds
        # the ledger globally stopped.
        with open_ledger(tmp_path, clock) as ledger:
            assert ledger.stopped
            assert (
                ledger.snapshot()["stopped"]["reason"]
                == "experiment_identity_mismatch"
            )
            with pytest.raises(ComparisonLedgerStopped):
                ledger.reserve(RUNS[CELLS[0]][0], "main", estimated_input_tokens=10)

    def test_closed_ledger_refuses_operations(self, tmp_path: Path) -> None:
        clock = Clock()
        ledger = started_ledger(tmp_path, clock)
        ledger.close()
        with pytest.raises(ComparisonBudgetError, match="ledger_closed"):
            ledger.reserve(RUNS[CELLS[0]][0], "main", estimated_input_tokens=10)
        with pytest.raises(ComparisonBudgetError, match="ledger_closed"):
            ledger.snapshot()
        with pytest.raises(ComparisonBudgetError, match="ledger_closed"):
            ledger.stop("x")

    def test_rolled_back_clock_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            clock.now -= 1.0
            with pytest.raises(
                ComparisonBudgetError, match="invalid_or_rolled_back_clock"
            ):
                ledger.reserve(RUNS[CELLS[0]][0], "main", estimated_input_tokens=10)

    def test_explicit_stop_persists_and_first_stop_wins(self, tmp_path: Path) -> None:
        clock = Clock()
        ledger = started_ledger(tmp_path, clock)
        ledger.stop("cleanup_verification_failed")
        ledger.stop("second_reason")
        ledger.close()
        with open_ledger(tmp_path, clock) as reopened:
            snap = reopened.snapshot()
            assert snap["stopped"]["reason"] == "cleanup_verification_failed"
            with pytest.raises(ComparisonLedgerStopped, match="stopped:cleanup"):
                reopened.reserve(
                    RUNS[CELLS[0]][0], "main", estimated_input_tokens=10
                )


class TestReservationShape:
    def test_main_reserve_commit_records_full_usage(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            run = RUNS[CELLS[0]][0]
            res = ledger.reserve(run, "main", estimated_input_tokens=100)
            assert res.tokens == 100 + PINNED_OUTPUT
            ledger.commit(
                res.request_id, ProviderUsage(90, 50), provider_request_id="p-1"
            )
            snap = ledger.snapshot()
            assert snap["global"]["committed"] == {"requests": 1, "tokens": 140}
            cell_view = snap["cells"][CELLS[0].cell_key()]
            assert cell_view["committed"] == {"requests": 1, "tokens": 140}
            record = snap["requests"][0]
            assert record["request_id"] == res.request_id
            assert record["run_identity"] == run.identity()
            assert record["kind"] == "main"
            assert record["usage_known"] is True
            assert record["input_tokens"] == 90
            assert record["output_tokens"] == 50
            assert record["charged_tokens"] == 140
            assert record["reserved_tokens"] == 100 + PINNED_OUTPUT
            assert record["provider_request_id"] == "p-1"

    def test_unknown_inner_run_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            foreign = make_cells(OTHER_COMMIT)[0].inner_run(0)
            with pytest.raises(ComparisonBudgetError, match="unknown_inner_run"):
                ledger.reserve(foreign, "main", estimated_input_tokens=10)
            assert ledger.snapshot()["global"]["committed"]["requests"] == 0

    @pytest.mark.parametrize(
        ("kind", "est", "out", "rule"),
        [
            ("retry", 10, PINNED_OUTPUT, "request_kind_invalid"),
            ("main", 0, PINNED_OUTPUT, "invalid_estimate"),
            ("main", -5, PINNED_OUTPUT, "invalid_estimate"),
            ("main", True, PINNED_OUTPUT, "invalid_estimate"),
            ("main", 10.5, PINNED_OUTPUT, "invalid_estimate"),
            ("main", 10, 8191, "max_output_tokens_not_pinned"),
            ("main", 10, 8193, "max_output_tokens_not_pinned"),
        ],
    )
    def test_invalid_reserve_arguments_rejected(
        self, tmp_path: Path, kind: str, est: object, out: int, rule: str
    ) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            with pytest.raises(ComparisonBudgetError, match=rule):
                ledger.reserve(
                    RUNS[CELLS[0]][0],
                    kind,
                    estimated_input_tokens=est,  # type: ignore[arg-type]
                    max_output_tokens=out,
                )

    def test_debug_limit_persisted_and_enforced(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            run = RUNS[CELLS[0]][0]
            with pytest.raises(ComparisonBudgetError, match="debug_limit_required"):
                ledger.reserve(run, "debug", estimated_input_tokens=10)
            for bad in (3, -1, True, 1.5):
                with pytest.raises(
                    ComparisonBudgetError, match="debug_limit_out_of_range"
                ):
                    ledger.reserve(
                        run,
                        "debug",
                        estimated_input_tokens=10,
                        debug_limit=bad,  # type: ignore[arg-type]
                    )
            # theta with max_debug_depth 0: no debug request allowed
            with pytest.raises(InnerRequestLimitReached, match="inner_debug_limit"):
                ledger.reserve(run, "debug", estimated_input_tokens=10, debug_limit=0)
            # depth 1: exactly one debug request
            res = ledger.reserve(run, "debug", estimated_input_tokens=10, debug_limit=1)
            ledger.commit(res.request_id, ProviderUsage(1, 1))
            with pytest.raises(InnerRequestLimitReached, match="inner_debug_limit"):
                ledger.reserve(run, "debug", estimated_input_tokens=10, debug_limit=1)
            # the persisted theta limit cannot be renegotiated
            with pytest.raises(ComparisonBudgetError, match="debug_limit_mismatch"):
                ledger.reserve(run, "debug", estimated_input_tokens=10, debug_limit=2)

    def test_inner_main_cap_at_six(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            run = RUNS[CELLS[0]][0]
            for _ in range(6):
                commit_main(ledger, run)
            with pytest.raises(
                InnerRequestLimitReached, match="inner_main_requests_exhausted"
            ):
                ledger.reserve(run, "main", estimated_input_tokens=10)

    def test_open_reservations_occupy_shape_slots(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            run = RUNS[CELLS[0]][0]
            for _ in range(6):
                ledger.reserve(run, "main", estimated_input_tokens=10)
            with pytest.raises(
                InnerRequestLimitReached, match="inner_main_requests_exhausted"
            ):
                ledger.reserve(run, "main", estimated_input_tokens=10)

    def test_snapshot_is_a_deep_copy(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            commit_main(ledger, RUNS[CELLS[0]][0])
            snap = ledger.snapshot()
            snap["global"]["committed"]["requests"] = 9999
            snap["cells"][CELLS[0].cell_key()]["committed"]["tokens"] = 0
            snap["requests"].clear()
            fresh = ledger.snapshot()
            assert fresh["global"]["committed"]["requests"] == 1
            assert fresh["cells"][CELLS[0].cell_key()]["committed"]["tokens"] == 10
            assert len(fresh["requests"]) == 1


class TestCellQuota:
    def test_cell_token_exact_boundary_then_truthful_termination(
        self, tmp_path: Path
    ) -> None:
        clock = Clock()
        ledger = started_ledger(tmp_path, clock)
        run = RUNS[CELLS[0]][0]
        # 191808 + 8192 == 200000 exactly: the boundary reservation is allowed.
        res = ledger.reserve(run, "main", estimated_input_tokens=191_808)
        assert res.tokens == 200_000
        ledger.commit(res.request_id, ProviderUsage(191_808, 8_192))
        # One more token of need terminates ONLY this cell, truthfully.
        with pytest.raises(CellBudgetExhausted, match="cell_tokens_exhausted"):
            ledger.reserve(run, "main", estimated_input_tokens=10)
        snap = ledger.snapshot()
        terminated = snap["cells"][CELLS[0].cell_key()]["terminated"]
        assert terminated["reason"] == "cell_tokens_exhausted"
        assert not ledger.stopped  # other cells continue
        commit_main(ledger, RUNS[CELLS[1]][0])
        ledger.close()
        # Termination survives resume and is not rewritten.
        clock.now += 60.0
        with open_ledger(tmp_path, clock) as reopened:
            with pytest.raises(CellBudgetExhausted, match="cell_terminated"):
                reopened.reserve(run, "main", estimated_input_tokens=10)
            snap = reopened.snapshot()
            assert (
                snap["cells"][CELLS[0].cell_key()]["terminated"]["reason"]
                == "cell_tokens_exhausted"
            )
            assert not reopened.stopped

    def test_cell_request_cap_cannot_be_exceeded(self, tmp_path: Path) -> None:
        """64 real requests fill a cell exactly; a 65th is refused.

        The cell request cap (64) equals 8 runs x (6 main + 2 debug) by
        frozen protocol arithmetic, so the inner shape caps fire first —
        the cell cap is defense in depth that can never be bypassed.
        """
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            for run in RUNS[CELLS[0]]:
                for _ in range(6):
                    commit_main(ledger, run)
                for _ in range(2):
                    res = ledger.reserve(
                        run, "debug", estimated_input_tokens=10, debug_limit=2
                    )
                    ledger.commit(res.request_id, ProviderUsage(6, 4))
            view = ledger.snapshot()["cells"][CELLS[0].cell_key()]
            assert view["committed"]["requests"] == 64
            assert view["remaining"]["requests"] == 0
            for run in RUNS[CELLS[0]]:
                with pytest.raises(InnerRequestLimitReached):
                    ledger.reserve(run, "main", estimated_input_tokens=10)
                with pytest.raises(InnerRequestLimitReached):
                    ledger.reserve(
                        run, "debug", estimated_input_tokens=10, debug_limit=2
                    )
            assert view["terminated"] is None
            assert ledger.snapshot()["cells"][CELLS[0].cell_key()][
                "committed"
            ]["requests"] == 64
            commit_main(ledger, RUNS[CELLS[1]][0])  # no borrowing needed elsewhere


class TestGlobalQuota:
    def test_global_token_exact_boundary_then_stop(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            for cell in CELLS:  # 27 cells x 200000 == 5400000 exactly
                res = ledger.reserve(
                    RUNS[cell][0], "main", estimated_input_tokens=191_808
                )
                ledger.commit(res.request_id, ProviderUsage(191_808, 8_192))
            snap = ledger.snapshot()
            assert snap["global"]["committed"]["tokens"] == 5_400_000
            assert snap["global"]["remaining"]["tokens"] == 0
            assert not ledger.stopped
            with pytest.raises(GlobalBudgetExhausted, match="tokens_exhausted"):
                ledger.reserve(RUNS[CELLS[0]][1], "main", estimated_input_tokens=10)
            assert ledger.stopped
            assert (
                ledger.snapshot()["stopped"]["reason"] == "global_exhausted:tokens"
            )

    def test_global_request_boundary_with_crafted_near_full_state(
        self, tmp_path: Path
    ) -> None:
        """A valid persisted state at 1727 requests admits exactly one more.

        Like the cell cap, 1728 == 27 x 64 means full cells shadow the
        global request cap; this crafts the largest valid state with one
        free slot and proves the boundary arithmetic on reopen.
        """
        clock = Clock()
        ledger = open_ledger(tmp_path, clock)
        ledger.close()
        path = tmp_path / "ledger.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["started_at"] = 9_000.0
        n = 0

        def add_record(cell: CellSpec, run: InnerRunSpec, kind: str) -> None:
            nonlocal n
            n += 1
            state["requests"].append(
                {
                    "request_id": f"crafted-{n:05d}",
                    "run_identity": run.identity(),
                    "cell_identity": cell.identity(),
                    "kind": kind,
                    "reserved_tokens": 100,
                    "charged_tokens": 90,
                    "usage_known": True,
                    "input_tokens": 50,
                    "output_tokens": 40,
                    "provider_request_id": f"prov-{n:05d}",
                    "reserved_at": 9_000.5,
                    "committed_at": 9_000.6,
                }
            )
            run_state = state["runs"][run.identity()]
            run_state["main_requests" if kind == "main" else "debug_requests"] += 1
            if kind == "debug":
                run_state["debug_limit"] = 2
            state["cells"][cell.identity()]["committed"]["requests"] += 1
            state["cells"][cell.identity()]["committed"]["tokens"] += 90
            state["global_committed"]["requests"] += 1
            state["global_committed"]["tokens"] += 90

        for cell in CELLS[:-1]:  # 26 full cells = 1664
            for run in RUNS[cell]:
                for _ in range(6):
                    add_record(cell, run, "main")
                for _ in range(2):
                    add_record(cell, run, "debug")
        last = CELLS[-1]  # 63 requests: run0 keeps one main slot free
        for run in RUNS[last]:
            mains = 5 if run.config_index == 0 else 6
            for _ in range(mains):
                add_record(last, run, "main")
            for _ in range(2):
                add_record(last, run, "debug")
        assert state["global_committed"]["requests"] == 1727
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

        clock.now = 9_100.0
        with open_ledger(tmp_path, clock) as reopened:
            assert reopened.started_at == 9_000.0  # crafted clock preserved
            free_run = RUNS[last][0]
            res = reopened.reserve(free_run, "main", estimated_input_tokens=10)
            reopened.commit(res.request_id, ProviderUsage(6, 4))
            snap = reopened.snapshot()
            assert snap["global"]["committed"]["requests"] == 1728
            assert snap["global"]["remaining"]["requests"] == 0
            # Every remaining slot is shape-capped: nothing can exceed 1728.
            for cell in CELLS:
                for run in RUNS[cell]:
                    with pytest.raises(
                        (InnerRequestLimitReached, CellBudgetExhausted)
                    ):
                        reopened.reserve(run, "main", estimated_input_tokens=10)

    def test_deadline_exhaustion_stops_globally_and_persists(
        self, tmp_path: Path
    ) -> None:
        clock = Clock()
        ledger = started_ledger(tmp_path, clock)
        clock.now = 10_000.0 + 86_399.0
        commit_main(ledger, RUNS[CELLS[0]][0])  # still inside the window
        clock.now = 10_000.0 + 86_400.0
        with pytest.raises(GlobalBudgetExhausted, match="elapsed_seconds_exhausted"):
            ledger.reserve(RUNS[CELLS[0]][0], "main", estimated_input_tokens=10)
        deadline = ledger.deadline_epoch
        ledger.close()
        with open_ledger(tmp_path, clock) as reopened:
            assert reopened.stopped
            assert (
                reopened.snapshot()["stopped"]["reason"]
                == "global_exhausted:elapsed_seconds"
            )
            assert reopened.deadline_epoch == deadline  # never extended
            with pytest.raises(ComparisonLedgerStopped):
                reopened.reserve(
                    RUNS[CELLS[1]][0], "main", estimated_input_tokens=10
                )


class TestFailClosedSettlement:
    def test_unknown_usage_retains_reservation_and_stops(
        self, tmp_path: Path
    ) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            run = RUNS[CELLS[0]][0]
            res = ledger.reserve(run, "main", estimated_input_tokens=100)
            ledger.commit(res.request_id, None)
            snap = ledger.snapshot()
            # The full conservative reservation is permanently charged...
            assert snap["global"]["committed"] == {
                "requests": 1,
                "tokens": 100 + PINNED_OUTPUT,
            }
            record = snap["requests"][0]
            assert record["usage_known"] is False
            assert record["charged_tokens"] == record["reserved_tokens"]
            # ...and the unknown balance stops the ledger globally.
            assert snap["stopped"]["reason"] == f"unknown_usage:{res.request_id}"
            with pytest.raises(ComparisonLedgerStopped):
                ledger.reserve(run, "main", estimated_input_tokens=10)

    def test_usage_above_reservation_stops_after_commit(
        self, tmp_path: Path
    ) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            run = RUNS[CELLS[0]][0]
            res = ledger.reserve(run, "main", estimated_input_tokens=10)
            ledger.commit(
                res.request_id, ProviderUsage(10, PINNED_OUTPUT + 1)
            )
            snap = ledger.snapshot()
            assert snap["global"]["committed"]["tokens"] == 10 + PINNED_OUTPUT + 1
            assert snap["stopped"]["reason"] == "usage_exceeded_reservation"

    def test_unresolved_reservation_on_reopen_stops(self, tmp_path: Path) -> None:
        clock = Clock()
        ledger = started_ledger(tmp_path, clock)
        run = RUNS[CELLS[0]][0]
        res = ledger.reserve(run, "main", estimated_input_tokens=100)
        ledger.close()  # crash: reservation never settled
        with open_ledger(tmp_path, clock) as reopened:
            snap = reopened.snapshot()
            assert (
                snap["stopped"]["reason"] == "unresolved_reservations_on_restart"
            )
            # The unresolved reservation still occupies its slots.
            assert snap["global"]["open_reservations"] == 1
            assert snap["global"]["open_tokens"] == 100 + PINNED_OUTPUT
            with pytest.raises(ComparisonLedgerStopped):
                reopened.reserve(RUNS[CELLS[1]][0], "main", estimated_input_tokens=10)
            with pytest.raises(ComparisonBudgetError, match="unknown_reservation"):
                reopened.commit("not-a-real-id", ProviderUsage(1, 1))
            assert res.request_id != "not-a-real-id"

    def test_double_and_mismatched_commits_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            res = ledger.reserve(RUNS[CELLS[0]][0], "main", estimated_input_tokens=10)
            ledger.commit(res.request_id, ProviderUsage(6, 4))
            with pytest.raises(ComparisonBudgetError, match="unknown_reservation"):
                ledger.commit(res.request_id, ProviderUsage(6, 4))  # double
            with pytest.raises(ComparisonBudgetError, match="unknown_reservation"):
                ledger.commit("deadbeef" * 4, ProviderUsage(1, 1))  # mismatched
            with pytest.raises(ComparisonBudgetError, match="request_id_invalid"):
                ledger.commit("", ProviderUsage(1, 1))
            assert ledger.snapshot()["global"]["committed"]["requests"] == 1

    def test_duplicate_provider_request_id_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            run = RUNS[CELLS[0]][0]
            first = ledger.reserve(run, "main", estimated_input_tokens=10)
            ledger.commit(
                first.request_id, ProviderUsage(6, 4), provider_request_id="p-dup"
            )
            second = ledger.reserve(run, "main", estimated_input_tokens=10)
            with pytest.raises(
                ComparisonBudgetError, match="duplicate_provider_request_id"
            ):
                ledger.commit(
                    second.request_id,
                    ProviderUsage(6, 4),
                    provider_request_id="p-dup",
                )
            # The refused commit did not consume the reservation.
            ledger.commit(second.request_id, ProviderUsage(6, 4))
            assert ledger.snapshot()["global"]["committed"]["requests"] == 2

    @pytest.mark.parametrize("bad_usage", ["usage", 123, (1, 2)])
    def test_wrong_usage_type_rejected(self, tmp_path: Path, bad_usage: object) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            res = ledger.reserve(RUNS[CELLS[0]][0], "main", estimated_input_tokens=10)
            with pytest.raises(ComparisonBudgetError, match="usage_type_invalid"):
                ledger.commit(res.request_id, bad_usage)  # type: ignore[arg-type]

    def test_provider_usage_validation(self) -> None:
        with pytest.raises(ComparisonBudgetError):
            ProviderUsage(-1, 0)
        with pytest.raises(ComparisonBudgetError):
            ProviderUsage(True, 0)
        with pytest.raises(ComparisonBudgetError):
            ProviderUsage(1.5, 0)  # type: ignore[arg-type]


class TestCorruption:
    def _valid_state(self, tmp_path: Path, clock: Clock) -> Path:
        ledger = started_ledger(tmp_path, clock)
        commit_main(ledger, RUNS[CELLS[0]][0])
        res = ledger.reserve(
            RUNS[CELLS[0]][0], "debug", estimated_input_tokens=20, debug_limit=2
        )
        ledger.commit(res.request_id, ProviderUsage(12, 8))
        ledger.close()
        return tmp_path / "ledger.json"

    def _mutate(self, path: Path, mutation) -> None:
        state = json.loads(path.read_text(encoding="utf-8"))
        mutation(state)
        path.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
        )

    def test_bool_counter_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)
        self._mutate(path, lambda s: s["global_committed"].update(requests=True))
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger"):
            open_ledger(tmp_path, clock)

    def test_float_counter_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)
        self._mutate(path, lambda s: s["global_committed"].update(tokens=2.0))
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger"):
            open_ledger(tmp_path, clock)

    def test_negative_counter_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)
        self._mutate(path, lambda s: s["global_committed"].update(tokens=-1))
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger"):
            open_ledger(tmp_path, clock)

    def test_nonfinite_clock_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)
        self._mutate(path, lambda s: s.update(started_at=math.inf))
        with pytest.raises(
            ComparisonBudgetError, match="corrupt_ledger:started_at"
        ):
            open_ledger(tmp_path, clock)

    def test_expanded_limits_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)
        self._mutate(
            path, lambda s: s["limits"]["global"].update(max_requests=2000)
        )
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger:limits"):
            open_ledger(tmp_path, clock)

    def test_counter_record_mismatch_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)
        self._mutate(path, lambda s: s["global_committed"].update(requests=99))
        with pytest.raises(
            ComparisonBudgetError, match="corrupt_ledger:inconsistent"
        ):
            open_ledger(tmp_path, clock)

    def test_duplicate_request_id_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)

        def mutate(state: dict) -> None:
            state["requests"][1]["request_id"] = state["requests"][0]["request_id"]

        self._mutate(path, mutate)
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger:request"):
            open_ledger(tmp_path, clock)

    def test_usage_total_mismatch_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)
        self._mutate(path, lambda s: s["requests"][0].update(charged_tokens=999))
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger:request"):
            open_ledger(tmp_path, clock)

    def test_non_bool_usage_flag_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)
        self._mutate(path, lambda s: s["requests"][0].update(usage_known=1))
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger:request"):
            open_ledger(tmp_path, clock)

    def test_run_counter_above_cap_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)

        def mutate(state: dict) -> None:
            run_id = state["requests"][0]["run_identity"]
            state["runs"][run_id]["main_requests"] = 7

        self._mutate(path, mutate)
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger:run"):
            open_ledger(tmp_path, clock)

    def test_debug_limit_above_inner_cap_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)

        def mutate(state: dict) -> None:
            run_id = state["requests"][0]["run_identity"]
            state["runs"][run_id]["debug_limit"] = 3

        self._mutate(path, mutate)
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger:run"):
            open_ledger(tmp_path, clock)

    def test_reservation_with_unknown_run_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)

        def mutate(state: dict) -> None:
            state["reservations"].append(
                {
                    "id": "stuck",
                    "run_identity": "0" * 64,
                    "cell_identity": state["requests"][0]["cell_identity"],
                    "kind": "main",
                    "tokens": 100,
                    "created_at": 10_000.0,
                }
            )

        self._mutate(path, mutate)
        with pytest.raises(
            ComparisonBudgetError, match="corrupt_ledger:reservation"
        ):
            open_ledger(tmp_path, clock)

    def test_bad_stop_record_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)
        self._mutate(path, lambda s: s.update(stopped={"reason": "", "at": 1.0}))
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger:stopped"):
            open_ledger(tmp_path, clock)

    def test_bad_event_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)
        self._mutate(path, lambda s: s["events"].append({"type": "", "at": 1.0}))
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger:event"):
            open_ledger(tmp_path, clock)

    def test_malformed_json_rejected(self, tmp_path: Path) -> None:
        clock = Clock()
        path = self._valid_state(tmp_path, clock)
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger"):
            open_ledger(tmp_path, clock)


class TestRace:
    def test_concurrent_reserves_never_exceed_cell_cap(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            runs = RUNS[CELLS[0]]

            def attempt(args):
                run, kind = args
                try:
                    if kind == "main":
                        res = ledger.reserve(run, "main", estimated_input_tokens=10)
                    else:
                        res = ledger.reserve(
                            run, "debug", estimated_input_tokens=10, debug_limit=2
                        )
                    # Real flow settles immediately (serial model calls), so
                    # open reservations never accumulate token pressure.
                    ledger.commit(res.request_id, ProviderUsage(6, 4))
                    return "ok"
                except InnerRequestLimitReached:
                    return "inner"
                except CellBudgetExhausted:
                    return "cell"

            attempts = [
                (run, kind)
                for run in runs
                for kind, count in (("main", 8), ("debug", 4))
                for _ in range(count)
            ]
            with ThreadPoolExecutor(max_workers=16) as pool:
                results = list(pool.map(attempt, attempts))
            assert results.count("ok") == 64  # exactly the cell capacity
            assert results.count("cell") == 0  # inner caps shadow it
            assert results.count("inner") == len(attempts) - 64
            view = ledger.snapshot()["cells"][CELLS[0].cell_key()]
            assert view["committed"]["requests"] == 64


class TestGuardedAttempt:
    def _usage_fn(self, response: dict) -> ProviderUsage:
        usage = response["usage"]
        return ProviderUsage(usage["prompt_tokens"], usage["completion_tokens"])

    def test_successful_attempt_charges_actual_usage_once(
        self, tmp_path: Path
    ) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            run = RUNS[CELLS[0]][0]
            calls = []
            seen: list[ComparisonReservation] = []

            def transport(request: dict) -> dict:
                calls.append(request)
                return {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}

            response = guarded_comparison_attempt(
                ledger,
                run,
                "main",
                transport,
                {"prompt": "x"},
                estimated_input_tokens=500,
                usage_fn=self._usage_fn,
                on_reserve=seen.append,
                provider_request_id_fn=lambda r: "prov-ok",
            )
            assert response["usage"]["prompt_tokens"] == 100
            assert len(calls) == 1  # retries pinned to zero: single attempt
            assert len(seen) == 1
            snap = ledger.snapshot()
            # Charged the ACTUAL 150, not the 8692 reservation: no double charge.
            assert snap["global"]["committed"]["tokens"] == 150
            assert snap["global"]["open_reservations"] == 0
            assert snap["requests"][0]["provider_request_id"] == "prov-ok"

    def test_transport_failure_retains_and_stops(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            run = RUNS[CELLS[0]][0]

            def broken(request: dict) -> dict:
                raise TimeoutError("provider unreachable")

            with pytest.raises(TimeoutError):
                guarded_comparison_attempt(
                    ledger,
                    run,
                    "main",
                    broken,
                    {},
                    estimated_input_tokens=500,
                    usage_fn=self._usage_fn,
                )
            snap = ledger.snapshot()
            assert snap["global"]["committed"]["tokens"] == 500 + PINNED_OUTPUT
            assert snap["stopped"]["reason"].startswith("unknown_usage:")

    def test_usage_parser_failure_retains_and_stops(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            run = RUNS[CELLS[0]][0]

            def bad_usage(response: dict) -> ProviderUsage:
                raise ValueError("usage_unavailable")

            with pytest.raises(ValueError, match="usage_unavailable"):
                guarded_comparison_attempt(
                    ledger,
                    run,
                    "main",
                    lambda request: {"usage": {}},
                    {},
                    estimated_input_tokens=500,
                    usage_fn=bad_usage,
                )
            assert ledger.stopped

    def test_conservative_estimate_covers_full_prompt_plus_output(
        self, tmp_path: Path
    ) -> None:
        messages = [
            {"role": "system", "content": "s" * 400},
            {"role": "user", "content": "u" * 1200},
        ]
        est = conservative_input_tokens(messages, {})
        # Full UTF-8 prompt bytes (system + messages + keys) plus framing,
        # never a whitespace heuristic.
        assert est >= 400 + 1200 + 256 + 64 * 2
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            res = ledger.reserve(
                RUNS[CELLS[0]][0], "main", estimated_input_tokens=est
            )
            assert res.tokens == est + PINNED_OUTPUT
        with pytest.raises(Exception, match="unsupported_nontext_request"):
            conservative_input_tokens([{"role": "user", "content": 5}], {})


BARRIER_ID = "f0" * 32


class TestFinalPhase:
    """Irreversible final-phase transition (EXP-006 supervisor requirement).

    After the verified all-cell barrier the ledger durably enters the final
    phase: every reservation/generation entry refuses WITHOUT charging,
    mutating or globally stopping, and the original clock/accounting are
    preserved across reopen/resume.
    """

    def test_no_call_or_charge_after_transition_through_guard(
        self, tmp_path: Path
    ) -> None:
        clock = Clock()
        ledger = started_ledger(tmp_path, clock)
        # Baseline accounting: one settled real reservation.
        request_id = commit_main(ledger, RUNS[CELLS[0]][0], est=100, in_t=60, out_t=40)
        assert request_id
        started_at = ledger.started_at
        deadline = ledger.deadline_epoch
        ledger.enter_final_phase(BARRIER_ID)
        assert not ledger.stopped
        assert ledger.final_phase == {"barrier_identity": BARRIER_ID, "at": 10000.0}
        # The ACTUAL reservation/provider path: unused configs of OTHER
        # cells/runs (incl. fresh run slots) all refuse before transport.
        calls: list[object] = []

        def transport(request: object) -> object:
            calls.append(request)
            return object()

        for run in (RUNS[CELLS[0]][0], RUNS[CELLS[1]][7], RUNS[CELLS[-1]][3]):
            with pytest.raises(ComparisonFinalPhaseActive):
                guarded_comparison_attempt(
                    ledger,
                    run,
                    "main",
                    transport,
                    {"prompt": "x"},
                    estimated_input_tokens=10,
                    usage_fn=lambda response: ProviderUsage(1, 1),
                )
        assert calls == []
        snapshot = ledger.snapshot()
        assert snapshot["global"]["committed"] == {"requests": 1, "tokens": 100}
        assert snapshot["stopped"] is None
        assert snapshot["final_phase"]["barrier_identity"] == BARRIER_ID
        ledger.close()
        # Reopen/resume: phase, clock and accounting preserved byte-exactly.
        reopened = open_ledger(tmp_path, clock)
        assert reopened.final_phase == {"barrier_identity": BARRIER_ID, "at": 10000.0}
        assert reopened.started_at == started_at
        assert reopened.deadline_epoch == deadline
        assert not reopened.stopped
        with pytest.raises(ComparisonFinalPhaseActive):
            reopened.reserve(
                RUNS[CELLS[2]][5], "debug", estimated_input_tokens=10, debug_limit=2
            )
        with pytest.raises(ComparisonBudgetError, match="clock_already_started"):
            reopened.start_clock()
        reopened.close()

    def test_transition_idempotent_same_barrier_only(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            ledger.enter_final_phase(BARRIER_ID)
            ledger.enter_final_phase(BARRIER_ID)  # crash-safe resume no-op
            with pytest.raises(ComparisonBudgetError, match="final_phase_barrier_mismatch"):
                ledger.enter_final_phase("0" * 64)
            with pytest.raises(ComparisonBudgetError, match="final_phase_barrier_invalid"):
                ledger.enter_final_phase("not-a-hash")
            assert not ledger.stopped

    def test_transition_requires_started_clock_and_unstopped(
        self, tmp_path: Path
    ) -> None:
        clock = Clock()
        ledger = open_ledger(tmp_path, clock)
        with pytest.raises(ComparisonBudgetError, match="clock_not_started"):
            ledger.enter_final_phase(BARRIER_ID)
        ledger.start_clock()
        ledger.stop("test_stop")
        with pytest.raises(ComparisonLedgerStopped):
            ledger.enter_final_phase(BARRIER_ID)
        ledger.close()

    def test_tampered_final_phase_fails_closed_on_open(self, tmp_path: Path) -> None:
        clock = Clock()
        with started_ledger(tmp_path, clock) as ledger:
            ledger.enter_final_phase(BARRIER_ID)
        path = tmp_path / "ledger.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["final_phase"] = {"barrier_identity": "zz", "at": 10000.0}
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger:final_phase"):
            open_ledger(tmp_path, clock)
        # A final phase with no started clock is inconsistent.
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["final_phase"] = {"barrier_identity": BARRIER_ID, "at": 10000.0}
        payload["started_at"] = None
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ComparisonBudgetError, match="corrupt_ledger:final_phase"):
            open_ledger(tmp_path, clock)
