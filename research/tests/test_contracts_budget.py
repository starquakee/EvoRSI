"""Contract tests for research/contracts/budget.py (US-003).

Pins the budget contract: reservation before EVERY model attempt including
retries, the 200000-token / 30-request / 5400-second limits, retain+stop on
unknown usage, never-reset persistence across restarts, serial access, and
checkpoint-safe cancellation. Fake transports only — no real model calls.
"""
from __future__ import annotations

import json
import time

import pytest

from research.contracts.budget import (
    BudgetError,
    BudgetExhausted,
    BudgetLedger,
    BudgetLimits,
    Cancelled,
    CancellationToken,
    LedgerLocked,
    LedgerStopped,
    ModelUsage,
    call_with_retries,
    guarded_attempt,
)


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _open(tmp_path, *, clock=None, limits=None) -> BudgetLedger:
    return BudgetLedger.open(
        tmp_path / "ledger.json",
        limits=limits,
        now_fn=(clock or FakeClock()),
    )


class TestDefaults:
    def test_prd_budget_limits(self):
        limits = BudgetLimits()
        assert limits.max_tokens == 200_000
        assert limits.max_requests == 30
        assert limits.max_elapsed_seconds == 5_400.0

    def test_fresh_ledger_snapshot(self, tmp_path):
        with _open(tmp_path) as ledger:
            snap = ledger.snapshot()
            assert snap["committed"] == {"requests": 0, "tokens": 0}
            assert snap["open_reservations"] == []
            assert snap["remaining"]["tokens"] == 200_000
            assert snap["remaining"]["requests"] == 30
            assert snap["stopped"] is None


class TestReservationAccounting:
    def test_reservation_holds_token_and_request_slots(self, tmp_path):
        with _open(tmp_path) as ledger:
            reservation = ledger.reserve(3_000, 2_000)
            snap = ledger.snapshot()
            assert snap["remaining"]["tokens"] == 200_000 - 5_000
            assert snap["remaining"]["requests"] == 29
            assert snap["committed"] == {"requests": 0, "tokens": 0}
            assert len(snap["open_reservations"]) == 1
            ledger.commit(reservation.reservation_id, ModelUsage(2_500, 800))
            snap = ledger.snapshot()
            assert snap["committed"] == {"requests": 1, "tokens": 3_300}
            assert snap["open_reservations"] == []
            assert snap["remaining"]["tokens"] == 200_000 - 3_300

    def test_token_limit_blocks_over_reservation(self, tmp_path):
        limits = BudgetLimits(
            max_tokens=1_000, max_requests=10, max_elapsed_seconds=5_400.0,
        )
        with _open(tmp_path, limits=limits) as ledger:
            ledger.reserve(600, 400)  # exactly the budget, still open
            with pytest.raises(BudgetExhausted, match="tokens"):
                ledger.reserve(1, 1)

    def test_request_limit_blocks_over_reservation(self, tmp_path):
        limits = BudgetLimits(
            max_tokens=1_000_000, max_requests=2, max_elapsed_seconds=5_400.0,
        )
        with _open(tmp_path, limits=limits) as ledger:
            ledger.reserve(1, 1)
            ledger.reserve(1, 1)
            with pytest.raises(BudgetExhausted, match="requests"):
                ledger.reserve(1, 1)

    def test_invalid_estimates_rejected(self, tmp_path):
        with _open(tmp_path) as ledger:
            with pytest.raises(ValueError):
                ledger.reserve(-1, 100)
            with pytest.raises(ValueError):
                ledger.reserve(0, 0)

    def test_unknown_reservation_commit_rejected(self, tmp_path):
        with _open(tmp_path) as ledger:
            with pytest.raises(BudgetError, match="unknown_reservation"):
                ledger.commit("no-such-id", ModelUsage(1, 1))


class TestElapsedLimit:
    def test_elapsed_deadline_blocks_reservation(self, tmp_path):
        clock = FakeClock()
        with _open(tmp_path, clock=clock) as ledger:
            ledger.reserve(10, 10)
            clock.advance(5_400.0)
            with pytest.raises(BudgetExhausted, match="elapsed"):
                ledger.reserve(10, 10)

    def test_restart_cannot_reset_elapsed_anchor(self, tmp_path):
        clock = FakeClock()
        with _open(tmp_path, clock=clock) as ledger:
            ledger.reserve(10, 10)
        clock.advance(6_000.0)
        with _open(tmp_path, clock=clock) as reopened:
            # started_at persisted; elapsed keeps counting across restart.
            with pytest.raises(BudgetExhausted, match="elapsed"):
                reopened.reserve(10, 10)


class TestStopAndUnknownUsage:
    def test_stop_blocks_new_reservations(self, tmp_path):
        with _open(tmp_path) as ledger:
            ledger.stop("operator_halt")
            with pytest.raises(LedgerStopped, match="operator_halt"):
                ledger.reserve(1, 1)

    def test_first_stop_reason_wins(self, tmp_path):
        with _open(tmp_path) as ledger:
            ledger.stop("first")
            ledger.stop("second")
            assert ledger.snapshot()["stopped"]["reason"] == "first"

    def test_stop_persists_across_restart(self, tmp_path):
        clock = FakeClock()
        with _open(tmp_path, clock=clock) as ledger:
            ledger.stop("budget_office_closed")
        with _open(tmp_path, clock=clock) as reopened:
            with pytest.raises(LedgerStopped, match="budget_office_closed"):
                reopened.reserve(1, 1)

    def test_unknown_usage_retains_reservation_and_stops(self, tmp_path):
        with _open(tmp_path) as ledger:
            reservation = ledger.reserve(3_000, 2_000)
            ledger.commit(reservation.reservation_id, None)
            snap = ledger.snapshot()
            # Conservative estimate (input + max output) permanently charged.
            assert snap["committed"] == {"requests": 1, "tokens": 5_000}
            assert snap["open_reservations"] == []
            assert snap["stopped"]["reason"].startswith("unknown_usage:")
            with pytest.raises(LedgerStopped):
                ledger.reserve(1, 1)


class TestPersistence:
    def test_restart_continues_totals_and_started_at(self, tmp_path):
        clock = FakeClock()
        with _open(tmp_path, clock=clock) as ledger:
            started = ledger.started_at
            reservation = ledger.reserve(100, 100)
            ledger.commit(reservation.reservation_id, ModelUsage(120, 30))
        clock.advance(60.0)
        with _open(tmp_path, clock=clock) as reopened:
            assert reopened.started_at == started
            snap = reopened.snapshot()
            assert snap["committed"] == {"requests": 1, "tokens": 150}
            assert snap["remaining"]["requests"] == 29
            assert snap["elapsed_seconds"] == pytest.approx(60.0)

    def test_no_reset_api(self, tmp_path):
        with _open(tmp_path) as ledger:
            assert not hasattr(ledger, "reset")
            assert not hasattr(ledger, "clear")

    def test_corrupt_ledger_fails_closed(self, tmp_path):
        (tmp_path / "ledger.json").write_text("{broken", encoding="utf-8")
        with pytest.raises(BudgetError, match="corrupt_ledger"):
            _open(tmp_path)

    def test_wrong_schema_version_fails_closed(self, tmp_path):
        (tmp_path / "ledger.json").write_text(json.dumps({
            "schema_version": 999,
        }))
        with pytest.raises(BudgetError, match="schema_version"):
            _open(tmp_path)

    def test_events_are_recorded(self, tmp_path):
        with _open(tmp_path) as ledger:
            reservation = ledger.reserve(10, 10)
            ledger.commit(reservation.reservation_id, ModelUsage(5, 5))
            ledger.stop("done")
        payload = json.loads((tmp_path / "ledger.json").read_text())
        kinds = [event["type"] for event in payload["events"]]
        assert kinds == ["reserve", "commit", "stop"]


class TestSerialAccess:
    def test_second_opener_refused(self, tmp_path):
        first = _open(tmp_path)
        try:
            with pytest.raises(LedgerLocked):
                _open(tmp_path)
        finally:
            first.close()

    def test_close_releases_lock(self, tmp_path):
        _open(tmp_path).close()
        with _open(tmp_path) as ledger:  # must not raise
            assert ledger.snapshot()["remaining"]["requests"] == 30


class TestGuardedAttempt:
    def test_reservation_exists_before_transport_runs(self, tmp_path):
        seen = {}

        def transport(request):
            snap = ledger.snapshot()
            seen["open"] = len(snap["open_reservations"])
            seen["committed_during_call"] = snap["committed"]["requests"]
            return {"usage": {"input": 100, "output": 40}}

        with _open(tmp_path) as ledger:
            response = guarded_attempt(
                ledger, CancellationToken(), transport, {"prompt": "x"},
                estimated_input_tokens=500, max_output_tokens=500,
                usage_fn=lambda r: ModelUsage(
                    r["usage"]["input"], r["usage"]["output"],
                ),
            )
            snap = ledger.snapshot()
            assert snap["committed"] == {"requests": 1, "tokens": 140}
            assert snap["open_reservations"] == []
        assert response["usage"]["output"] == 40
        assert seen == {"open": 1, "committed_during_call": 0}

    def test_cancelled_token_blocks_before_reservation(self, tmp_path):
        def transport(request):  # pragma: no cover - must not be reached
            raise AssertionError("transport called after cancellation")

        token = CancellationToken()
        token.stop("deadline")
        with _open(tmp_path) as ledger:
            with pytest.raises(Cancelled, match="stopped"):
                guarded_attempt(
                    ledger, token, transport, {},
                    estimated_input_tokens=1, max_output_tokens=1,
                    usage_fn=lambda r: None,
                )
            assert ledger.snapshot()["committed"]["requests"] == 0
            assert ledger.snapshot()["open_reservations"] == []

    def test_transport_exception_retains_and_stops(self, tmp_path):
        def transport(request):
            raise ConnectionError("dropped mid-request")

        with _open(tmp_path) as ledger:
            with pytest.raises(ConnectionError):
                guarded_attempt(
                    ledger, CancellationToken(), transport, {},
                    estimated_input_tokens=300, max_output_tokens=200,
                    usage_fn=lambda r: None,
                )
            snap = ledger.snapshot()
            # Usage unknown: conservative reservation retained, ledger stopped.
            assert snap["committed"] == {"requests": 1, "tokens": 500}
            assert snap["stopped"]["reason"].startswith("unknown_usage:")

    def test_unreported_usage_retains_and_stops(self, tmp_path):
        with _open(tmp_path) as ledger:
            guarded_attempt(
                ledger, CancellationToken(), lambda r: {"ok": True}, {},
                estimated_input_tokens=100, max_output_tokens=100,
                usage_fn=lambda r: None,
            )
            snap = ledger.snapshot()
            assert snap["committed"] == {"requests": 1, "tokens": 200}
            assert snap["stopped"] is not None


class TestRetries:
    def _usage(self, response) -> ModelUsage:
        return ModelUsage(response["in"], response["out"])

    def test_every_retry_reserves_before_calling(self, tmp_path):
        calls = []

        def transport(request):
            calls.append(ledger.snapshot()["open_reservations"])
            if len(calls) < 3:
                return {"status": "rate_limited", "in": 50, "out": 10}
            return {"status": "ok", "in": 60, "out": 20}

        with _open(tmp_path) as ledger:
            response = call_with_retries(
                ledger, CancellationToken(), transport, {},
                max_attempts=5,
                estimated_input_tokens=500, max_output_tokens=500,
                usage_fn=self._usage,
                should_retry=lambda r: r["status"] == "rate_limited",
            )
            assert response["status"] == "ok"
            # Every attempt (incl. retries) held exactly one reservation
            # while the transport ran.
            assert [len(open_res) for open_res in calls] == [1, 1, 1]
            snap = ledger.snapshot()
            assert snap["committed"] == {"requests": 3, "tokens": 200}
            assert snap["open_reservations"] == []

    def test_request_budget_exhaustion_stops_retry_loop(self, tmp_path):
        limits = BudgetLimits(
            max_tokens=1_000_000, max_requests=1, max_elapsed_seconds=5_400.0,
        )
        attempts = []

        def transport(request):
            attempts.append(1)
            return {"status": "rate_limited", "in": 10, "out": 10}

        with _open(tmp_path, limits=limits) as ledger:
            with pytest.raises(BudgetExhausted, match="requests"):
                call_with_retries(
                    ledger, CancellationToken(), transport, {},
                    max_attempts=5,
                    estimated_input_tokens=10, max_output_tokens=10,
                    usage_fn=self._usage,
                    should_retry=lambda r: True,
                )
        assert len(attempts) == 1  # second attempt never reached the model

    def test_deadline_between_attempts_stops_retry_loop(self, tmp_path):
        token = CancellationToken(deadline_epoch=time.time() + 3_600)
        attempts = []

        def transport(request):
            attempts.append(1)
            token.deadline_epoch = time.time() - 1  # clock jumps past deadline
            return {"status": "rate_limited", "in": 10, "out": 10}

        with _open(tmp_path) as ledger:
            with pytest.raises(Cancelled, match="deadline"):
                call_with_retries(
                    ledger, token, transport, {},
                    max_attempts=5,
                    estimated_input_tokens=10, max_output_tokens=10,
                    usage_fn=self._usage,
                    should_retry=lambda r: True,
                )
        assert len(attempts) == 1


class TestCancellationToken:
    def test_deadline_enforced(self):
        token = CancellationToken.after(60.0, now=1_000.0)
        token.check(now=1_059.9)
        with pytest.raises(Cancelled, match="deadline"):
            token.check(now=1_060.0)

    def test_stop_persists(self):
        token = CancellationToken()
        token.check(now=1.0)
        token.stop("operator", now=2.0)
        with pytest.raises(Cancelled, match="stopped:operator"):
            token.check(now=3.0)

    def test_checkpoint_roundtrip_preserves_deadline_and_stop(self):
        token = CancellationToken.after(60.0, now=1_000.0)
        token.stop("halt", now=1_010.0)
        restored = CancellationToken.from_dict(token.to_dict())
        assert restored.deadline_epoch == pytest.approx(1_060.0)
        assert restored.stopped_reason == "halt"
        assert restored.stopped_at == pytest.approx(1_010.0)
        # Resume can never revive a stopped run.
        with pytest.raises(Cancelled, match="stopped:halt"):
            restored.check(now=1_020.0)

    def test_checkpoint_roundtrip_of_live_token_keeps_absolute_deadline(self):
        token = CancellationToken.after(90.0, now=5_000.0)
        restored = CancellationToken.from_dict(token.to_dict())
        restored.check(now=5_089.0)
        with pytest.raises(Cancelled, match="deadline"):
            restored.check(now=5_090.0)
