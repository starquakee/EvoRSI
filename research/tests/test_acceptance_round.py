"""US-009 retry-round authorization and budget lower-bound tests (offline).

The original .runtime/acceptance-us009 tree and its stopped ledger are
immutable. A retry round activates ONLY under a supervisor-written
authorization bound to the current commit with a unique round identity and
caps no larger than the original pins. No automatic new directory/budget on
failure, no path traversal into arbitrary fresh runtimes, no budget
expansion on restart. All fixtures live in tmp paths; no authorization or
approval file is ever written to the real tree.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from research.contracts.budget import BudgetError, BudgetLedger, BudgetLimits, ModelUsage
from research.search import acceptance as ap
from research.search import acceptance_config as ac
from research.search.acceptance import AcceptanceError, run_acceptance
from research.search.acceptance_config import (
    RoundAuthorizationRefused,
    evaluate_round_authorization,
    is_valid_round_runtime_dir,
    round_runtime_dir,
)
from research.tests.test_acceptance_controller import (
    COMMIT,
    VECTORS,
    FakeAlgo,
    _fake_child,
    _repo,
)

HEAD = "f" * 40


def _auth(**overrides):
    payload = {
        "schema": ac.ROUND_AUTHORIZATION_SCHEMA,
        "implementation_commit": HEAD,
        "authorize_retry_round": True,
        "round_id": "round2",
        "caps": {"max_tokens": 200_000, "max_requests": 30, "max_elapsed_seconds": 5400},
    }
    payload.update(overrides)
    return payload


# ------------------------------------------------------------- pure checks
def test_valid_authorization_returns_round_and_caps():
    round_id, limits = evaluate_round_authorization(_auth(), head_commit=HEAD)
    assert round_id == "round2"
    assert limits.max_tokens == 200_000
    assert limits.max_requests == 30
    assert limits.max_elapsed_seconds == 5400.0


@pytest.mark.parametrize(
    "overrides,rule",
    [
        ({"schema": "other"}, "round_authorization_schema_mismatch"),
        ({"implementation_commit": "b" * 40}, "round_authorization_commit_mismatch"),
        ({"implementation_commit": ""}, "round_authorization_commit_mismatch"),
        ({"authorize_retry_round": False}, "round_authorization_not_approved"),
        ({"authorize_retry_round": "true"}, "round_authorization_not_approved"),
        ({"round_id": "round1"}, "round_id_invalid"),
        ({"round_id": "round0"}, "round_id_invalid"),
        ({"round_id": "round02"}, "round_id_invalid"),
        ({"round_id": "../round2"}, "round_id_invalid"),
        ({"round_id": "round2/../x"}, "round_id_invalid"),
        ({"round_id": "round 2"}, "round_id_invalid"),
        ({"round_id": ""}, "round_id_invalid"),
        ({"caps": {"max_tokens": 200_001, "max_requests": 30, "max_elapsed_seconds": 5400}},
         "round_authorization_tokens_exceed_pin"),
        ({"caps": {"max_tokens": 200_000, "max_requests": 31, "max_elapsed_seconds": 5400}},
         "round_authorization_requests_exceed_pin"),
        ({"caps": {"max_tokens": 200_000, "max_requests": 30, "max_elapsed_seconds": 5401}},
         "round_authorization_elapsed_exceeds_pin"),
        ({"caps": {"max_tokens": 200_000, "max_requests": 30, "max_elapsed_seconds": 0}},
         "round_authorization_elapsed_exceeds_pin"),
        ({"caps": {"max_tokens": True, "max_requests": 30, "max_elapsed_seconds": 5400}},
         "round_authorization_tokens_exceed_pin"),
        ({"caps": {"max_tokens": "200000", "max_requests": 30, "max_elapsed_seconds": 5400}},
         "round_authorization_tokens_exceed_pin"),
        ({"caps": {}}, "round_authorization_tokens_exceed_pin"),
        ({"caps": None}, "round_authorization_caps_missing"),
    ],
)
def test_authorization_refusals(overrides, rule):
    with pytest.raises(RoundAuthorizationRefused, match=rule):
        evaluate_round_authorization(_auth(**overrides), head_commit=HEAD)


def test_tighter_caps_are_allowed():
    round_id, limits = evaluate_round_authorization(
        _auth(caps={"max_tokens": 100_000, "max_requests": 12, "max_elapsed_seconds": 1800}),
        head_commit=HEAD,
    )
    assert (limits.max_tokens, limits.max_requests, limits.max_elapsed_seconds) == (
        100_000, 12, 1800.0,
    )


def test_only_round2_may_activate():
    """Narrow contract: the single requested retry is round2; later rounds
    are explicitly refused (the path helper stays future-compatible)."""
    with pytest.raises(RoundAuthorizationRefused, match="round_unsupported"):
        evaluate_round_authorization(_auth(round_id="round3"), head_commit=HEAD)
    with pytest.raises(RoundAuthorizationRefused, match="round_unsupported"):
        evaluate_round_authorization(_auth(round_id="round10"), head_commit=HEAD)
    # The derived-path helper remains valid for future rounds.
    assert round_runtime_dir("round3") == Path(".runtime/acceptance-us009-round3")


def test_round_runtime_dir_derivation_and_traversal_refusal():
    assert round_runtime_dir("round1") == ac.RUNTIME_DIR
    assert round_runtime_dir("round2") == Path(".runtime/acceptance-us009-round2")
    for bad in ("round0", "round02", "../round2", "/abs", "round2x"):
        with pytest.raises(RoundAuthorizationRefused, match="round_id_invalid"):
            round_runtime_dir(bad)


def test_round_runtime_dir_validation():
    assert is_valid_round_runtime_dir(ac.RUNTIME_DIR)
    assert is_valid_round_runtime_dir(Path(".runtime/acceptance-us009-round2"))
    assert is_valid_round_runtime_dir(Path(".runtime/acceptance-us009-round10"))
    assert not is_valid_round_runtime_dir(Path(".runtime/other"))
    assert not is_valid_round_runtime_dir(Path(".runtime/acceptance-us009-round1"))
    assert not is_valid_round_runtime_dir(Path(".runtime/acceptance-us009-fresh"))
    assert not is_valid_round_runtime_dir(Path("/abs/acceptance-us009-round2"))


def test_check_round_authorization_io(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "git_head_commit", lambda root: HEAD)
    with pytest.raises(RoundAuthorizationRefused, match="round_authorization_unavailable"):
        ac.check_round_authorization(tmp_path)
    path = tmp_path / ".runtime" / ac.ROUND_AUTHORIZATION_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RoundAuthorizationRefused, match="round_authorization_unavailable"):
        ac.check_round_authorization(tmp_path)
    path.write_text(json.dumps(_auth()), encoding="utf-8")
    round_id, _ = ac.check_round_authorization(tmp_path)
    assert round_id == "round2"


# ------------------------------------------------------- orchestration
def _repo_with_prior_ledger(tmp_path, monkeypatch, *, prior_requests=17, prior_tokens=81717):
    """Tmp repo with an immutable-looking stopped round-1 ledger."""
    repo, runtime = _repo(tmp_path, monkeypatch)
    runtime.mkdir(parents=True, exist_ok=True)
    with BudgetLedger.open(
        runtime / ac.LEDGER_FILENAME, limits=ac.acceptance_budget_limits()
    ) as ledger:
        reservation = ledger.reserve(prior_tokens // 2, prior_tokens - prior_tokens // 2)
        ledger.commit(
            reservation.reservation_id,
            ModelUsage(input_tokens=prior_tokens // 2, output_tokens=prior_tokens - prior_tokens // 2),
        )
        # top up request count to the recorded prior total (zero-token usage)
        for _ in range(prior_requests - 1):
            r = ledger.reserve(1, 1)
            ledger.commit(r.reservation_id, ModelUsage(input_tokens=0, output_tokens=0))
        ledger.stop("search_stopped:KeyboardInterrupt")
    return repo, runtime


def _write_authorization(repo: Path, *, round_id="round2", tokens=200_000, requests=30, elapsed=5400):
    """Isolated tmp authorization fixture (never written to the real tree)."""
    path = repo / ".runtime" / ac.ROUND_AUTHORIZATION_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": ac.ROUND_AUTHORIZATION_SCHEMA,
        "implementation_commit": COMMIT,
        "authorize_retry_round": True,
        "round_id": round_id,
        "caps": {"max_tokens": tokens, "max_requests": requests, "max_elapsed_seconds": elapsed},
    }), encoding="utf-8")


def test_retry_round_uses_authorized_dir_caps_and_aggregates_cost(tmp_path, monkeypatch):
    repo, round1_runtime = _repo_with_prior_ledger(tmp_path, monkeypatch)
    round1_ledger_bytes = (round1_runtime / ac.LEDGER_FILENAME).read_bytes()
    _write_authorization(repo)
    round2_dir = repo / ac.round_runtime_dir("round2")
    calls: list = []
    summary = run_acceptance(
        repo,
        round2_dir,
        algo_factory=lambda: FakeAlgo(VECTORS),
        child_runner=_fake_child(round2_dir, calls),
        round_id="round2",
    )
    assert summary["verdict"] == "complete"
    assert summary["round_id"] == "round2"
    assert calls == ["cfg0", "cfg1", "cfg2", "cfg3"]
    aggregate = summary["aggregate_cost"]
    assert aggregate["prior_round_committed"] == {"requests": 17, "tokens": 81717}
    assert aggregate["this_round_committed"]["requests"] == 4
    assert aggregate["total_committed"] == {
        "requests": 17 + 4,
        "tokens": 81717 + aggregate["this_round_committed"]["tokens"],
    }
    # The original tree and its stopped ledger are untouched.
    assert (round1_runtime / ac.LEDGER_FILENAME).read_bytes() == round1_ledger_bytes
    assert not (round1_runtime / "runs").exists()
    with BudgetLedger.open(round1_runtime / ac.LEDGER_FILENAME) as ledger:
        assert ledger.snapshot()["stopped"] is not None


def test_retry_round_rejects_mismatched_or_traversal_location(tmp_path, monkeypatch):
    repo, _ = _repo(tmp_path, monkeypatch)
    with pytest.raises(AcceptanceError, match="acceptance_runtime_dir_fixed"):
        run_acceptance(
            repo,
            repo / ".runtime/acceptance-us009-round3",  # not the round2 dir
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=_fake_child(repo / ac.RUNTIME_DIR, []),
            round_id="round2",
        )
    with pytest.raises(RoundAuthorizationRefused, match="round_id_invalid"):
        run_acceptance(
            repo,
            repo / ".runtime/acceptance-us009-round2",
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=_fake_child(repo / ac.RUNTIME_DIR, []),
            round_id="../round2",
        )


def test_round_ledger_limits_cannot_expand_on_restart(tmp_path, monkeypatch):
    repo, _ = _repo_with_prior_ledger(tmp_path, monkeypatch)
    _write_authorization(repo, tokens=100_000, requests=20, elapsed=1800)
    round2_dir = repo / ac.round_runtime_dir("round2")
    round2_dir.mkdir(parents=True)
    with BudgetLedger.open(
        round2_dir / ac.LEDGER_FILENAME,
        limits=BudgetLimits(max_tokens=100_000, max_requests=20, max_elapsed_seconds=1800.0),
    ):
        pass
    # A later caller presenting LARGER caps than the persisted/authorized
    # ones is refused: restart can never expand a round budget.
    with pytest.raises((BudgetError, AcceptanceError), match="limits_cannot_expand|round_authorization_caps_mismatch"):
        run_acceptance(
            repo,
            round2_dir,
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=_fake_child(round2_dir, []),
            budget_limits=BudgetLimits(max_tokens=200_000, max_requests=30, max_elapsed_seconds=5400.0),
            round_id="round2",
        )


def test_failure_creates_no_automatic_retry_round(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    calls: list = []
    with pytest.raises(AcceptanceError, match="outer_run_incomplete"):
        run_acceptance(
            repo,
            runtime,
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=_fake_child(runtime, calls, behaviors={"cfg1": {"status": "failed"}}),
        )
    assert not list((repo / ".runtime").glob("acceptance-us009-round*"))


def test_default_round_refuses_old_stopped_ledger(tmp_path, monkeypatch):
    repo, runtime = _repo_with_prior_ledger(tmp_path, monkeypatch)
    calls: list = []
    with pytest.raises(AcceptanceError, match="acceptance_ledger_stopped"):
        run_acceptance(
            repo,
            runtime,
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=_fake_child(runtime, calls),
        )
    assert calls == []


def test_retry_refuses_active_prior_ledger_without_mutating_it(tmp_path, monkeypatch):
    """A nonterminal prior ledger (not stopped, even with empty reservations)
    can still spend: refuse before any new budget, and never mutate it."""
    repo, round1_runtime = _repo(tmp_path, monkeypatch)
    round1_runtime.mkdir(parents=True, exist_ok=True)
    ledger_path = round1_runtime / ac.LEDGER_FILENAME
    with BudgetLedger.open(ledger_path, limits=ac.acceptance_budget_limits()):
        pass  # active: not stopped
    before = ledger_path.read_bytes()
    _write_authorization(repo)
    round2_dir = repo / ac.round_runtime_dir("round2")
    calls: list = []
    with pytest.raises(AcceptanceError, match="prior_round_ledger_active"):
        run_acceptance(
            repo,
            round2_dir,
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=_fake_child(round2_dir, calls),
            round_id="round2",
        )
    assert calls == []
    assert ledger_path.read_bytes() == before
    assert not (round2_dir / ac.LEDGER_FILENAME).exists()


def test_round2_failure_checkpoint_preserves_prior_and_new_committed_cost(tmp_path, monkeypatch):
    """A failed authorized round2 checkpoints prior+new KNOWN committed cost
    and status-filtered run records (failed runs are not 'completed')."""
    repo, round1_runtime = _repo_with_prior_ledger(tmp_path, monkeypatch)
    _write_authorization(repo)
    round2_dir = repo / ac.round_runtime_dir("round2")
    calls: list = []
    with pytest.raises(AcceptanceError, match="outer_run_incomplete:cfg1"):
        run_acceptance(
            repo,
            round2_dir,
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=_fake_child(round2_dir, calls, behaviors={"cfg1": {"status": "failed"}}),
            round_id="round2",
        )
    checkpoint = json.loads((round2_dir / "acceptance-checkpoint.json").read_text())
    aggregate = checkpoint["aggregate_cost"]
    assert aggregate["prior_round_committed"] == {"requests": 17, "tokens": 81717}
    assert aggregate["this_round_committed"]["requests"] == 2  # cfg0 + cfg1
    assert aggregate["total_committed"]["requests"] == 19
    statuses = {run["identity"]["config_hash"][:8]: run["status"] for run in checkpoint["runs"]}
    assert set(statuses.values()) == {"completed", "failed"}
    assert len(checkpoint["completed_runs"]) == 1  # only cfg0


# ------------------------------------------------------- budget lower bound
def test_insufficient_remaining_requests_stop_before_any_child(tmp_path, monkeypatch):
    """4 configs x 6 main generations = 24 main requests minimum; with fewer
    remaining, the acceptance stops with remaining_budget_insufficient and
    spends NOTHING."""
    repo, runtime = _repo(tmp_path, monkeypatch)
    runtime.mkdir(parents=True, exist_ok=True)
    with BudgetLedger.open(
        runtime / ac.LEDGER_FILENAME, limits=ac.acceptance_budget_limits()
    ) as ledger:
        for _ in range(7):  # 23 remaining < 24 minimum main
            reservation = ledger.reserve(10, 10)
            ledger.commit(reservation.reservation_id, ModelUsage(input_tokens=10, output_tokens=10))
    calls: list = []
    with pytest.raises(AcceptanceError, match="remaining_budget_insufficient"):
        run_acceptance(
            repo, runtime,
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=_fake_child(runtime, calls),
        )
    assert calls == []
    assert not (runtime / "runs" / "cfg0" / "run-result.json").exists()
    checkpoint = json.loads((runtime / "acceptance-checkpoint.json").read_text())
    assert "remaining_budget_insufficient" in checkpoint["reason"]


def test_lower_bound_stops_before_next_child_preserving_completed_run(tmp_path, monkeypatch):
    """After cfg0 completes, fewer than 3x6=18 main requests remain: stop
    BEFORE launching cfg1; cfg0's complete artifacts stay intact."""
    repo, runtime = _repo(tmp_path, monkeypatch)
    runtime.mkdir(parents=True, exist_ok=True)
    with BudgetLedger.open(
        runtime / ac.LEDGER_FILENAME, limits=ac.acceptance_budget_limits()
    ) as ledger:
        for _ in range(6):  # 24 remain: enough for the standalone check
            reservation = ledger.reserve(10, 10)
            ledger.commit(reservation.reservation_id, ModelUsage(input_tokens=10, output_tokens=10))

    def hungry_child(spec, spec_path, *, repo_root):
        result = _fake_child(runtime, [])(spec, spec_path, repo_root=repo_root)
        with BudgetLedger.open(runtime / ac.LEDGER_FILENAME) as ledger:
            for _ in range(9):  # 14 remain after cfg0 < 18 needed for cfg1..3
                reservation = ledger.reserve(10, 10)
                ledger.commit(reservation.reservation_id, ModelUsage(input_tokens=10, output_tokens=10))
        return result

    calls: list = []

    def child(spec, spec_path, *, repo_root):
        calls.append(spec.run_id)
        return hungry_child(spec, spec_path, repo_root=repo_root)

    with pytest.raises(AcceptanceError, match="remaining_budget_insufficient"):
        run_acceptance(
            repo, runtime,
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=child,
        )
    assert calls == ["cfg0"]  # cfg1 never launched, no wasted calls
    result_path = runtime / "runs" / "cfg0" / "run-result.json"
    assert json.loads(result_path.read_text())["status"] == "completed"
    assert not (runtime / "runs" / "cfg1").exists()
    checkpoint = json.loads((runtime / "acceptance-checkpoint.json").read_text())
    assert "remaining_budget_insufficient" in checkpoint["reason"]
