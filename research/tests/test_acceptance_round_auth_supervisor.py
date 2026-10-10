"""The orchestration API must enforce authorization before creating budgets."""
from __future__ import annotations

import json

from research.contracts.budget import BudgetError, BudgetLedger, BudgetLimits
from research.search import acceptance_config as ac
from research.search.acceptance import AcceptanceError, run_acceptance
from research.tests.test_acceptance_controller import COMMIT, FakeAlgo, VECTORS, _fake_child, _repo

REFUSALS = (AcceptanceError, BudgetError, ac.RoundAuthorizationRefused)

def invoke(repo, runtime, calls, **kwargs):
    return run_acceptance(repo, runtime, algo_factory=lambda: FakeAlgo(VECTORS),
        child_runner=_fake_child(runtime, calls), **kwargs)

def test_direct_retry_orchestration_requires_authorization_before_new_ledger(tmp_path, monkeypatch):
    repo, _ = _repo(tmp_path, monkeypatch)
    runtime = repo/ac.round_runtime_dir("round2")
    calls = []
    try:
        invoke(repo, runtime, calls, round_id="round2")
    except REFUSALS:
        pass
    assert calls == []
    assert not (runtime/ac.LEDGER_FILENAME).exists()

def test_default_round_cannot_expand_original_pins_through_library_argument(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    calls = []
    try:
        invoke(repo, runtime, calls, budget_limits=BudgetLimits(max_tokens=200001, max_requests=31, max_elapsed_seconds=5401))
    except REFUSALS:
        pass
    assert calls == []
    assert not (runtime/ac.LEDGER_FILENAME).exists()

def test_stopped_original_refusal_preserves_prior_checkpoint_bytes(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    runtime.mkdir(parents=True, exist_ok=True)
    with BudgetLedger.open(runtime/ac.LEDGER_FILENAME, limits=ac.acceptance_budget_limits()) as ledger:
        ledger.stop("prior_authorized_stop")
    checkpoint = runtime/ac.CHECKPOINT_FILENAME
    checkpoint.write_text(json.dumps({"schema":"prior-run-evidence", "reason":"prior_authorized_stop"}))
    before = checkpoint.read_bytes()
    calls = []
    try:
        invoke(repo, runtime, calls)
    except REFUSALS:
        pass
    assert calls == []
    assert checkpoint.read_bytes() == before

def test_retry_runtime_symlink_cannot_alias_and_modify_original_tree(tmp_path, monkeypatch):
    repo, original = _repo(tmp_path, monkeypatch)
    original.mkdir(parents=True, exist_ok=True)
    with BudgetLedger.open(original/ac.LEDGER_FILENAME, limits=ac.acceptance_budget_limits()) as ledger:
        ledger.stop("prior_authorized_stop")
    checkpoint = original/ac.CHECKPOINT_FILENAME
    checkpoint.write_text(json.dumps({"schema":"prior-run-evidence"}))
    before = checkpoint.read_bytes()
    authorization = repo/".runtime"/ac.ROUND_AUTHORIZATION_FILENAME
    authorization.write_text(json.dumps({"schema":ac.ROUND_AUTHORIZATION_SCHEMA,
        "implementation_commit":COMMIT,"authorize_retry_round":True,"round_id":"round2",
        "caps":{"max_tokens":200000,"max_requests":30,"max_elapsed_seconds":5400}}))
    runtime = repo/ac.round_runtime_dir("round2")
    runtime.symlink_to(original, target_is_directory=True)
    calls = []
    try:
        invoke(repo, runtime, calls, round_id="round2")
    except REFUSALS:
        pass
    assert calls == []
    assert checkpoint.read_bytes() == before


def authorize_round2(repo):
    payload = {"schema":ac.ROUND_AUTHORIZATION_SCHEMA,
        "implementation_commit":COMMIT,"authorize_retry_round":True,"round_id":"round2",
        "caps":{"max_tokens":200000,"max_requests":30,"max_elapsed_seconds":5400}}
    (repo/".runtime"/ac.ROUND_AUTHORIZATION_FILENAME).write_text(json.dumps(payload))

def test_retry_checks_prior_cost_before_any_new_request(tmp_path, monkeypatch):
    repo, _ = _repo(tmp_path, monkeypatch)
    authorize_round2(repo)
    runtime = repo/ac.round_runtime_dir("round2")
    calls = []
    try:
        invoke(repo, runtime, calls, round_id="round2")
    except REFUSALS:
        pass
    assert calls == [], "Missing prior cost must be rejected before spending new budget"

def test_retry_refuses_unknown_prior_usage_without_mutating_original(tmp_path, monkeypatch):
    repo, original = _repo(tmp_path, monkeypatch)
    original.mkdir(parents=True, exist_ok=True)
    ledger_path = original/ac.LEDGER_FILENAME
    with BudgetLedger.open(ledger_path, limits=ac.acceptance_budget_limits()) as ledger:
        ledger.reserve(10, 10)
        ledger.stop("prior_unknown_usage")
    before = ledger_path.read_bytes()
    authorize_round2(repo)
    runtime = repo/ac.round_runtime_dir("round2")
    calls = []
    try:
        invoke(repo, runtime, calls, round_id="round2")
    except REFUSALS:
        pass
    assert calls == [], "Open prior reservation must not disappear from aggregate accounting"
    assert ledger_path.read_bytes() == before

def test_authorized_round2_completes_real_controller_cache_path_and_aggregates_cost(tmp_path, monkeypatch):
    from research.contracts.budget import ModelUsage
    repo, original = _repo(tmp_path, monkeypatch)
    original.mkdir(parents=True, exist_ok=True)
    ledger_path = original/ac.LEDGER_FILENAME
    with BudgetLedger.open(ledger_path, limits=ac.acceptance_budget_limits()) as ledger:
        reservation = ledger.reserve(10, 10)
        ledger.commit(reservation.reservation_id, ModelUsage(input_tokens=10, output_tokens=10))
        ledger.stop("prior_authorized_stop")
    before = ledger_path.read_bytes()
    authorize_round2(repo)
    runtime = repo/ac.round_runtime_dir("round2")
    calls = []
    result = invoke(repo, runtime, calls, round_id="round2")
    assert result["verdict"] == "complete"
    assert calls == ["cfg0", "cfg1", "cfg2", "cfg3"]
    assert result["aggregate_cost"]["total_committed"] == {"requests":5, "tokens":420}
    assert ledger_path.read_bytes() == before
    assert result["cache_events"][0]["incremental_model_cost"] == {"requests":0, "tokens":0}
