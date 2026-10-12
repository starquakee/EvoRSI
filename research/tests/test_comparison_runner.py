"""EXP-005 offline tests: comparison runner spec, transport guard, control.

Research-venv safe (no dojo imports): the runner keeps heavy third-party
imports function-local. All ledgers are tmp-path fixtures bound to the real
frozen roster (27 cells from the committed data manifest); no model calls,
no live ledger, no sandbox.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from research.comparison import budget as cb
from research.comparison import runner as cr
from research.comparison.budget import (
    CellBudgetExhausted,
    ComparisonBudgetError,
    ComparisonLedger,
    ComparisonLedgerStopped,
    GlobalBudgetExhausted,
    InnerRequestLimitReached,
    ProviderUsage,
)
from research.comparison.control import (
    ComparisonMainBudgetControl,
    comparison_termination_reason,
    run_comparison_search,
)
from research.comparison.data import load_task_bindings
from research.comparison.protocol import (
    FIXED_THETA,
    ComparisonSpecError,
    ModelProfile,
    decode_theta,
)
from research.comparison.runner import (
    ComparisonReviewRefused,
    ComparisonRunnerError,
    ComparisonRunSpec,
    run_dir_name,
    select_incumbent,
    verify_task_data_identity,
)
from research.comparison.specs import InnerRunSpec, iter_cells
from research.comparison.transport import (
    ComparisonRequestGuard,
    comparison_request_profile,
    install_comparison_guard,
)
from research.contracts.budget import CancellationToken, Cancelled
from research.evaluator.service import EvaluatorRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT = "c" * 40


def _cells():
    return iter_cells(load_task_bindings(), COMMIT)


def _spec(config_index: int = 0, *, cell_index: int = 0, **overrides) -> ComparisonRunSpec:
    cell = _cells()[cell_index]
    run = cell.inner_run(config_index)
    base = dict(
        run=run,
        decoded=decode_theta(FIXED_THETA),
        run_dir=str(Path(".runtime/comparison-istratde/runs") / run_dir_name(run)),
        ledger_path=".runtime/comparison-istratde/comparison-ledger.json",
        runner_commit=COMMIT,
    )
    base.update(overrides)
    return ComparisonRunSpec(**base)


@pytest.fixture()
def ledger(tmp_path):
    handle = ComparisonLedger.open(tmp_path / "ledger.json", cells=_cells())
    handle.start_clock()
    yield handle
    handle.close()


# --------------------------------------------------------------------- spec
def test_spec_roundtrip_and_identity_binding():
    spec = _spec()
    spec.validate()
    raw = json.loads(json.dumps(spec.to_dict()))
    assert ComparisonRunSpec.from_dict(raw) == spec
    assert spec.run.run_key() == "banknote:11:FIXED:0"
    # The run identity is registered in the ledger experiment roster.
    assert spec.run.identity() == InnerRunSpec(
        cell=spec.run.cell, config_index=0, inner_seed=spec.run.inner_seed
    ).identity()


@pytest.mark.parametrize(
    "mutate,rule",
    [
        (lambda d: d.update(schema="other"), "spec_schema_mismatch"),
        (lambda d: d.update(inner_seed=d["inner_seed"] ^ 1), "spec_inner_seed_mismatch"),
        (lambda d: d.update(run_identity="0" * 64), "spec_run_identity_mismatch"),
        (lambda d: d.update(cell_identity="0" * 64), "spec_cell_identity_mismatch"),
        (lambda d: d.update(run_dir="/abs/path"), "spec_path_not_relative"),
        (
            lambda d: d.update(run_dir=".runtime/comparison-istratde/other/x"),
            "spec_run_dir_outside_runtime",
        ),
        (
            lambda d: d.update(run_dir=".runtime/comparison-istratde/runs/wrong"),
            "spec_run_dir_name_mismatch",
        ),
        (lambda d: d.update(ledger_path=".runtime/other.json"), "spec_ledger_path_mismatch"),
        (lambda d: d.update(resource_type="tpu"), "spec_resource_type_invalid"),
        (lambda d: d.update(job_timeout_seconds=10), "spec_job_timeout_invalid"),
        (
            lambda d: d.update(decoded={"crossover_prob": 0.5}),
            "spec_decoded_dims_mismatch",
        ),
        (
            lambda d: d.update(
                decoded={"crossover_prob": 0.5, "score_weight": 0.0,
                         "delta_weight": 0.0, "novelty_weight": 0.0,
                         "max_debug_depth": 1}
            ),
            "spec_theta_invalid:empty_parent_selection_weights",
        ),
    ],
)
def test_spec_fail_closed(mutate, rule):
    data = _spec().to_dict()
    mutate(data)
    with pytest.raises(ComparisonRunnerError, match=rule):
        ComparisonRunSpec.from_dict(data)


def test_spec_runner_commit_invalid_rejected_at_cell_binding():
    data = _spec().to_dict()
    data["runner_commit"] = "xyz"
    with pytest.raises(ComparisonSpecError, match="source_commit_invalid"):
        ComparisonRunSpec.from_dict(data)


def test_spec_tampered_task_binding_detected():
    data = _spec().to_dict()
    data["task_binding"]["train_csv_sha256"] = "0" * 64
    with pytest.raises(ComparisonRunnerError, match="spec_run_identity_mismatch"):
        ComparisonRunSpec.from_dict(data)


def test_spec_config_index_out_of_range():
    cell = _cells()[0]
    with pytest.raises(ComparisonSpecError, match="config_index_out_of_range"):
        cell.inner_run(8)


# --------------------------------------------------------------------- gate
def test_live_gate_refuses_before_any_activity(tmp_path):
    # EXP-007: the gate is wired; without the ignored supervisor review
    # every live attempt refuses BEFORE any ledger/auth/model activity.
    with pytest.raises(ComparisonReviewRefused, match="comparison_review_absent"):
        cr.check_comparison_review(REPO_ROOT)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec().to_dict()), encoding="utf-8")
    assert cr.main(["run", "--spec", str(spec_path), "--repo-root", str(REPO_ROOT)]) == 2
    # Nothing live was created: no ledger, no review file, no result.
    runtime = REPO_ROOT / ".runtime/comparison-istratde"
    assert not (runtime / "comparison-ledger.json").exists()
    assert not (runtime / "supervisor-review.json").exists()
    assert not (tmp_path / "run-result.json").exists()


# ---------------------------------------------------------- dataset identity
def test_verify_task_data_identity_real_repo():
    registry = EvaluatorRegistry.load()
    for binding in load_task_bindings().values():
        evidence = verify_task_data_identity(
            binding, repo_root=REPO_ROOT, registry=registry
        )
        assert evidence["registry_task_id"] == f"comparison_{binding.task_id}"
        assert evidence["split"] == "validation"
        assert evidence["worker_data_public_path"] == (
            f"/mnt/rsi_data/comparison/{binding.task_id}/data/public"
        )
        assert evidence["validation_labels_sha256"] == binding.validation_labels_sha256


def test_verify_task_data_identity_hash_mismatch(tmp_path):
    binding = load_task_bindings()["banknote"]
    registry = EvaluatorRegistry.load()
    public = tmp_path / "tasks/comparison/banknote/data/public"
    public.mkdir(parents=True)
    (public / "train.csv").write_bytes(b"tampered")
    (public / "test.csv").write_bytes(b"x")
    with pytest.raises(ComparisonRunnerError, match="task_data_hash_mismatch:train.csv"):
        verify_task_data_identity(binding, repo_root=tmp_path, registry=registry)


def test_verify_task_data_identity_missing(tmp_path):
    binding = load_task_bindings()["wdbc"]
    with pytest.raises(ComparisonRunnerError, match="task_data_missing:train.csv"):
        verify_task_data_identity(
            binding, repo_root=tmp_path, registry=EvaluatorRegistry.load()
        )


# ------------------------------------------------------------------ profile
def test_request_profile_matches_pinned_model_profile():
    profile = ModelProfile()
    assert comparison_request_profile() == {
        "model": "openai/k3",
        "base_url": "https://api.kimi.com/coding/v1",
        "stream": True,
        "max_tokens": 8192,
        "temperature": 1.0,
        "top_p": 0.95,
        "reasoning_effort": "low",
        "allowed_openai_params": ["reasoning_effort"],
        "max_retries": 0,
        "num_retries": 0,
    }
    assert profile.transport_retries == 0
    assert profile.allowed_openai_params == ("reasoning_effort",)


# --------------------------------------------------------------- guard units
def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "write code"},
    ]


def _kwargs() -> dict[str, Any]:
    return dict(comparison_request_profile())


def _guard(ledger, run, *, kind="main", debug_budget=None, token=None, sink=None):
    return ComparisonRequestGuard(
        ledger,
        token or CancellationToken(deadline_epoch=time.time() + 600),
        run,
        kind=kind,
        debug_budget=debug_budget,
        attempt_sink=sink,
    )


def test_guard_commits_exact_provider_usage(ledger):
    run = _spec().run
    seen = []

    def thunk():
        return {"usage": {"prompt_tokens": 5, "completion_tokens": 3,
                          "total_tokens": 8, "usage_provenance": "provider"}}

    response = _guard(ledger, run, sink=lambda *a: seen.append(a)).run(
        thunk, messages=_messages(), model_kwargs=_kwargs()
    )
    assert response["usage"]["total_tokens"] == 8
    snapshot = ledger.snapshot()
    assert snapshot["global"]["committed"] == {"requests": 1, "tokens": 8}
    assert snapshot["reservations"] == []
    assert len(seen) == 1
    # The reservation covered the FULL exact prompt bytes + pinned 8192.
    reservation = seen[0][0]
    assert reservation.tokens > 8192
    record = ledger.snapshot()["requests"][0]
    assert record["kind"] == "main"
    assert record["input_tokens"] == 5 and record["output_tokens"] == 3


def test_guard_profile_drift_refused_before_reservation(ledger):
    run = _spec().run
    kwargs = _kwargs()
    kwargs["temperature"] = 0.5
    with pytest.raises(ComparisonBudgetError, match="request_profile_mismatch:temperature"):
        _guard(ledger, run).run(lambda: None, messages=_messages(), model_kwargs=kwargs)
    snapshot = ledger.snapshot()
    assert snapshot["global"]["committed"]["requests"] == 0
    assert snapshot["stopped"] is None


@pytest.mark.parametrize("key", ["stream", "max_tokens", "top_p", "reasoning_effort", "model"])
def test_guard_every_profile_pin_enforced(ledger, key):
    run = _spec().run
    kwargs = _kwargs()
    kwargs[key] = "bad"
    with pytest.raises(ComparisonBudgetError, match=f"request_profile_mismatch:{key}"):
        _guard(ledger, run).run(lambda: None, messages=_messages(), model_kwargs=kwargs)
    assert ledger.snapshot()["global"]["committed"]["requests"] == 0


def test_guard_transport_failure_retains_reservation_and_stops(ledger):
    run = _spec().run

    def thunk():
        raise OSError("boom")

    with pytest.raises(OSError):
        _guard(ledger, run).run(thunk, messages=_messages(), model_kwargs=_kwargs())
    snapshot = ledger.snapshot()
    assert snapshot["stopped"]["reason"].startswith("unknown_usage:")
    assert snapshot["global"]["committed"]["requests"] == 1
    assert snapshot["requests"][0]["usage_known"] is False
    assert snapshot["requests"][0]["charged_tokens"] == snapshot["requests"][0]["reserved_tokens"]
    # No next request is possible.
    with pytest.raises(ComparisonLedgerStopped):
        _guard(ledger, run).run(thunk, messages=_messages(), model_kwargs=_kwargs())


def test_guard_missing_provider_usage_is_unknown_and_stops(ledger):
    run = _spec().run

    def thunk():
        return {"usage": {"usage_provenance": "estimated"}}

    with pytest.raises(Exception):
        _guard(ledger, run).run(thunk, messages=_messages(), model_kwargs=_kwargs())
    assert ledger.snapshot()["stopped"]["reason"].startswith("unknown_usage:")


def test_guard_deadline_refuses_before_reservation(ledger):
    run = _spec().run
    token = CancellationToken(deadline_epoch=time.time() - 1)
    with pytest.raises(Cancelled):
        _guard(ledger, run, token=token).run(
            lambda: None, messages=_messages(), model_kwargs=_kwargs()
        )
    assert ledger.snapshot()["global"]["committed"]["requests"] == 0


def test_guard_debug_budget_total_cap(ledger):
    run = _spec().run

    def thunk():
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2, "usage_provenance": "provider"}}

    guard = _guard(ledger, run, kind="debug", debug_budget=2)
    guard.run(thunk, messages=_messages(), model_kwargs=_kwargs())
    guard.run(thunk, messages=_messages(), model_kwargs=_kwargs())
    with pytest.raises(InnerRequestLimitReached, match="inner_debug_limit_reached"):
        guard.run(thunk, messages=_messages(), model_kwargs=_kwargs())
    snapshot = ledger.snapshot()
    run_recs = [r for r in snapshot["requests"] if r["run_identity"] == run.identity()]
    assert sum(1 for r in run_recs if r["kind"] == "debug") == 2


def test_guard_debug_budget_zero_refuses(ledger):
    run = _spec().run
    with pytest.raises(InnerRequestLimitReached):
        _guard(ledger, run, kind="debug", debug_budget=0).run(
            lambda: None, messages=_messages(), model_kwargs=_kwargs()
        )


def test_guard_main_cap_six(ledger):
    run = _spec().run

    def thunk():
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1,
                          "total_tokens": 2, "usage_provenance": "provider"}}

    guard = _guard(ledger, run)
    for _ in range(6):
        guard.run(thunk, messages=_messages(), model_kwargs=_kwargs())
    with pytest.raises(InnerRequestLimitReached, match="inner_main_requests_exhausted"):
        guard.run(thunk, messages=_messages(), model_kwargs=_kwargs())


def test_install_guard_refuses_unbudgeted_operators(ledger):
    run = _spec().run
    client = SimpleNamespace(request_guard=None, client=None)
    install_comparison_guard(
        client, ledger, CancellationToken(deadline_epoch=time.time() + 600),
        run, operator="analyze", debug_budget=2,
    )
    with pytest.raises(ComparisonBudgetError, match="operator_not_budgeted:analyze"):
        client.request_guard.run(lambda: None, messages=_messages(), model_kwargs=_kwargs())
    assert ledger.snapshot()["global"]["committed"]["requests"] == 0
    with pytest.raises(ComparisonBudgetError, match="guard_already_installed"):
        install_comparison_guard(
            client, ledger, CancellationToken(), run, operator="draft", debug_budget=2
        )


def test_terminate_cell_public_api(ledger):
    cell = _spec().run.cell
    ledger.terminate_cell(cell.identity(), "cell_main_budget_insufficient")
    with pytest.raises(CellBudgetExhausted, match="cell_terminated"):
        ledger.reserve(_spec().run, "main", estimated_input_tokens=100)
    # Another cell is unaffected.
    other = _spec(cell_index=1).run
    reservation = ledger.reserve(other, "main", estimated_input_tokens=100)
    ledger.commit(reservation.request_id, ProviderUsage(1, 1))
    snapshot = ledger.snapshot()
    assert snapshot["stopped"] is None
    assert snapshot["cells"][cell.cell_key()]["terminated"]["reason"] == (
        "cell_main_budget_insufficient"
    )


# ------------------------------------------------------------- budget control
def _control(ledger, run, *, debug_budget, tmp_path):
    return ComparisonMainBudgetControl(
        ledger, run, debug_budget=debug_budget,
        control_log_path=tmp_path / "control.jsonl",
    )


def test_control_require_main_counts_and_debug_denials(ledger, tmp_path):
    run = _spec().run
    control = _control(ledger, run, debug_budget=2, tmp_path=tmp_path)
    for _ in range(6):
        control.require_main("draft")
    assert control.main_needed() == 0
    control.note_debug_attempt()
    control.note_debug_attempt()
    assert control.debug_denial_reason() is not None
    assert "inner_debug_budget_exhausted" in control.debug_denial_reason()


def test_control_debug_disabled_at_theta_depth_zero(ledger, tmp_path):
    control = _control(ledger, _spec().run, debug_budget=0, tmp_path=tmp_path)
    assert control.debug_denial_reason() == "debug_disabled:theta_max_debug_depth_0"
    control.record_debug_skipped(control.debug_denial_reason())
    log = (tmp_path / "control.jsonl").read_text()
    assert "debug_skipped" in log and "theta_max_debug_depth_0" in log


def test_reserve_cell_token_exhaustion_terminates_only_that_cell(ledger, tmp_path):
    run = _spec().run
    cell = run.cell
    other = cell.inner_run(1)
    # Drain most of the cell's TOKEN quota via another config (actual usage
    # is what commits; usage stays within each conservative reservation).
    for _ in range(6):
        r = ledger.reserve(other, "main", estimated_input_tokens=30000)
        ledger.commit(r.request_id, ProviderUsage(30000, 1000))
    # Cell committed 186000 tokens; a 13000+8192 reservation crosses 200000.
    with pytest.raises(CellBudgetExhausted, match="cell_tokens_exhausted"):
        ledger.reserve(run, "main", estimated_input_tokens=13000)
    snapshot = ledger.snapshot()
    assert snapshot["stopped"] is None
    assert snapshot["cells"][cell.cell_key()]["terminated"]["reason"] == (
        "cell_tokens_exhausted"
    )
    # Another cell is unaffected.
    other_cell_run = _spec(cell_index=1).run
    reservation = ledger.reserve(other_cell_run, "main", estimated_input_tokens=100)
    ledger.commit(reservation.request_id, ProviderUsage(1, 1))
    assert ledger.snapshot()["stopped"] is None


def test_control_debug_denied_without_slack_beyond_mains(ledger, tmp_path):
    run = _spec().run
    cell = run.cell
    # Drain the cell so exactly the 6 remaining mains fit but no debug slack.
    for i in range(1, 8):
        other = cell.inner_run(i)
        for _ in range(6):
            r = ledger.reserve(other, "main", estimated_input_tokens=100)
            ledger.commit(r.request_id, ProviderUsage(1, 1))
        for _ in range(2):
            r = ledger.reserve(other, "debug", estimated_input_tokens=100, debug_limit=2)
            ledger.commit(r.request_id, ProviderUsage(1, 1))
    for _ in range(2):
        r = ledger.reserve(run, "debug", estimated_input_tokens=100, debug_limit=2)
        ledger.commit(r.request_id, ProviderUsage(1, 1))
    # Cell committed 58; the 6 mains of this run fit exactly (defense in
    # depth: the frozen arithmetic guarantees this), debug needs 1 more.
    control = _control(ledger, run, debug_budget=2, tmp_path=tmp_path)
    denial = control.debug_denial_reason()
    assert denial is not None and "insufficient_for_optional_debug" in denial
    for _ in range(6):
        control.require_main("draft")  # admitted exactly at equality


# --------------------------------------------------------------- run control
class _StubSolver:
    """Minimal solver stand-in for run_comparison_search classification."""

    def __init__(self, error: BaseException | None = None):
        self.cfg = SimpleNamespace(execution_mode="generation", step_limit=12)
        self.state = SimpleNamespace(current_step=0, current_generation=0)
        self.before_operator = None
        self._error = error
        self.searched = False

    def debug_cycle(self, state, task, node):
        return state, [node], None

    def search(self, task, state):
        self.searched = True
        if self._error is not None:
            raise self._error

    def save_checkpoint(self):
        pass


def _run_control(ledger, solver, run, tmp_path, control=None):
    control = control or _control(ledger, run, debug_budget=2, tmp_path=tmp_path)
    return run_comparison_search(
        solver,
        SimpleNamespace(stop_requested=False),
        {},
        ledger=ledger,
        run=run,
        token=CancellationToken(deadline_epoch=time.time() + 600),
        budget_control=control,
        cancel_live_jobs=lambda: [{"cleanup_verified": True}],
        checkpoint_path=tmp_path / "checkpoint.json",
    )


def test_run_control_completed(ledger, tmp_path):
    solver = _StubSolver()
    termination = _run_control(ledger, solver, _spec().run, tmp_path)
    assert termination == "completed"
    assert solver.searched
    assert ledger.snapshot()["stopped"] is None
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["schema"] == "comparison-search-control.v1"
    assert checkpoint["cleanup_error"] is None


def test_run_control_cell_exhaustion_does_not_stop_globally(ledger, tmp_path):
    solver = _StubSolver(CellBudgetExhausted("cell_tokens_exhausted"))
    termination = _run_control(ledger, solver, _spec().run, tmp_path)
    assert termination == "CellBudgetExhausted:cell_tokens_exhausted"
    assert ledger.snapshot()["stopped"] is None


def test_run_control_inner_limit_does_not_stop_globally(ledger, tmp_path):
    solver = _StubSolver(InnerRequestLimitReached("inner_main_requests_exhausted"))
    termination = _run_control(ledger, solver, _spec().run, tmp_path)
    assert termination.startswith("InnerRequestLimitReached")
    assert ledger.snapshot()["stopped"] is None


def test_run_control_unexpected_error_stops_globally_sanitized(ledger, tmp_path):
    solver = _StubSolver(RuntimeError("provider secret detail sk-xyz\nmultiline"))
    termination = _run_control(ledger, solver, _spec().run, tmp_path)
    assert termination == "RuntimeError"  # provider text never persists
    stopped = ledger.snapshot()["stopped"]
    assert stopped is not None
    assert "sk-xyz" not in json.dumps(ledger.snapshot())


def test_run_control_global_exhaustion_stops(ledger, tmp_path):
    solver = _StubSolver(GlobalBudgetExhausted("tokens_exhausted"))
    termination = _run_control(ledger, solver, _spec().run, tmp_path)
    assert termination.startswith("GlobalBudgetExhausted")
    assert ledger.snapshot()["stopped"] is not None


def test_run_control_cleanup_failure_stops_globally(ledger, tmp_path):
    solver = _StubSolver()
    control = _control(ledger, _spec().run, debug_budget=2, tmp_path=tmp_path)
    with pytest.raises(ComparisonBudgetError, match="sandbox_cleanup_unproven"):
        run_comparison_search(
            solver,
            SimpleNamespace(stop_requested=False),
            {},
            ledger=ledger,
            run=_spec().run,
            token=CancellationToken(deadline_epoch=time.time() + 600),
            budget_control=control,
            cancel_live_jobs=lambda: [{"cleanup_verified": False}],
            checkpoint_path=tmp_path / "checkpoint.json",
        )
    assert ledger.snapshot()["stopped"]["reason"] == "sandbox_cleanup_unproven"
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["cleanup_error"] == "sandbox_cleanup_unproven"


def test_run_control_double_install_refused(ledger, tmp_path):
    solver = _StubSolver()
    solver.before_operator = lambda: None
    with pytest.raises(ComparisonBudgetError, match="search_control_already_installed"):
        _run_control(ledger, solver, _spec().run, tmp_path)


def test_run_control_debug_chain_mid_cap_is_graceful(ledger, tmp_path):
    """Mid-chain InnerRequestLimitReached truncates the chain truthfully."""
    run = _spec().run
    control = _control(ledger, run, debug_budget=2, tmp_path=tmp_path)
    solver = _StubSolver()

    def exploding_debug_cycle(state, task, node):
        raise InnerRequestLimitReached("inner_debug_limit_reached")

    solver.debug_cycle = exploding_debug_cycle

    def search(task, state):
        solver.searched = True
        state, path, fixed = solver.debug_cycle(state, task, "node")
        assert fixed is None and path == ["node"]

    solver.search = search
    termination = run_comparison_search(
        solver,
        SimpleNamespace(stop_requested=False),
        {},
        ledger=ledger,
        run=run,
        token=CancellationToken(deadline_epoch=time.time() + 600),
        budget_control=control,
        cancel_live_jobs=lambda: [{"cleanup_verified": True}],
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    assert termination == "completed"
    assert ledger.snapshot()["stopped"] is None
    log = (tmp_path / "control.jsonl").read_text()
    assert "debug_chain_capped" in log


def test_termination_reason_sanitization():
    assert comparison_termination_reason(RuntimeError("sk-secret\nmore")) == "RuntimeError"
    reason = comparison_termination_reason(Cancelled("deadline_exceeded"))
    assert reason == "Cancelled:deadline_exceeded"


# ------------------------------------------------------------- incumbent pick
class _Metric:
    def __init__(self, value, job_id=""):
        self.value = value
        self.info = {"job_id": job_id} if job_id else {}


class _Node:
    def __init__(self, nid, value, buggy=False):
        self.id = nid
        self.metric = _Metric(value)
        self.is_buggy = buggy
        self.code = f"# {nid}"


class _Journal:
    def __init__(self, nodes):
        self.nodes = nodes

    @staticmethod
    def is_root_node(node):
        return node.id == "root"


def test_select_incumbent_highest_validation_tie_earliest():
    nodes = [
        _Node("root", 0.0),
        _Node("a", 0.5),
        _Node("b", 0.7),
        _Node("c", 0.7),   # tie with b: earliest wins
        _Node("d", 0.9, buggy=True),  # buggy candidates never incumbents
        _Node("e", float("nan")),
    ]
    best = select_incumbent(_Journal(nodes))
    assert best is not None and best.id == "b"


def test_select_incumbent_none_when_all_failed():
    nodes = [_Node("root", 0.0), _Node("a", 0.9, buggy=True)]
    assert select_incumbent(_Journal(nodes)) is None


# --------------------------------------------------------------- config hash
def test_comparison_config_identity_binds_run_and_theta():
    spec = _spec()
    identity = cr.comparison_config_identity(spec)
    other = _spec(config_index=1)
    assert cr.comparison_config_identity(other) != identity
    tampered = ComparisonRunSpec(
        run=spec.run,
        decoded=dict(spec.decoded, crossover_prob=0.9),
        run_dir=spec.run_dir,
        ledger_path=spec.ledger_path,
        runner_commit=spec.runner_commit,
    )
    assert cr.comparison_config_identity(tampered) != identity
