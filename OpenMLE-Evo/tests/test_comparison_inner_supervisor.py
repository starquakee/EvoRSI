"""Independent full-Evo edge cases; only provider/sandbox are offline fakes."""
import json

import pytest

import test_comparison_inner as fixture
from research.comparison.budget import ComparisonBudgetError, ProviderUsage
from research.comparison.runner import ComparisonRunnerError


def test_mid_run_cell_cap_retains_incumbent_then_other_cell_completes(tmp_path, monkeypatch):
    base = tmp_path / "shared"
    cells = fixture._cells()
    cell = cells[fixture._cell_index("banknote")]

    def precommit(ledger):
        # Legal provider usage and inner quotas: 8 + 8 + 4 requests in OTHER
        # configurations of the same cell. No reservation is exceeded.
        for config, count in ((1, 8), (2, 8), (3, 4)):
            run = cell.inner_run(config)
            for index in range(count):
                kind = "main" if index < 6 else "debug"
                request = ledger.reserve(run, kind,
                    estimated_input_tokens=8192,
                    debug_limit=2 if kind == "debug" else None)
                ledger.commit(request.request_id, ProviderUsage(4000, 4000))
        assert ledger.snapshot()["cells"][cell.cell_key()]["committed"]["tokens"] == 160000

    with monkeypatch.context() as patch:
        first = fixture._run_inner_offline(base, patch, task_id="banknote",
            completion_usage={"prompt_tokens": 4000, "completion_tokens": 4000,
                              "total_tokens": 8000}, ledger_precommit=precommit)
    result = first["result"]
    assert result["status"] == "cell_terminated"
    assert 1 <= result["requests"]["main"] < 6
    assert result["incumbent"] is not None
    assert result["worker_cleanup_verified"] is True
    snapshot = fixture._ledger_view(first["ledger_path"])
    assert snapshot["stopped"] is None
    assert snapshot["cells"][cell.cell_key()]["terminated"] is not None
    first_bytes = (first["run_dir"] / "run-result.json").read_bytes()

    # This fixture currently recreates its source/data links. Remove only
    # those two temporary symlinks, never their targets or run evidence.
    for name in ("OpenMLE-Evo", "tasks"):
        link = base / name
        assert link.is_symlink()
        link.unlink()
    with monkeypatch.context() as patch:
        second = fixture._run_inner_offline(base, patch, task_id="wdbc",
            completion_usage={"prompt_tokens": 4000, "completion_tokens": 4000,
                              "total_tokens": 8000}, shared_counter={"n": 100})
    assert second["result"]["status"] == "completed"
    assert second["result"]["requests"]["main"] == 6
    assert second["result"]["generations_completed"] == 3
    assert second["result"]["worker_cleanup_verified"] is True
    assert fixture._ledger_view(second["ledger_path"])["stopped"] is None
    assert (first["run_dir"] / "run-result.json").read_bytes() == first_bytes


def test_depth_two_mid_chain_cap_preserves_six_mains(tmp_path, monkeypatch):
    # Main1 fails, debug2 succeeds (one slot spent); main3 fails, debug4
    # fails (last slot spent). Attempting the second repair in that second
    # chain must skip it without dropping the remaining main candidates.
    outcome = fixture._run_inner_offline(tmp_path / "mixed", monkeypatch,
        task_id="banknote", theta=(.5, 1., .4, .25, 2.),
        fail_markers=frozenset({"call-1", "call-3", "call-4"}))
    result = outcome["result"]
    assert result["status"] == "completed"
    assert result["requests"]["main"] == 6
    assert result["requests"]["debug"] == 2
    assert result["generations_completed"] == 3
    assert result["incumbent"] is not None
    assert len(outcome["calls"]) == 8
    assert fixture._ledger_view(outcome["ledger_path"])["stopped"] is None


@pytest.mark.parametrize("defect", ["source_identity", "cleanup", "evaluation_split", "source_identity_failed_job"])
def test_integrity_failure_inside_debug_stops_before_next_model_request(
    tmp_path, monkeypatch, defect
):
    import httpx
    base = tmp_path / defect
    real_transport = httpx.MockTransport
    real_completion = fixture._make_completion
    observed = {}

    def capture_completion(calls, **kwargs):
        observed["calls"] = calls
        return real_completion(calls, **kwargs)

    def mutate_transport(handler):
        def service(request):
            response = handler(request)
            target_job = "job_0000000000000001" if defect == "source_identity_failed_job" else "job_0000000000000002"
            if request.method == "GET" and request.url.path.endswith(target_job):
                body = response.json()
                if defect.startswith("source_identity"):
                    body["result"]["source_identity_verified"] = False
                elif defect == "cleanup":
                    body["result"]["worker_cleanup_verified"] = False
                else:
                    body["result"]["evaluation"]["split"] = "final"
                return httpx.Response(response.status_code, json=body)
            return response
        return real_transport(service)

    monkeypatch.setattr(fixture, "_make_completion", capture_completion)
    monkeypatch.setattr(httpx, "MockTransport", mutate_transport)
    try:
        fixture._run_inner_offline(base, monkeypatch, task_id="banknote",
            fail_markers=frozenset({"call-1"}))
    except (ComparisonBudgetError, ComparisonRunnerError):
        pass  # A durable integrity stop may raise before result assembly.
    snapshot = fixture._ledger_view(base / ".runtime/comparison-istratde/comparison-ledger.json")
    assert snapshot["stopped"] is not None
    assert len(observed["calls"]) == (1 if defect == "source_identity_failed_job" else 2)


def test_post_search_binding_stop_cannot_report_completed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    base = tmp_path/"binding-stop"
    original = fixture.cr._bind_incumbent
    reached = []
    def altered_score(incumbent, jobs, predictions, ledger):
        assert incumbent is not None
        reached.append(True)
        altered = SimpleNamespace(id=incumbent.id, code=incumbent.code,
            operators_used=incumbent.operators_used,
            metric=SimpleNamespace(value=.999, info=incumbent.metric.info))
        return original(altered, jobs, predictions, ledger)
    monkeypatch.setattr(fixture.cr, "_bind_incumbent", altered_score)
    outcome = None
    try:
        outcome = fixture._run_inner_offline(base, monkeypatch, task_id="banknote")
    except (ComparisonRunnerError, ComparisonBudgetError):
        pass
    assert reached == [True]
    snapshot = fixture._ledger_view(base/".runtime/comparison-istratde/comparison-ledger.json")
    assert snapshot["stopped"] is not None
    if outcome is not None and outcome["result"] is not None:
        assert outcome["result"]["status"] == "failed"
        assert outcome["result"]["incumbent"] is None


def test_unresolved_submission_intent_prevents_completed_result(tmp_path, monkeypatch):
    import research.adapters.sandbox_eval_client as sec
    base = tmp_path/"pending-intent"
    original = sec.SandboxEvalClient
    reached = []
    class PendingIntentClient(original):
        def __init__(self, *args, **kwargs):
            self.test_observer = kwargs["observer"]
            super().__init__(*args, **kwargs)
        def cancel_all_live(self):
            cleanup = super().cancel_all_live()
            # Model the durable submission-intent / job-registration gap:
            # there is no known job id yet to put in live_job_ids().
            reached.append(True)
            self.test_observer({"event": "submission_intent", "trace_id": "unresolved-submit"})
            return cleanup
    monkeypatch.setattr(sec, "SandboxEvalClient", PendingIntentClient)
    outcome = None
    try:
        outcome = fixture._run_inner_offline(base, monkeypatch, task_id="banknote")
    except (ComparisonRunnerError, ComparisonBudgetError):
        pass
    assert reached
    snapshot = fixture._ledger_view(base/".runtime/comparison-istratde/comparison-ledger.json")
    assert snapshot["stopped"] is not None
    if outcome is not None and outcome["result"] is not None:
        assert outcome["result"]["status"] == "failed"
        assert outcome["result"]["worker_cleanup_verified"] is False
