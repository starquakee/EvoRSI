"""Reserve main-candidate requests; skip optional debug without fake outcomes."""
from __future__ import annotations

import json

from research.contracts.budget import BudgetLimits
from research.search import acceptance as ap
from research.search import acceptance_config as ac
from research.search import acceptance_inner as ai
from research.tests.test_acceptance_controller import FakeAlgo, VECTORS
from test_acceptance_repair import _offline_env, _fake_sandbox, _Stream, LITELLM_MODULE

def test_no_debug_request_when_entire_outer_main_budget_has_no_slack(tmp_path, monkeypatch):
    _offline_env(tmp_path, monkeypatch)
    posts = _fake_sandbox(monkeypatch)
    calls = []
    def completion(**kwargs):
        calls.append(kwargs)
        return _Stream("", {"prompt_tokens":10, "completion_tokens":20})
    monkeypatch.setattr(LITELLM_MODULE+".completion_fn", completion)
    def child(spec, spec_path, *, repo_root):
        ai.run_inner(spec_path, repo_root=repo_root)
        return json.loads((repo_root/spec.run_dir/"run-result.json").read_text())
    runtime = tmp_path/ac.RUNTIME_DIR
    try:
        ap.run_acceptance(tmp_path, runtime, algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=child, budget_limits=BudgetLimits(max_tokens=200000,max_requests=24,max_elapsed_seconds=5400))
    except ap.AcceptanceError:
        pass
    # Four inner runs need 24 main calls. After one invalid draft, 23 requests
    # remain for 23 main candidates. A debug would make completion impossible.
    assert len(calls) == 6
    ledger = json.loads((runtime/ac.LEDGER_FILENAME).read_text())
    assert ledger["committed"] == {"requests":6, "tokens":180}
    assert ledger["reservations"] == []
    assert posts == []
    run = runtime/"runs/cfg0"
    assert len((run/"generation-outcomes.jsonl").read_text().splitlines()) == 6
    attempts = [json.loads(line) for line in (run/"request-trace.jsonl").read_text().splitlines()]
    assert [entry["operator"] for entry in attempts] == ["draft"] * 6
    controls = [json.loads(line) for line in (run/"budget-control.jsonl").read_text().splitlines()]
    assert any(entry["event"] == "debug_skipped" for entry in controls)
    result = json.loads((run/"run-result.json").read_text())
    assert result["status"] == "failed"  # all fake drafts are empty, not a fabricated success
    assert result["main_candidates"] == 6
    assert result["generations_completed"] == 3
    assert result["best_score"] is None
