"""Exercise real GenericLLM metrics shape and honest job outcome counters."""
from __future__ import annotations

import json
from types import SimpleNamespace

from research.search.acceptance_inner import job_outcome_counts, make_generation_outcome_sink

def test_outcome_sink_reads_provider_usage_inside_actual_generic_llm_envelope(tmp_path):
    solver = SimpleNamespace(last_generation_feedback=None)
    sink = make_generation_outcome_sink(run_dir=tmp_path, run_id="cfg0", secrets=[],
        stop_file=tmp_path/"stop-request.json", solver=solver)
    metrics = {"usage": {
        "finish_reason": "length", "prompt_tokens": 530,
        "completion_tokens": 4096, "total_tokens": 4626,
        "completion_tokens_details": {"reasoning_tokens": 4093},
        "reasoning_content": "private reasoning fixture must not be persisted",
        "usage_provenance": "provider",
    }, "completion_text": "", "prompt_messages": []}
    sink("draft", "", "", metrics)
    raw = (tmp_path/"generation-outcomes.jsonl").read_text()
    record = json.loads(raw)
    assert record["usage"]["finish_reason"] == "length"
    assert record["usage"]["prompt_tokens"] == 530
    assert record["usage"]["completion_tokens"] == 4096
    assert record["usage"]["reasoning_tokens"] == 4093
    assert record["truncated"] is True
    assert "private reasoning fixture" not in raw
    assert "finish_reason=length" in solver.last_generation_feedback

def test_cleanup_ack_for_never_started_job_does_not_count_as_execution():
    counts = job_outcome_counts([{
        "job_id": "job_cancelled_queued", "status": "cancelled",
        "worker_cleanup_verified": True, "execution_never_started": True,
        "source_identity_verified": False, "evidence_verified": False,
        "score": None, "scoring_result": "cancelled",
    }])
    assert counts["submitted"] == 1
    assert counts["executed_and_cleaned"] == 0
    assert counts["scored"] == 0
    assert counts["cancelled"] == 1

def test_scoring_failure_is_not_counted_as_code_execution_failure():
    counts = job_outcome_counts([{
        "job_id": "job_bad_prediction", "status": "failed",
        "worker_cleanup_verified": True, "source_identity_verified": True,
        "evidence_verified": False, "score": None,
        "scoring_result": "scoring_failed", "evaluation": {"reason": "prediction_malformed"},
    }])
    assert counts["scored"] == 0
    assert counts["execution_failed"] == 0
    assert counts.get("scoring_failed", 0) == 1


def test_acceptance_transport_rejects_other_model_before_reservation(tmp_path):
    import pytest
    from research.contracts.budget import BudgetError, BudgetLedger, BudgetLimits, CancellationToken
    from research.search.acceptance_inner import acceptance_request_profile
    from research.search.budget_transport import RequestGuard, TransportGuardConfig
    profile = acceptance_request_profile()
    calls = []
    def provider():
        calls.append(True)
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    with BudgetLedger.open(tmp_path / "model-pin-ledger.json", limits=BudgetLimits(max_tokens=20000, max_requests=2, max_elapsed_seconds=120)) as ledger:
        guard = RequestGuard(ledger, CancellationToken(), TransportGuardConfig(estimated_input_tokens=100, max_output_tokens=8192, required_request_profile=profile))
        with pytest.raises(BudgetError):
            guard.run(provider, model_kwargs={**profile, "model": "openai/not-k3"})
        assert calls == []
        assert ledger.snapshot()["committed"]["requests"] == 0
        assert ledger.snapshot()["open_reservations"] == []


def test_acceptance_transport_accepts_real_litellm_model_name(tmp_path):
    from research.contracts.budget import BudgetLedger, BudgetLimits, CancellationToken
    from research.search.acceptance_inner import acceptance_request_profile
    from research.search.budget_transport import RequestGuard, TransportGuardConfig
    # Captured real request trace and LiteLLMClient._query_client use openai/k3,
    # even though the configured client model_id is simply k3.
    profile = acceptance_request_profile()
    calls = []
    def provider():
        calls.append(True)
        return {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    with BudgetLedger.open(tmp_path / "real-model-ledger.json", limits=BudgetLimits(max_tokens=20000, max_requests=2, max_elapsed_seconds=120)) as ledger:
        guard = RequestGuard(ledger, CancellationToken(), TransportGuardConfig(estimated_input_tokens=100, max_output_tokens=8192, required_request_profile=profile))
        guard.run(provider, model_kwargs={**profile, "model": "openai/k3"})
        assert calls == [True]
        assert ledger.snapshot()["committed"] == {"requests": 1, "tokens": 2}
