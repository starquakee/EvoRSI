"""US-009 sandbox task adapter tests (offline; fake client, real translation).

Only fully verified canonical evaluator evidence may become a score; every
other outcome is a deterministic failure that parse_eval_result maps to a
buggy node WITHOUT calling the analyze LLM. Evidence fixtures mirror the
real API shape recorded by the US-008 probes.
"""
from __future__ import annotations

import copy

import pytest

from research.adapters.sandbox_eval_client import SandboxResult
from research.contracts.budget import BudgetError, CancellationToken, Cancelled
from research.contracts.source_gate import load_policy
from research.search.sandbox_task import (
    AUX_EVAL_INFO,
    EXECUTION_OUTPUT,
    VALID_SOLUTION,
    VALIDATION_FITNESS,
    SandboxAcceptanceTask,
)

CODE = "print('hi')\n"
ANSWER_SHA256 = "f25ab07cfc53a420662e3ef899df728de124136c70277b66236ca06f08638579"


class FakeClient:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = []
        self.cancelled = False

    def submit_and_wait(self, source, **kwargs):
        self.calls.append((source, kwargs))
        outcome = self.behavior
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def cancel_all_live(self):
        self.cancelled = True
        return [{"job_id": "job_x", "cleanup_verified": True}]

    def live_job_ids(self):
        return frozenset()


def _gate(source: str) -> dict:
    import hashlib

    policy = load_policy()
    data = source.encode("utf-8", errors="surrogatepass")
    file_hash = hashlib.sha256(data).hexdigest()
    tree = hashlib.sha256()
    tree.update(b"main.py\0")
    tree.update(file_hash.encode("ascii") + b"\n")
    return {
        "allowed": True,
        "policy_version": policy.policy_version,
        "policy_sha256": policy.policy_sha256,
        "source_sha256": tree.hexdigest(),
        "entrypoint": "main.py",
        "entrypoint_sha256": file_hash,
    }


def _completed(score=0.55, source: str = CODE):
    return SandboxResult(
        job_id="job_1",
        status="completed",
        score=score,
        raw={
            "status": "completed",
            "completed_at": "2026-10-08T14:45:40.480165Z",
            "result": {
                "result": "success",
                "score": score,
                "source_gate": _gate(source),
                "source_identity_verified": True,
                "worker_cleanup_verified": True,
                "evaluation": {
                    "task_id": "hello_synth",
                    "split": "test",
                    "metric": "accuracy",
                    "direction": "maximize",
                    "evaluator": "external_trusted",
                    "score": score,
                    "row_count": 40,
                    "prediction_sha256": "p" * 64,
                    "answer_sha256": ANSWER_SHA256,
                    "registry_version": "evaluator-registry.v1",
                },
            },
        },
    )


def _task(behavior) -> SandboxAcceptanceTask:
    return SandboxAcceptanceTask(FakeClient(behavior), CancellationToken())  # type: ignore[arg-type]


def test_scored_completion_maps_to_validation_fitness():
    task = _task(_completed(0.55))
    state, result = task.step_task({}, CODE)
    assert result[VALIDATION_FITNESS] == 0.55
    assert result[VALID_SOLUTION] is True
    assert result[AUX_EVAL_INFO]["status"] == "completed"
    assert result[AUX_EVAL_INFO]["status_code"] == 200
    assert result[EXECUTION_OUTPUT].exit_code == 0
    # canonical evaluator evidence is propagated for the record
    evaluation = result[AUX_EVAL_INFO]["evaluation"]
    assert evaluation["metric"] == "accuracy"
    assert evaluation["direction"] == "maximize"
    assert evaluation["answer_sha256"] == ANSWER_SHA256
    assert task.jobs[0]["job_id"] == "job_1"
    assert task.jobs[0]["score"] == 0.55
    assert task.jobs[0]["evidence_verified"] is True
    # the sandbox submission used the isolated-stack contract
    kwargs = task._client.calls[0][1]
    assert kwargs["data_dir"] == "/mnt/rsi_data/hello_synth"
    assert kwargs["task_id"] == "hello_synth"


@pytest.mark.parametrize(
    "path,value",
    [
        (("result", "source_identity_verified"), False),
        (("result", "worker_cleanup_verified"), False),
        (("completed_at",), None),
        (("result", "source_gate"), None),
        (("result", "source_gate", "allowed"), False),
        (("result", "source_gate", "entrypoint_sha256"), "0" * 64),
        (("result", "source_gate", "source_sha256"), "0" * 64),
        (("result", "source_gate", "policy_version"), "unreviewed-policy"),
        (("result", "source_gate", "policy_sha256"), "0" * 64),
        (("result", "evaluation"), {}),
        (("result", "evaluation", "evaluator"), "candidate_claim"),
        (("result", "evaluation", "metric"), "logloss"),
        (("result", "evaluation", "task_id"), "other-task"),
        (("result", "evaluation", "direction"), "minimize"),
        (("result", "evaluation", "score"), 0.9),
        (("result", "evaluation", "prediction_sha256"), ""),
        (("result", "evaluation", "answer_sha256"), "0" * 64),
        (("result", "evaluation", "row_count"), 1),
        (("result", "evaluation", "registry_version"), "other-registry"),
    ],
)
def test_unverified_completion_evidence_never_scores(path, value):
    outcome = _completed(0.55)
    parent = outcome.raw
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    task = _task(outcome)
    _, result = task.step_task({}, CODE)
    assert result[VALID_SOLUTION] is False
    assert result[VALIDATION_FITNESS] is None
    assert result[AUX_EVAL_INFO]["status"] == "unknown"
    assert result[AUX_EVAL_INFO]["feedback"].startswith("evidence_unverified:")
    assert task.jobs[0]["evidence_verified"] is False


def test_submitted_source_is_hash_bound():
    """The gate evidence must bind the EXACT submitted bytes: submitting
    different code than the evidence claims fails verification."""
    outcome = _completed(0.55, source=CODE)
    task = _task(outcome)
    _, result = task.step_task({}, "print('tampered')\n")
    assert result[VALID_SOLUTION] is False


def test_scoring_failed_maps_to_deterministic_failure():
    raw = {
        "status": "failed",
        "result": {"result": "scoring_failed", "evaluation": {"reason": "prediction_oversized"}},
    }
    task = _task(SandboxResult("job_2", "failed", None, raw))
    _, result = task.step_task({}, "code")
    assert result[VALIDATION_FITNESS] is None
    assert result[VALID_SOLUTION] is False
    assert result[AUX_EVAL_INFO]["status"] == "scoring_failed"
    assert "prediction_oversized" in result[AUX_EVAL_INFO]["feedback"]
    assert result[EXECUTION_OUTPUT].exit_code == 1


def test_failed_status_maps_to_failure():
    task = _task(SandboxResult("job_3", "failed", None, {"status": "failed", "result": {"result": "failed", "error": "exit 1"}}))
    _, result = task.step_task({}, "code")
    assert result[AUX_EVAL_INFO]["status"] == "failed"
    assert result[VALIDATION_FITNESS] is None


def test_cancelled_status_maps_to_cancelled_failure():
    task = _task(SandboxResult("job_4", "cancelled", None, {"status": "cancelled", "result": {}}))
    _, result = task.step_task({}, "code")
    assert result[AUX_EVAL_INFO]["status"] == "cancelled"
    assert result[VALIDATION_FITNESS] is None


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), True, 1.5, -0.2])
def test_untrusted_or_missing_score_is_a_failure(score):
    task = _task(_completed(score))
    _, result = task.step_task({}, CODE)
    assert result[VALIDATION_FITNESS] is None
    assert result[VALID_SOLUTION] is False
    assert result[AUX_EVAL_INFO]["status"] in {"unknown", "failed"}
    assert result[AUX_EVAL_INFO]["status_code"] >= 400


def test_string_result_body_with_valid_evidence_is_parsed():
    import json

    outcome = _completed(0.5)
    raw = copy.deepcopy(outcome.raw)
    raw["result"] = json.dumps(raw["result"])
    task = _task(SandboxResult("job_5", "completed", 0.5, raw))
    _, result = task.step_task({}, CODE)
    assert result[VALIDATION_FITNESS] == 0.5


def test_admission_denial_is_invalid_candidate_not_crash():
    task = _task(ValueError("forbidden_marker:../"))
    _, result = task.step_task({}, "open('../secret')")
    assert result[AUX_EVAL_INFO]["status"] == "invalid"
    assert result[VALIDATION_FITNESS] is None
    assert task.jobs[0]["status"] == "admission_denied"
    assert task.jobs[0]["job_id"] is None  # never submitted


@pytest.mark.parametrize(
    "reason",
    ["policy_unavailable:missing", "sandbox_endpoint_or_api_key_unavailable", "invalid_job_id"],
)
def test_infrastructure_failures_propagate(reason):
    task = _task(ValueError(reason))
    with pytest.raises(ValueError, match=reason):
        task.step_task({}, "code")


def test_wait_timeout_maps_to_timeout_failure_with_cancel_evidence():
    task = _task(TimeoutError("sandbox job job_6 did not finish; cancel_evidence={'cleanup_verified': True}"))
    _, result = task.step_task({}, "code")
    assert result[AUX_EVAL_INFO]["status"] == "timeout"
    assert result[EXECUTION_OUTPUT].timed_out is True
    assert task.jobs[0]["status"] == "timeout"


def test_empty_code_never_submitted():
    task = _task(_completed())
    _, result = task.step_task({}, "   ")
    assert result[AUX_EVAL_INFO]["status"] == "invalid"
    assert task._client.calls == []


def test_cleanup_unproven_budget_error_propagates():
    task = _task(BudgetError("sandbox_cleanup_unproven"))
    with pytest.raises(BudgetError, match="sandbox_cleanup_unproven"):
        task.step_task({}, "code")


def test_stopped_token_aborts_before_submission():
    token = CancellationToken()
    token.stop("operator_halt")
    task = SandboxAcceptanceTask(FakeClient(_completed()), token)
    with pytest.raises(Cancelled):
        task.step_task({}, "code")
    assert task._client.calls == []


def test_cancel_all_live_delegates():
    task = _task(_completed())
    evidence = task.cancel_all_live()
    assert task._client.cancelled is True
    assert evidence[0]["cleanup_verified"] is True


def test_stop_requested_flag():
    task = _task(_completed())
    assert task.stop_requested is False
    task.stop_requested = True
    assert task.stop_requested is True
