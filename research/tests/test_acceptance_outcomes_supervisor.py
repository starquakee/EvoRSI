"""Regressions for the controller-shaped execution failure seen in US009."""
from __future__ import annotations

import hashlib
from typing import Any

from research.adapters.sandbox_eval_client import SandboxResult
from research.contracts.budget import CancellationToken
from research.contracts.source_gate import load_policy
from research.search.sandbox_task import (
    AUX_EVAL_INFO, EXECUTION_OUTPUT, VALIDATION_FITNESS, VALID_SOLUTION,
    SandboxAcceptanceTask,
)

SOURCE = "print('synthetic fixture')\n"
ERROR = "ValueError: not enough values to unpack (expected 3, got 2)"

class FailedClient:
    def submit_and_wait(self, source: str, **kwargs: Any) -> SandboxResult:
        assert source == SOURCE
        policy = load_policy()
        file_hash = hashlib.sha256(source.encode()).hexdigest()
        tree_hash = hashlib.sha256(b"main.py\0" + file_hash.encode() + b"\n").hexdigest()
        return SandboxResult("job_failed_fixture", "failed", None, {
            "job_id": "job_failed_fixture", "status": "failed",
            "completed_at": "2026-10-09T01:00:00Z",
            "result": {
                "result": "code_execution_error", "score": None,
                "source_identity_verified": True, "worker_cleanup_verified": True,
                "source_gate": {
                    "allowed": True, "policy_version": policy.policy_version,
                    "policy_sha256": policy.policy_sha256,
                    "entrypoint": "main.py", "entrypoint_sha256": file_hash,
                    "source_sha256": tree_hash,
                },
                "run_log": "training progress\n" * 1000 + "Traceback (most recent call last):\n" + ERROR + "\n",
            },
        })

def run_failure() -> tuple[SandboxAcceptanceTask, dict[str, Any]]:
    task = SandboxAcceptanceTask(FailedClient(), CancellationToken())  # type: ignore[arg-type]
    _, result = task.step_task({}, SOURCE)
    return task, result

def test_execution_failure_keeps_job_identity_and_actionable_tail_feedback() -> None:
    _, result = run_failure()
    info = result[AUX_EVAL_INFO]
    assert info.get("job_id") == "job_failed_fixture"
    feedback = info["feedback"]
    assert "code_execution_error" in feedback
    assert ERROR in feedback
    assert len(feedback) <= 4096
    assert ERROR in "".join(result[EXECUTION_OUTPUT].term_out)

def test_execution_proof_is_not_misreported_as_scoring_proof() -> None:
    task, result = run_failure()
    assert result[VALID_SOLUTION] is False
    assert result[VALIDATION_FITNESS] is None
    evidence = task.jobs[0]
    assert evidence["job_id"] == "job_failed_fixture"
    assert evidence["source_identity_verified"] is True
    assert evidence["worker_cleanup_verified"] is True
    assert evidence["evidence_verified"] is False
    assert evidence["score"] is None
    assert evidence["scoring_result"] == "code_execution_error"
    assert evidence["evaluation"] == {}
