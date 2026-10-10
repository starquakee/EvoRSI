"""Production dojo task adapter over the trusted sandbox client (US-009).

This is the REAL ``step_task`` for the acceptance inner Evo run: model
generated code is submitted through ``SandboxEvalClient(require_cleanup=True)``
to the isolated 6581 stack, and ONLY validated canonical evaluator evidence
is translated into the eval_result the solver parses. A "completed" job is
scored ONLY when its persisted evidence verifies end to end: source identity
and cleanup verified, the source gate evidence binds the exact submitted
bytes under the CURRENT policy, and the evaluation block matches the pinned
registry (task/metric/direction/split, answer hash, row count, registry
version, external evaluator marker) with the claimed score. Every unproven,
mismatched, unscored, failed, nonfinite, cancelled or admission-denied
outcome maps to a deterministic failure (never a model-judged score).

The module stays dojo-free so it imports in all three environments; the
execution-result object is a small duck-typed dataclass (``term_out`` /
``exec_time`` / ``exit_code`` / ``timed_out``) compatible with
``Node.absorb_exec_result``. The inner runner may inject dojo's real
``ExecutionResult`` class via ``execution_result_cls``.
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.adapters.sandbox_eval_client import SandboxEvalClient, SandboxResult
from research.contracts.budget import CancellationToken
from research.contracts.results import hash_source
from research.contracts.source_gate import load_policy
from research.evaluator.service import EvaluatorRegistry
from research.search.acceptance_config import (
    JOB_TIMEOUT_SECONDS,
    RESOURCE_TYPE,
    SANDBOX_DATA_DIR,
    TASK_ID,
)

# eval_result key strings (dojo.core.tasks.constants values; duplicated so
# this module never imports dojo).
EXECUTION_OUTPUT = "execution_output"
VALIDATION_FITNESS = "validation_fitness"
AUX_EVAL_INFO = "aux_eval_info"
VALID_SOLUTION = "valid_solution"
VALID_SOLUTION_FEEDBACK = "valid_solution_feedback"

#: Aux statuses parse_eval_result treats as deterministic failures.
STATUS_COMPLETED = "completed"
STATUS_SCORING_FAILED = "scoring_failed"
STATUS_TIMEOUT = "timeout"
STATUS_CANCELLED = "cancelled"
STATUS_INVALID = "invalid"
STATUS_FAILED = "failed"
STATUS_UNKNOWN = "unknown"

_FEEDBACK_LIMIT = 2000


@dataclass
class SandboxExecutionResult:
    """Duck-compatible with dojo ExecutionResult (absorb_exec_result fields)."""

    term_out: list[str]
    exec_time: float
    exit_code: int | None = None
    timed_out: bool = False


def _truncate(text: Any) -> str:
    value = str(text)
    return value if len(value) <= _FEEDBACK_LIMIT else value[:_FEEDBACK_LIMIT] + "..."


def _summarize_run_log(log: str | None, limit: int = 300) -> str:
    """One bounded, meaningful line from the canonical run_log (the actual
    execution error, not a fabricated reason)."""
    if not log:
        return ""
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    if not lines:
        return ""
    import re

    summary = lines[-1]
    for line in reversed(lines):
        if re.search(r"(?i)error|exception|traceback", line):
            summary = line
            break
    return summary[:limit]


def _submitted_bytes(source: str) -> bytes:
    """Exactly the encoding the API gates/writes (utf-8, surrogatepass)."""
    return source.encode("utf-8", errors="surrogatepass")


def _single_file_tree_sha256(entrypoint: str, data: bytes) -> str:
    """Recompute the API's whole-artifact tree hash for a one-file artifact.

    Mirrors research.contracts.source_snapshot.capture_source: sorted
    (relative, sha256(data)) entries as ``name\\0hexdigest\\n``. The API
    records this tree hash as ``source_sha256``; recomputing it binds the
    evidence to the exact submitted bytes, not a digest-shaped string.
    """
    import hashlib

    tree = hashlib.sha256()
    tree.update(entrypoint.encode("utf-8") + b"\0")
    tree.update(hashlib.sha256(data).hexdigest().encode("ascii") + b"\n")
    return tree.hexdigest()


def _answer_row_count(answer_path: Path) -> int:
    """Trusted controller-side count of answer rows (for evidence binding)."""
    with open(answer_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return max(0, len(rows) - 1)  # minus header


class SandboxAcceptanceTask:
    """Synchronous task interface for Evolutionary.search over the sandbox.

    Construction loads the CURRENT source policy and evaluator registry
    (fail closed if either is unavailable): completion evidence is verified
    against these pins, never against candidate-supplied claims.

    ``jobs`` accumulates redacted per-job evidence (ids, statuses, scores,
    evaluator evidence) for the acceptance record — never source code or
    credentials.
    """

    def __init__(
        self,
        client: SandboxEvalClient,
        token: CancellationToken,
        *,
        data_dir: str = SANDBOX_DATA_DIR,
        task_id: str = TASK_ID,
        resource_type: str = RESOURCE_TYPE,
        job_timeout_seconds: int = JOB_TIMEOUT_SECONDS,
        name_prefix: str = "us009-acceptance",
        execution_result_cls: type | None = None,
    ):
        self._client = client
        self._token = token
        self._data_dir = data_dir
        self._task_id = task_id
        self._resource_type = resource_type
        self._job_timeout = job_timeout_seconds
        self._name_prefix = name_prefix
        self._result_cls = execution_result_cls or SandboxExecutionResult
        self._policy = load_policy()  # fail closed
        self._registry = EvaluatorRegistry.load()  # fail closed
        self._spec = self._registry.task(task_id)  # fail closed
        self._expected_rows = _answer_row_count(self._spec.answer_path)
        self.stop_requested = False
        self.jobs: list[dict[str, Any]] = []

    # ------------------------------------------------------------ helpers
    def _exec_result(self, feedback: str, exec_time: float, exit_code: int | None,
                     timed_out: bool) -> Any:
        return self._result_cls(
            term_out=[_truncate(feedback)],
            exec_time=exec_time,
            exit_code=exit_code,
            timed_out=timed_out,
        )

    def _failure(self, aux_status: str, feedback: str, exec_time: float,
                 *, timed_out: bool = False,
                 job_id: str | None = None) -> dict[str, Any]:
        aux: dict[str, Any] = {
            "status": aux_status,
            "status_code": 500,
            "feedback": _truncate(feedback),
        }
        if job_id:
            aux["job_id"] = job_id
        return {
            EXECUTION_OUTPUT: self._exec_result(feedback, exec_time, 1, timed_out),
            VALIDATION_FITNESS: None,
            AUX_EVAL_INFO: aux,
            VALID_SOLUTION: False,
            VALID_SOLUTION_FEEDBACK: _truncate(feedback),
        }

    def _verify_completion_evidence(
        self,
        raw: dict[str, Any],
        body: dict[str, Any],
        evaluation: Any,
        source: str,
        claimed_score: float,
    ) -> str | None:
        """Return None when the completion evidence verifies, else the
        machine-readable mismatch reason (the job is then a failure)."""
        if body.get("result") == "scoring_failed":
            return "scoring_failed"
        if body.get("source_identity_verified") is not True:
            return "source_identity_unverified"
        if body.get("worker_cleanup_verified") is not True:
            return "worker_cleanup_unverified"
        completed_at = raw.get("completed_at")
        if not isinstance(completed_at, str) or not completed_at:
            return "completion_unproven"
        gate = body.get("source_gate")
        if not isinstance(gate, dict):
            return "source_gate_evidence_missing"
        if gate.get("allowed") is not True:
            return "source_gate_not_allowed"
        submitted = _submitted_bytes(source)
        entrypoint = gate.get("entrypoint")
        if not isinstance(entrypoint, str) or not entrypoint:
            return "entrypoint_missing"
        if gate.get("entrypoint_sha256") != hash_source(submitted):
            return "entrypoint_hash_mismatch"
        if gate.get("source_sha256") != _single_file_tree_sha256(entrypoint, submitted):
            return "source_hash_mismatch"
        if gate.get("policy_version") != self._policy.policy_version:
            return "policy_version_mismatch"
        if gate.get("policy_sha256") != self._policy.policy_sha256:
            return "policy_hash_mismatch"
        if not isinstance(evaluation, dict) or not evaluation:
            return "evaluation_missing"
        if evaluation.get("evaluator") != "external_trusted":
            return "evaluator_untrusted"
        if evaluation.get("task_id") != self._spec.task_id:
            return "evaluation_task_mismatch"
        if evaluation.get("metric") != self._spec.metric:
            return "evaluation_metric_mismatch"
        if evaluation.get("direction") != self._spec.direction.value:
            return "evaluation_direction_mismatch"
        if evaluation.get("split") != self._spec.split:
            return "evaluation_split_mismatch"
        eval_score = evaluation.get("score")
        if (
            isinstance(eval_score, bool)
            or not isinstance(eval_score, (int, float))
            or float(eval_score) != claimed_score
        ):
            return "evaluation_score_mismatch"
        prediction_sha256 = evaluation.get("prediction_sha256")
        if not isinstance(prediction_sha256, str) or not prediction_sha256:
            return "prediction_hash_missing"
        if evaluation.get("answer_sha256") != self._spec.answer_sha256:
            return "answer_hash_mismatch"
        if evaluation.get("row_count") != self._expected_rows:
            return "row_count_mismatch"
        if evaluation.get("registry_version") != self._registry.registry_version:
            return "registry_version_mismatch"
        return None

    # ------------------------------------------------------------ dojo API
    def step_task(self, state: dict, code: Any,
                  generation_feedback: str | None = None) -> tuple[dict, dict[str, Any]]:
        """Submit candidate code to the isolated sandbox and translate ONLY
        validated canonical evaluator evidence into an eval_result.

        ``generation_feedback`` (acceptance runner) carries the precise
        generation-outcome classification for empty extractions, so debug
        receives the actual cause (truncation, unclosed fence, invalid
        Python) instead of an unexplained ``empty_candidate_code``."""
        self._token.check()
        started = time.monotonic()
        if not isinstance(code, str) or not code.strip():
            reason = (
                f"empty_candidate_code:{generation_feedback}"
                if generation_feedback
                else "empty_candidate_code"
            )
            return state, self._failure(STATUS_INVALID, reason, 0.0)
        try:
            result = self._client.submit_and_wait(
                code,
                name=f"{self._name_prefix}-{len(self.jobs)}",
                data_dir=self._data_dir,
                resource_type=self._resource_type,
                timeout=self._job_timeout,
                cancellation_token=self._token,
                task_id=self._task_id,
            )
        except ValueError as exc:
            # Pre-execution source-gate / job-field denial: the candidate is
            # invalid. Infrastructure/configuration failures fail the RUN
            # closed instead of being misreported as candidate outcomes.
            reason = str(exc)
            for prefix in (
                "policy_unavailable",
                "sandbox_endpoint_or_api_key_unavailable",
                "invalid_job_id",
            ):
                if reason.startswith(prefix):
                    raise
            elapsed = time.monotonic() - started
            self.jobs.append({
                "job_id": None, "status": "admission_denied",
                "score": None, "reason": _truncate(reason),
            })
            return state, self._failure(STATUS_INVALID, f"admission_denied:{reason}", elapsed)
        except TimeoutError as exc:
            # The client already cancelled the remote job (evidence attached).
            elapsed = time.monotonic() - started
            self.jobs.append({
                "job_id": None, "status": STATUS_TIMEOUT,
                "score": None, "reason": _truncate(exc),
            })
            return state, self._failure(STATUS_TIMEOUT, str(exc), elapsed, timed_out=True)
        return state, self._translate(result, code, time.monotonic() - started)

    def _translate(self, result: SandboxResult, source: str,
                   exec_time: float) -> dict[str, Any]:
        raw: dict[str, Any] = result.raw if isinstance(result.raw, dict) else {}
        body = raw.get("result") or {}
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except ValueError:
                body = {}
        if not isinstance(body, dict):
            body = {}
        evaluation = body.get("evaluation")
        if not isinstance(evaluation, dict):
            evaluation = {}
        score = result.score
        score_value: float | None = None
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
            and 0.0 <= float(score) <= 1.0
        ):
            score_value = float(score)
        mismatch: str | None = None
        if result.status == "completed" and score_value is not None:
            mismatch = self._verify_completion_evidence(
                raw, body, evaluation, source, score_value
            )
        scored = mismatch is None and result.status == "completed" and score_value is not None
        scoring_result = body.get("result")
        execution_error = ""
        if not scored and result.status == "failed":
            # Retain the ACTUAL bounded execution error from the canonical
            # run_log — inline when the controller persisted it in the result
            # body, otherwise fetched (bounded + credential-redacted) from
            # the canonical logs endpoint. Never a fabricated reason.
            log_text = body.get("run_log")
            if not isinstance(log_text, str) or not log_text:
                fetch_log = getattr(self._client, "fetch_job_log", None)
                log_text = fetch_log(result.job_id) if callable(fetch_log) else None
            execution_error = _summarize_run_log(log_text)
        evidence = {
            "job_id": result.job_id,
            "status": result.status,
            "score": score_value if scored else None,
            "scoring_result": scoring_result,
            "reason": scoring_result,
            "evidence_verified": scored,
            "evidence_mismatch": mismatch,
            "source_identity_verified": body.get("source_identity_verified") is True,
            "worker_cleanup_verified": body.get("worker_cleanup_verified") is True,
            "completed_at_proven": bool(
                isinstance(raw.get("completed_at"), str) and raw.get("completed_at")
            ),
            "execution_error": execution_error,
            "evaluation": {
                key: evaluation.get(key)
                for key in (
                    "task_id", "split", "metric", "direction", "score",
                    "row_count", "prediction_sha256", "answer_sha256",
                    "registry_version", "reason", "prediction_snapshot",
                )
                if key in evaluation
            },
        }
        self.jobs.append(evidence)
        if scored:
            assert score_value is not None
            return {
                EXECUTION_OUTPUT: self._exec_result(
                    f"accuracy={score_value}", exec_time, 0, False
                ),
                VALIDATION_FITNESS: score_value,
                AUX_EVAL_INFO: {
                    "status": STATUS_COMPLETED,
                    "status_code": 200,
                    "score": score_value,
                    "job_id": result.job_id,
                    "evaluation": evidence["evaluation"],
                    "feedback": f"trusted evaluator accuracy={score_value}",
                },
                VALID_SOLUTION: True,
                VALID_SOLUTION_FEEDBACK: f"accuracy={score_value}",
            }
        if mismatch is not None:
            aux_status = STATUS_UNKNOWN
            feedback = f"evidence_unverified:{mismatch}"
        elif body.get("result") == "scoring_failed":
            aux_status = STATUS_SCORING_FAILED
            feedback = f"scoring_failed: {evaluation.get('reason') or body.get('error') or ''}"
        elif result.status == "cancelled":
            aux_status = STATUS_CANCELLED
            feedback = "cancelled"
        elif result.status == "timeout":
            aux_status = STATUS_TIMEOUT
            feedback = "timeout"
        elif result.status == "failed":
            aux_status = STATUS_FAILED
            detail = execution_error or body.get("error") or ""
            feedback = f"failed:{scoring_result or 'error'}: {detail}"
        else:
            aux_status = STATUS_UNKNOWN
            feedback = f"untrusted outcome: {result.status}"
        return self._failure(aux_status, feedback, exec_time,
                             timed_out=aux_status == STATUS_TIMEOUT,
                             job_id=result.job_id)

    def cancel_all_live(self) -> list[dict[str, Any]]:
        """Best-effort cancel of every registered live job (never raises)."""
        return self._client.cancel_all_live()
