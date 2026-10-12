"""Comparison-specific trusted sandbox task adapter (EXP-005).

Extends the US-009 SandboxAcceptanceTask with comparison semantics:

- ORDINARY candidate failures (execution error, timeout, cancellation,
  admission denial, trusted scoring rejection of an invalid submission)
  remain honest raw failures — the candidate is invalid, the search
  continues, nothing is stopped.
- TRUSTED-EVIDENCE integrity failures on a claimed-completed job (source
  identity unverified, worker cleanup unverified, source-gate hash/policy
  mismatch, evaluator task/metric/direction/split/score/hash/row/registry
  mismatch, unproven completion) are infrastructure-level contradictions:
  the adapter durably stops the comparison ledger AND the shared
  cancellation token AT DETECTION. Evo's debug cycle swallows task
  exceptions, so the stop cannot rely on propagation — the next
  before_operator / guard boundary fails closed before any further
  provider request.
- Job evidence persists exact source-gate identities (entrypoint and
  executed-source hashes, policy identity) and the submitted-source hash,
  not only boolean verified flags.
"""
from __future__ import annotations

import hashlib
from typing import Any

from research.comparison.budget import ComparisonBudgetError, ComparisonLedger
from research.contracts.budget import Cancelled
from research.search.sandbox_task import SandboxAcceptanceTask, _submitted_bytes

_GATE_EVIDENCE_KEYS = (
    "allowed",
    "entrypoint",
    "entrypoint_sha256",
    "source_sha256",
    "policy_version",
    "policy_sha256",
)


class ComparisonSandboxTask(SandboxAcceptanceTask):
    """SandboxAcceptanceTask + immediate durable stop on integrity failure."""

    def __init__(self, *args: Any, ledger: ComparisonLedger, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._comparison_ledger = ledger

    def _integrity_stop(self, reason: str) -> None:
        """Durably stop the ledger AND the shared token at detection time.

        Evo's debug cycle swallows task exceptions, so a trusted-evidence
        contradiction cannot rely on propagation to the outer runner.
        """
        self._comparison_ledger.stop(reason)
        self._token.stop(reason)

    def step_task(
        self, state: Any, code: Any, generation_feedback: Any = None
    ) -> tuple[Any, dict[str, Any]]:
        """Infrastructure/cleanup errors escaping the base adapter (policy
        unavailable, endpoint/auth failures, unproven cleanup, ...) are
        integrity failures: stop BEFORE Evo's debug-cycle handler can
        swallow the exception. Ordinary candidate failures never raise
        here (they are returned as failure eval_results)."""
        try:
            return super().step_task(
                state, code, generation_feedback=generation_feedback
            )
        except (Cancelled, ComparisonBudgetError):
            raise  # control-flow stops carry their own reasons upstream
        except Exception as exc:
            self._integrity_stop(f"task_integrity:{type(exc).__name__}")
            raise

    def _translate(
        self, result: Any, source: str, exec_time: float
    ) -> dict[str, Any]:
        outcome = super()._translate(result, source, exec_time)
        job = self.jobs[-1]
        # Exact executed-source identity for every submitted job.
        job["submitted_source_sha256"] = hashlib.sha256(
            _submitted_bytes(source)
        ).hexdigest()
        raw = result.raw if isinstance(result.raw, dict) else {}
        body = raw.get("result") or {}
        if isinstance(body, str):
            body = {}
        if isinstance(body, dict):
            gate = body.get("source_gate")
            if isinstance(gate, dict):
                job["source_gate"] = {
                    key: gate.get(key) for key in _GATE_EVIDENCE_KEYS
                }
        mismatch = job.get("evidence_mismatch")
        if mismatch is None and isinstance(body, dict):
            # Integrity contradictions on NON-completed jobs too: an
            # explicitly unverified source identity or unproven cleanup on
            # an executed (e.g. execution-error) job is a trusted-stack
            # contradiction, not an ordinary candidate failure. Absent keys
            # (never-submitted outcomes) are not.
            if body.get("source_identity_verified") is False:
                mismatch = "source_identity_unverified"
            elif body.get("worker_cleanup_verified") is False:
                mismatch = "worker_cleanup_unverified"
        if mismatch is not None:
            self._integrity_stop(f"evidence_integrity:{mismatch}")
        return outcome
