"""Authenticated OpenMLE Sandbox client delegating to the versioned source gate.

Origin: legacy rsi-gpu tree, sandbox_eval_client.py (read-only; exact source
path and hash recorded in research/provenance/manifest.json).
Consolidated under research/adapters for US-001; US-004 repair replaced the
local legacy marker list with the SAME versioned policy engine used by the
sandbox API admission and the Evo clients (research.contracts.source_gate).
Credentials are read from the environment only (SANDBOX_ENDPOINT / SANDBOX_API_KEY).

US-007 repair: remote-job cancellation. The legacy client raised TimeoutError
on a wait timeout WITHOUT cancelling the remote job, leaving sandbox work
running (and consuming GPU) after the caller gave up. SandboxEvalClient now
tracks live remote job ids in a thread-safe registry (register on submit,
unregister on terminal status) and cancels via the server-side endpoint
DELETE /api/v1/jobs/{job_id}: on wait timeout (and on poll aborts while the
job may still be running) the job is cancelled FIRST and the cancel evidence
is attached to the raised TimeoutError. cancel_job/cancel_all_live never
raise; failures are captured into the evidence dict.

NOTE: the source gate is a pre-execution keyword/policy filter, not a
security boundary. Real isolation is enforced by the sandbox worker, not by
this filter. A missing/unreadable policy fails closed (deny).
"""
from __future__ import annotations

import os
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from research.contracts.budget import BudgetError, CancellationToken
from research.contracts.sandbox_lifecycle import cleanup_confirmed

from research.contracts.source_gate import (
    SourcePolicyError,
    check_source_bytes,
    load_policy,
    validate_job_fields,
)

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


@dataclass(frozen=True)
class SandboxResult:
    job_id: str
    status: str
    score: float | None
    raw: dict


# Legacy compatibility alias: the marker list now comes from the versioned
# policy artifact (a superset of the original hard-coded tuple). The tuple
# below is only a fallback for documentation/testing when the bundled policy
# cannot be loaded; the gate itself always loads the policy and fails closed.
_LEGACY_MARKERS = (
    "curl ", "wget ", "requests.get", "urllib.request", "socket.",
    "subprocess", "os.system", "os.popen", "shutil.rmtree", "/etc/",
    "..\\", "../", "PRIVATE_KEY", "OPENAI_API_KEY", "PRIMARY_KEY",
)
try:
    FORBIDDEN_MARKERS: tuple[str, ...] = tuple(load_policy().denied_source_markers)
except SourcePolicyError:  # pragma: no cover - policy bundled in-repo
    FORBIDDEN_MARKERS = _LEGACY_MARKERS


def safety_gate_source(source: str) -> tuple[bool, str]:
    """Gate source bytes with the versioned policy; fail closed on errors."""
    try:
        policy = load_policy()
    except SourcePolicyError as exc:
        return False, f"policy_unavailable:{exc}"
    verdict = check_source_bytes(
        source.encode("utf-8", errors="surrogatepass"), policy
    )
    if verdict.allowed:
        return True, "ok"
    return False, "; ".join(f"{d.rule}:{d.reason}" for d in verdict.denials)


class SandboxEvalClient:
    """Stateful sandbox client with a live-job registry and cancellation.

    The registry tracks remote job ids that may still be running server-side
    so a client-side give-up (wait timeout, poll error) always pairs with a
    best-effort remote cancel. All cancel paths are fail-safe: they capture
    errors into an evidence dict instead of raising.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        *, require_cleanup: bool = False,
        observer: Any = None,
    ):
        self._endpoint = endpoint
        self._api_key = api_key
        self._resolved_endpoint: str | None = None
        self._resolved_api_key: str | None = None
        self._live_jobs: set[str] = set()
        self._uncertain_submissions: set[str] = set()
        self.require_cleanup = require_cleanup
        # Optional observer (acceptance runner): receives machine-readable
        # registry events so live submitted job ids and unresolved submission
        # intents are persisted BEFORE polling — a killed process must never
        # take the cleanup evidence with it.
        self._observer = observer
        self._live_lock = threading.Lock()

    def _emit(self, event: dict) -> None:
        if self._observer is not None:
            self._observer(event)

    # ------------------------------------------------------------ registry
    def _register(self, job_id: str, trace_id: str | None = None) -> None:
        with self._live_lock:
            self._live_jobs.add(job_id)
        self._emit({"event": "job_registered", "job_id": job_id, "trace_id": trace_id})

    def _unregister(self, job_id: str) -> None:
        with self._live_lock:
            self._live_jobs.discard(job_id)
        self._emit({"event": "job_unregistered", "job_id": job_id})

    def live_job_ids(self) -> frozenset[str]:
        """Snapshot of the currently registered (possibly live) job ids."""
        with self._live_lock:
            return frozenset(self._live_jobs)

    def uncertain_submission_ids(self) -> frozenset[str]:
        """Submission trace ids whose server-side outcome is unknown (never
        claim these cleaned without explicit proof)."""
        with self._live_lock:
            return frozenset(self._uncertain_submissions)

    # ---------------------------------------------------------------- auth
    def _resolve_auth(
        self,
        endpoint: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        resolved_endpoint = (
            endpoint
            or self._endpoint
            or self._resolved_endpoint
            or os.environ.get("SANDBOX_ENDPOINT")
        )
        resolved_api_key = (
            api_key
            or self._api_key
            or self._resolved_api_key
            or os.environ.get("SANDBOX_API_KEY")
        )
        return resolved_endpoint, resolved_api_key

    # ------------------------------------------------------------- submit
    def submit_and_wait(
        self,
        source: str,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        name: str = "rsi-de-hello-synth",
        data_dir: str = "/mnt/pubdatasets2/tasks/hello_synth",
        resource_type: str = "gpu",
        gpu_count: int = 1,
        timeout: int = 300,
        poll_interval: float = 2.0,
        cancellation_token: CancellationToken | None = None,
        task_id: str | None = None,
    ) -> SandboxResult:
        ok, reason = safety_gate_source(source)
        if not ok:
            raise ValueError(reason)
        try:
            policy = load_policy()
        except SourcePolicyError as exc:  # fail closed
            raise ValueError(f"policy_unavailable:{exc}") from exc
        environment = {"EXECUTION_MODE": "shell", "DATA_DIR": data_dir}
        field_denials = validate_job_fields(
            policy, data_dir=data_dir, environment=environment
        )
        if field_denials:
            raise ValueError(
                "; ".join(f"{d.rule}:{d.reason}" for d in field_denials)
            )
        resolved_endpoint, resolved_api_key = self._resolve_auth(endpoint, api_key)
        if resolved_endpoint is None or resolved_api_key is None:
            # Preserve the legacy fail-fast behavior (missing env raised
            # KeyError) as an explicit configuration error.
            raise ValueError("sandbox_endpoint_or_api_key_unavailable")
        self._resolved_endpoint = resolved_endpoint
        self._resolved_api_key = resolved_api_key
        trace_id = uuid.uuid4().hex
        headers = {"X-API-Key": resolved_api_key, "X-Trace-ID": trace_id}
        payload = {
            "name": name,
            "code": source,
            "data_dir": data_dir,
            "timeout": timeout,
            "resource_type": resource_type,
            "gpu_count": gpu_count,
            "priority": 1,
            "idempotency_key": trace_id,
            "environment": environment,
        }
        if task_id is not None:
            payload["task_id"] = task_id
        if cancellation_token is not None:
            cancellation_token.check()
        # Persist the submission intent BEFORE the POST: a process killed
        # mid-request leaves an unknown-identity marker, never a silent gap.
        self._emit({"event": "submission_intent", "trace_id": trace_id})
        with httpx.Client(base_url=resolved_endpoint, timeout=30.0, trust_env=False) as client:
            if cancellation_token is not None and cancellation_token.deadline_epoch is not None:
                client.timeout = httpx.Timeout(max(0.01, min(30.0, cancellation_token.deadline_epoch-time.time())))
            try:
                response = client.post("/api/v1/jobs", json=payload, headers=headers)
            except httpx.TransportError:
                with self._live_lock:
                    self._uncertain_submissions.add(trace_id)
                self._emit({"event": "submission_uncertain", "trace_id": trace_id})
                raise
            if response.status_code >= 500:
                with self._live_lock:
                    self._uncertain_submissions.add(trace_id)
                self._emit({"event": "submission_uncertain", "trace_id": trace_id})
            response.raise_for_status()
            try:
                job_id = response.json()["job_id"]
                if not isinstance(job_id, str) or not job_id:
                    raise ValueError("invalid_job_id")
            except (ValueError, KeyError, TypeError):
                with self._live_lock:
                    self._uncertain_submissions.add(trace_id)
                self._emit({"event": "submission_uncertain", "trace_id": trace_id})
                raise
            try:
                self._register(job_id, trace_id)
            except Exception:
                # Durable-registry write failed AFTER the server created the
                # job: the identity is known, so cancel it explicitly before
                # propagating (never leave an untracked live job).
                self.cancel_job(job_id)
                raise
            try:
                try:
                    deadline = time.monotonic() + timeout + 60
                    while time.monotonic() < deadline:
                        remaining = deadline - time.monotonic()
                        if cancellation_token is not None:
                            cancellation_token.check()
                            if cancellation_token.deadline_epoch is not None:
                                remaining = min(remaining, cancellation_token.deadline_epoch - time.time())
                        client.timeout = httpx.Timeout(max(0.01, min(30.0, remaining)))
                        current = client.get(
                            f"/api/v1/jobs/{job_id}", headers=headers
                        )
                        current.raise_for_status()
                        body = current.json()
                        status = body.get("status")
                        if status in TERMINAL_STATUSES and (status != "cancelled" or cleanup_confirmed(body)):
                            if self.require_cleanup and not cleanup_confirmed(body):
                                raise BudgetError("sandbox_cleanup_unproven")
                            self._unregister(job_id)
                            result = body.get("result") or {}
                            if isinstance(result, str):
                                result = json.loads(result)
                            score = result.get("score")
                            if score is None:
                                score = result.get("submit_score")
                            return SandboxResult(job_id, status, score, body)
                        delay = min(poll_interval, max(0.0, deadline-time.monotonic()))
                        if cancellation_token is not None and cancellation_token.deadline_epoch is not None:
                            delay = min(delay, max(0.0, cancellation_token.deadline_epoch-time.time()))
                        time.sleep(delay)
                except BaseException:
                    # The poll aborted while the remote job may still be
                    # running: cancel it (best-effort) before propagating.
                    self.cancel_job(job_id)
                    raise
                cancel_evidence = self.cancel_job(job_id)
                raise TimeoutError(
                    f"sandbox job {job_id} did not finish; "
                    f"cancel_evidence={cancel_evidence!r}"
                )
            finally:
                # Unknown cleanup stays registered for run-level retry/checkpoint.
                pass

    # ------------------------------------------------------------- cancel
    def cancel_job(self, job_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
        """Cancel and await explicit cleanup; retain unconfirmed live jobs."""
        evidence: dict[str, Any] = {
            "job_id": job_id, "cancel_http_status": None, "cancel_error": None,
            "terminal_status": None, "cleanup_verified": False,
        }
        try:
            endpoint, api_key = self._resolve_auth(None, None)
            if endpoint is None or api_key is None:
                evidence["cancel_error"] = "sandbox_endpoint_or_api_key_unavailable"
                return evidence
            headers = {"X-API-Key": api_key, "X-Trace-ID": uuid.uuid4().hex}
            deadline = time.monotonic() + timeout
            with httpx.Client(base_url=endpoint, timeout=min(30.0, max(0.01, timeout)), trust_env=False) as client:
                response = client.delete(f"/api/v1/jobs/{job_id}", headers=headers)
                evidence["cancel_http_status"] = response.status_code
                if response.status_code >= 400 and response.status_code != 404:
                    evidence["cancel_error"] = "cancel_http_error"
                    return evidence
                while time.monotonic() < deadline:
                    client.timeout = httpx.Timeout(max(0.01, min(30.0, deadline-time.monotonic())))
                    current = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
                    current.raise_for_status()
                    body = current.json()
                    if body.get("status") in TERMINAL_STATUSES:
                        evidence["terminal_status"] = body["status"]
                    if cleanup_confirmed(body):
                        evidence["cleanup_verified"] = True
                        self._unregister(job_id)
                        return evidence
                    time.sleep(min(1.0, max(0.0, deadline-time.monotonic())))
                evidence["cancel_error"] = "cleanup_unconfirmed"
        except Exception as exc:
            evidence["cancel_error"] = type(exc).__name__
        return evidence

    def cancel_all_live(self) -> list[dict[str, Any]]:
        """Cancel every currently registered live job id (best-effort)."""
        with self._live_lock:
            job_ids = sorted(self._live_jobs)
        results = [self.cancel_job(job_id) for job_id in job_ids]
        with self._live_lock:
            unknown = sorted(self._uncertain_submissions)
        results.extend({"job_id": None, "trace_id": trace_id,
                        "cleanup_verified": False, "cancel_error": "submission_identity_unknown"}
                       for trace_id in unknown)
        return results


_DEFAULT_CLIENT = SandboxEvalClient()


def submit_and_wait(
    source: str,
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    name: str = "rsi-de-hello-synth",
    data_dir: str = "/mnt/pubdatasets2/tasks/hello_synth",
    resource_type: str = "gpu",
    gpu_count: int = 1,
    timeout: int = 300,
    poll_interval: float = 2.0,
) -> SandboxResult:
    """Module-level convenience wrapper over the shared default client."""
    return _DEFAULT_CLIENT.submit_and_wait(
        source,
        endpoint=endpoint,
        api_key=api_key,
        name=name,
        data_dir=data_dir,
        resource_type=resource_type,
        gpu_count=gpu_count,
        timeout=timeout,
        poll_interval=poll_interval,
    )


def cancel_job(job_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """Cancel one remote job via the shared default client (never raises)."""
    return _DEFAULT_CLIENT.cancel_job(job_id, timeout=timeout)


def cancel_all_live() -> list[dict[str, Any]]:
    """Cancel all jobs registered on the shared default client."""
    return _DEFAULT_CLIENT.cancel_all_live()
