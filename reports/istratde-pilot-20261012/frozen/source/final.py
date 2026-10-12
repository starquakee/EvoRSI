"""Final phase (EXP-006): barrier-gated publication and one-shot frozen
program evaluation.

After ``Orchestrator.verify_final_barrier`` passes, this module:

1. durably enters the IRREVERSIBLE final phase on the original ledger
   (bound to the verified barrier identity) BEFORE any final input is
   copied — afterwards every model reservation/generation entry refuses
   (budget.ComparisonFinalPhaseActive), with the original clock and
   accounting preserved;
2. re-computes the barrier from disk and requires the caller-supplied
   evidence to be byte-identical (a caller-created dictionary never
   suffices), then publishes the hash-bound final data via
   ``data.publish_final_refit`` into the IGNORED runtime publication root
   (train+validation refit -> final ``train.csv``, final features -> final
   ``test.csv``, final labels -> ``answers/.../final_answer.csv`` plus the
   phase-specific ``registry.final.v1.json`` with split ``final``; the
   tracked search registry and search data stay byte-identical);
3. executes each frozen winning program AT MOST ONCE on the refit/final
   data through a ``FinalExecutor`` — a durable per-cell intent written
   BEFORE submission makes a crashed submit refuse duplicates forever, and
   the sandbox timeout is bounded by the ORIGINAL remaining deadline,
   checked before AND immediately after execution;
4. persists the actual prediction bytes plus the full trusted provenance
   (source/entrypoint/policy, job, task/split/metric/direction, answer
   hash, row count, cleanup state) and independently recomputes accuracy
   from the prediction bytes against the trusted sealed final labels,
   requiring agreement with the evaluator score; on resume every artifact
   (publication files, intent, frozen winner bytes, predictions, pinned
   answer) is re-hashed and the score recomputed;
5. records truthful results: raw accuracy or null, utility = accuracy on
   success else 0. No LLM call exists in this phase (no transport import),
   final errors never reopen search, and no final result enters any model
   prompt or algorithm tell. The final phase shares the original 24h
   ledger deadline; it never starts a new timer.

Offline tests inject a synthetic executor and synthetic roots; the real
``SandboxFinalExecutor`` wiring is exercised by EXP-007 preflight probes
and the supervised EXP-008 run only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from research.comparison.budget import (
    ComparisonBudgetError,
    ComparisonLedger,
    ReadOnlyComparisonLedger,
)
from research.comparison import deploy
from research.comparison.data import (
    FINAL_REGISTRY_FILENAME,
    ComparisonDataError,
    _refit_trainval_bytes,
    publish_final_refit,
)
from research.comparison.orchestrate import (
    BARRIER_SCHEMA,
    OrchestrationError,
    Orchestrator,
    _write_durable,
)
from research.comparison.protocol import ComparisonSpecError, canonical_json
from research.comparison.schedule import JOB_TIMEOUT_SECONDS
from research.contracts.source_gate import load_policy
from research.search.sandbox_task import _single_file_tree_sha256

FINAL_INTENT_SCHEMA = "istratde-comparison-final-intent.v1"
FINAL_RESULT_SCHEMA = "istratde-comparison-final-result.v1"
FINAL_PUBLICATION_RECORD = "publication.json"
FINAL_LIVE_JOBS_SCHEMA = "istratde-comparison-final-live-jobs.v1"
PREDICTION_FILENAME = "prediction.csv"

#: Accuracy recomputation agreement bound (float division noise only).
_SCORE_TOLERANCE = 1e-9


class FinalPhaseError(RuntimeError):
    """A final-phase integrity/publication constraint was violated."""


def final_task_id(task_id: str) -> str:
    """Trusted evaluator task id for the final phase of a comparison task."""
    return f"comparison_{task_id}_final"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_durable_bytes(path: Path, data: bytes) -> None:
    """Durable binary write (exclusive tmp + fsync + replace + dir fsync)."""
    tmp = path.with_name(path.name + ".tmp")
    if tmp.is_symlink():
        raise FinalPhaseError(f"durable_write_tmp_symlink:{path.name}")
    if tmp.exists():
        tmp.unlink()
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class FinalExecution:
    """Outcome of ONE final execution of a frozen program.

    ``status`` "completed" requires a trusted evaluator score, prediction
    bytes/hash, verified cleanup AND the full bound provenance payload in
    ``evidence`` (source-gate entrypoint/executed-source/policy hashes,
    evaluator task/split/metric/direction/answer/registry identity, exact
    job identity and the durable cleanup/submission state) — a dataclass of
    trusted-looking booleans alone is not audit evidence. ``failed`` is an
    honest candidate-level failure (raw null, utility 0) and still carries
    its job/cleanup evidence. Infrastructure, source-identity or cleanup
    contradictions — on FAILED jobs as well as successful ones — must be
    RAISED by the executor instead (the caller stops the ledger).
    """

    status: str
    job_id: str | None
    score: float | None
    prediction_bytes: bytes | None
    prediction_sha256: str | None
    cleanup_verified: bool
    reason: str | None
    evidence: Mapping[str, Any] | None = None


class FinalExecutor(Protocol):
    """Executes one frozen program against the final data (no LLM)."""

    def execute(
        self,
        *,
        task_id: str,
        source_bytes: bytes,
        cell_key: str,
        timeout_seconds: float,
    ) -> FinalExecution: ...


def _verify_execution_evidence(
    cell: Any,
    *,
    status: str,
    job_id: Any,
    prediction_sha256: Any,
    winner_source_sha256: str,
    answer_sha256: str,
    evidence: Any,
    winner_bytes: bytes,
    registry_version: str,
    rows: int | None = None,
) -> None:
    """Full final-execution provenance verification, SHARED by initial
    acceptance (``_translate_execution``) and resume
    (``_validate_final_result``).

    Proves: consistent job identity; the ACTUAL durable cleanup sets (live
    jobs, uncertain submissions, submission intents) all empty — a claimed
    boolean alone never proves cleanup; no explicit source-identity/cleanup
    contradiction on successful OR failed jobs (absent keys on
    never-submitted failures are not contradictions); and, for completed
    executions, the exact source entrypoint/executed-source hashes, the
    whole-artifact tree hash recomputed from the pinned ``winner_bytes``,
    the pinned source policy and registry version, trusted evaluator
    task/split/metric/direction/answer identity, row count and canonical
    prediction-snapshot binding. ``winner_bytes`` and ``registry_version``
    are REQUIRED — pinning is never silently skipped. Raises
    FinalPhaseError on any contradiction.
    """
    if not isinstance(evidence, Mapping):
        raise FinalPhaseError("final_evidence_invalid")
    if evidence.get("job_id") != job_id:
        raise FinalPhaseError("final_evidence_invalid")
    cleanup = evidence.get("cleanup")
    if (
        not isinstance(cleanup, Mapping)
        or cleanup.get("live_jobs") != []
        or cleanup.get("uncertain_submissions") != []
        or cleanup.get("submission_intents") != []
    ):
        raise FinalPhaseError("final_cleanup_unproven")
    if (
        evidence.get("source_identity_verified") is False
        or evidence.get("worker_cleanup_verified") is False
    ):
        raise FinalPhaseError("final_execution_integrity_contradiction")
    evidence_status = evidence.get("status")
    if not isinstance(evidence_status, str) or not evidence_status:
        raise FinalPhaseError("final_evidence_invalid")
    if status != "completed":
        if evidence_status == "completed":
            raise FinalPhaseError("final_evidence_invalid")
        return
    if (
        evidence_status != "completed"
        or not isinstance(job_id, str)
        or not job_id
        or evidence.get("source_identity_verified") is not True
        or evidence.get("worker_cleanup_verified") is not True
        or evidence.get("task_id") != final_task_id(cell.task_id)
        or evidence.get("split") != "final"
        or evidence.get("metric") != "accuracy"
        or evidence.get("direction") != "maximize"
        or evidence.get("evaluator") != "external_trusted"
        or evidence.get("answer_sha256") != answer_sha256
        or evidence.get("prediction_sha256") != prediction_sha256
        or evidence.get("submitted_source_sha256") != winner_source_sha256
        or evidence.get("entrypoint_sha256") != winner_source_sha256
        or not isinstance(evidence.get("entrypoint"), str)
        or not evidence["entrypoint"]
        or evidence.get("registry_version") != registry_version
    ):
        raise FinalPhaseError("final_evidence_invalid")
    try:
        deploy.validate_job_snapshot(evidence.get("prediction_snapshot"), job_id)
    except ComparisonSpecError as exc:
        raise FinalPhaseError("final_evidence_invalid") from exc
    policy = load_policy()
    if (
        evidence.get("policy_version") != policy.policy_version
        or evidence.get("policy_sha256") != policy.policy_sha256
    ):
        raise FinalPhaseError("final_evidence_invalid")
    if rows is not None and evidence.get("row_count") != rows:
        raise FinalPhaseError("final_evidence_invalid")
    if evidence.get("source_sha256") != _single_file_tree_sha256(
        str(evidence["entrypoint"]), winner_bytes
    ):
        raise FinalPhaseError("final_evidence_invalid")


def recompute_accuracy(
    prediction_bytes: bytes, answer_bytes: bytes
) -> tuple[float, int]:
    """Independently recompute accuracy from prediction and answer CSVs.

    Strict: both files are ``id,label`` with matching id sets and binary
    labels; any structural deviation raises (fail closed).
    """

    def _parse(data: bytes, rule: str) -> dict[str, int]:
        try:
            reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
            if reader.fieldnames is None or list(reader.fieldnames) != ["id", "label"]:
                raise ValueError("header")
            rows: dict[str, int] = {}
            for row in reader:
                label = int(row["label"])
                if label not in (0, 1) or row["id"] in rows:
                    raise ValueError("row")
                rows[row["id"]] = label
            return rows
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            raise FinalPhaseError(rule) from exc

    predictions = _parse(prediction_bytes, "final_prediction_malformed")
    answers = _parse(answer_bytes, "final_answer_malformed")
    if not answers or set(predictions) != set(answers):
        raise FinalPhaseError("final_prediction_rows_mismatch")
    correct = sum(1 for row_id, label in answers.items() if predictions[row_id] == label)
    return correct / len(answers), len(answers)


class SandboxFinalExecutor:
    """Real final executor over the isolated 6581 stack (EXP-007/008).

    Uses the PHASE-SPECIFIC registry written by the verified publication
    (explicit ``registry_path``; never the tracked search registry),
    submits the frozen program bytes as ONE CPU job on the final-phase
    task (``comparison_<task>_final``; the dispatcher issues the exact
    data/final read grant from the pinned registry mapping), tracks live
    jobs/submission intents DURABLY (observer -> live-jobs file under the
    REQUIRED ``state_dir``; tracking is never silently disabled) and
    cancels/verifies cleanup on every exceptional path, then requires the
    full trusted-evidence chain: source gate + executed-source/tree/policy
    hashes, external evaluator identity (split final, accuracy, maximize,
    answer hash, registry version), the actual source-identity/worker-
    cleanup facts and verified empty cleanup sets. Source-identity or
    cleanup contradictions raise on FAILED jobs too. A genuine clean
    candidate failure — including an ordinary job timeout whose
    cancellation and cleanup are fully proven — returns ``status="failed"``
    (raw null / utility 0); unknown submissions, identity contradictions,
    unproven cleanup and infrastructure errors raise instead.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        registry: Any = None,
        registry_path: Path | None = None,
        timeout_seconds: float = JOB_TIMEOUT_SECONDS,
        state_dir: Path | None = None,
        deadline_fn: Callable[[], float | None] | None = None,
    ) -> None:
        if state_dir is None:
            # Durable submission/live-job tracking is mandatory: without it
            # an interrupted submission could never be accounted for.
            raise FinalPhaseError("final_state_dir_required")
        self._repo_root = repo_root
        self._timeout = float(timeout_seconds)
        self._state_dir = state_dir
        # The ORIGINAL experiment deadline epoch (from the ledger), kept
        # DISTINCT from the ordinary per-job timeout: an ordinary clean job
        # timeout is a candidate failure, only the global deadline is a
        # global stop.
        self._deadline_fn = deadline_fn
        if registry is None:
            if registry_path is None:
                raise FinalPhaseError("final_registry_required")
            from research.evaluator.service import EvaluatorRegistry

            registry = EvaluatorRegistry.load(registry_path)
        self._registry = registry

    def _persist_live_jobs(
        self, cell_key: str, client: Any, submission_intents: set[str]
    ) -> None:
        directory = self._state_dir / cell_key.replace(":", "-")
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": FINAL_LIVE_JOBS_SCHEMA,
            "cell_key": cell_key,
            "submission_intents": sorted(submission_intents),
            "live_jobs": sorted(client.live_job_ids()),
            "uncertain_submissions": sorted(client.uncertain_submission_ids()),
            "updated_at": time.time(),
        }
        _write_durable_bytes(
            directory / "live-jobs.json",
            (canonical_json(payload) + "\n").encode("utf-8"),
        )

    def execute(
        self,
        *,
        task_id: str,
        source_bytes: bytes,
        cell_key: str,
        timeout_seconds: float,
    ) -> FinalExecution:
        from research.adapters.sandbox_eval_client import SandboxEvalClient
        from research.comparison import deploy
        from research.contracts.budget import CancellationToken
        from research.contracts.results import MetricDirection, metric_direction
        from research.search.acceptance_inner import load_sandbox_api_key
        from research.search.sandbox_task import _submitted_bytes

        spec = self._registry.task(task_id)
        if spec.split != "final" or spec.metric != "accuracy":
            raise FinalPhaseError(f"final_task_identity_mismatch:{task_id}")
        if metric_direction(spec.metric) is not MetricDirection.MAXIMIZE:
            raise FinalPhaseError(f"final_task_identity_mismatch:{task_id}")
        grant = getattr(spec, "worker_data_public_path", None)
        if not isinstance(grant, str) or not grant.endswith("/data/final"):
            raise FinalPhaseError(f"final_grant_missing:{task_id}")
        timeout = min(float(timeout_seconds), self._timeout)
        if not math.isfinite(timeout) or timeout < 1.0:
            # Never round a sub-second remainder up: the ORIGINAL remaining
            # deadline bounds the network/poll/job waits exactly.
            raise FinalPhaseError("final_timeout_invalid")
        timeout_int = int(timeout)
        source = source_bytes.decode("utf-8", errors="surrogatepass")
        submission_intents: set[str] = set()

        def _observer(event: Any) -> None:
            # Durable live-job/submission-intent tracking BEFORE polling,
            # surviving interruption for emergency cleanup (EXP-005
            # conventions). `client` is bound before any event can fire.
            if isinstance(event, Mapping):
                if event.get("event") == "submission_intent":
                    submission_intents.add(str(event.get("trace_id")))
                elif event.get("event") == "job_registered":
                    submission_intents.discard(str(event.get("trace_id")))
            self._persist_live_jobs(cell_key, client, submission_intents)

        client = SandboxEvalClient(
            endpoint=deploy.GATEWAY_URL,
            api_key=load_sandbox_api_key(self._repo_root / deploy.AUTH_FILE),
            require_cleanup=True,
            observer=_observer,
        )
        self._persist_live_jobs(cell_key, client, submission_intents)

        def _cleanup_state() -> dict[str, Any]:
            return {
                "live_jobs": sorted(client.live_job_ids()),
                "uncertain_submissions": sorted(client.uncertain_submission_ids()),
                "submission_intents": sorted(submission_intents),
            }

        def _cleanup_unproven() -> bool:
            return bool(
                client.live_job_ids()
                or client.uncertain_submission_ids()
                or submission_intents
            )

        # TWO DISTINCT LIMITS: the ordinary per-job timeout stays
        # ``timeout`` (the frozen job cap, possibly clamped by the caller to
        # the remaining global time), while the cancellation token is bound
        # to the ORIGINAL experiment deadline epoch. An ordinary clean job
        # timeout is then raised by the client as TimeoutError (candidate
        # failure below); only actual global deadline expiry cancels.
        deadline_epoch = self._deadline_fn() if self._deadline_fn is not None else None
        token = (
            CancellationToken(deadline_epoch=deadline_epoch)
            if deadline_epoch is not None
            else CancellationToken.after(timeout)
        )
        try:
            result = client.submit_and_wait(
                source,
                name=f"comparison-final-{cell_key.replace(':', '-')}",
                data_dir=grant,
                resource_type="cpu",
                timeout=timeout_int,
                cancellation_token=token,
                task_id=task_id,
            )
        except TimeoutError:
            # Ordinary job timeout: the client already cancelled the remote
            # job with cleanup evidence attached. With cleanup fully proven
            # this is a CANDIDATE-level failure (raw null / utility 0), not
            # an infrastructure stop.
            client.cancel_all_live()
            self._persist_live_jobs(cell_key, client, submission_intents)
            if _cleanup_unproven():
                raise FinalPhaseError("final_cleanup_unproven") from None
            return FinalExecution(
                status="failed",
                job_id=None,
                score=None,
                prediction_bytes=None,
                prediction_sha256=None,
                cleanup_verified=True,
                reason="timeout",
                evidence={
                    "job_id": None,
                    "status": "timeout",
                    "cleanup": _cleanup_state(),
                },
            )
        except Exception as exc:
            # Cancellation + cleanup verification on EVERY exceptional path;
            # unproven cleanup is the integrity failure, never the original
            # exception alone.
            client.cancel_all_live()
            self._persist_live_jobs(cell_key, client, submission_intents)
            if _cleanup_unproven():
                raise FinalPhaseError("final_cleanup_unproven") from exc
            raise
        self._persist_live_jobs(cell_key, client, submission_intents)
        cleanup_ok = not _cleanup_unproven()
        raw = result.raw if isinstance(result.raw, dict) else {}
        body_raw = raw.get("result")
        body: dict[str, Any] = body_raw if isinstance(body_raw, dict) else {}
        cleanup_evidence = _cleanup_state()
        # Cleanup and source-identity contradictions are global integrity
        # failures on FAILED jobs as well as successful ones (EXP-005
        # conventions); absent keys on never-submitted outcomes are not.
        if not cleanup_ok or body.get("worker_cleanup_verified") is False:
            raise FinalPhaseError("final_cleanup_unproven")
        if body.get("source_identity_verified") is False:
            raise FinalPhaseError("final_source_identity_unverified")
        if result.status != "completed" or result.score is None:
            evidence: dict[str, Any] = {
                "job_id": result.job_id,
                "status": str(result.status),
                "cleanup": cleanup_evidence,
            }
            for fact in ("source_identity_verified", "worker_cleanup_verified"):
                if fact in body:
                    evidence[fact] = body[fact] is True
            return FinalExecution(
                status="failed",
                job_id=result.job_id,
                score=None,
                prediction_bytes=None,
                prediction_sha256=None,
                cleanup_verified=True,
                reason=str(result.status),
                evidence=evidence,
            )
        evaluation = body.get("evaluation")
        gate = body.get("source_gate")
        if not isinstance(evaluation, dict) or not isinstance(gate, dict):
            raise FinalPhaseError("final_evidence_missing")
        # A completed job must carry the ACTUAL trusted facts, not merely
        # lack explicit contradictions.
        if body.get("source_identity_verified") is not True:
            raise FinalPhaseError("final_source_identity_unverified")
        if body.get("worker_cleanup_verified") is not True:
            raise FinalPhaseError("final_cleanup_unproven")
        submitted = _submitted_bytes(source)
        submitted_sha = hashlib.sha256(submitted).hexdigest()
        if gate.get("allowed") is not True or not gate.get("entrypoint"):
            raise FinalPhaseError("final_source_gate_refused")
        from research.contracts.results import hash_source

        if gate.get("entrypoint_sha256") != hash_source(submitted):
            raise FinalPhaseError("final_source_gate_mismatch")
        if gate.get("source_sha256") != _single_file_tree_sha256(
            str(gate["entrypoint"]), submitted
        ):
            raise FinalPhaseError("final_source_gate_mismatch")
        policy = load_policy()
        if (
            gate.get("policy_version") != policy.policy_version
            or gate.get("policy_sha256") != policy.policy_sha256
        ):
            raise FinalPhaseError("final_source_gate_mismatch")
        if evaluation.get("evaluator") != "external_trusted":
            raise FinalPhaseError("final_evaluator_untrusted")
        if (
            evaluation.get("task_id") != task_id
            or evaluation.get("split") != "final"
            or evaluation.get("metric") != "accuracy"
            or evaluation.get("direction") != "maximize"
            or evaluation.get("answer_sha256") != spec.answer_sha256
            or evaluation.get("registry_version") != self._registry.registry_version
        ):
            raise FinalPhaseError("final_evaluation_identity_mismatch")
        score = evaluation.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) != float(result.score)
        ):
            raise FinalPhaseError("final_score_invalid")
        prediction_sha256 = evaluation.get("prediction_sha256")
        snapshot = evaluation.get("prediction_snapshot")
        if not isinstance(prediction_sha256, str) or not isinstance(snapshot, str):
            raise FinalPhaseError("final_prediction_evidence_missing")
        try:
            storage_root = self._repo_root / deploy.DISPATCHER_STORAGE_ROOT
            host_path = deploy.job_snapshot_host_path(
                snapshot, result.job_id, storage_root=storage_root
            )
        except ComparisonSpecError as exc:
            raise FinalPhaseError("final_prediction_path_invalid") from exc
        try:
            prediction_bytes = host_path.read_bytes()
        except OSError as exc:
            raise FinalPhaseError("final_prediction_unreadable") from exc
        if hashlib.sha256(prediction_bytes).hexdigest() != prediction_sha256:
            raise FinalPhaseError("final_prediction_hash_mismatch")
        completed_evidence: dict[str, Any] = {
            "job_id": result.job_id,
            "status": "completed",
            "task_id": task_id,
            "split": "final",
            "metric": "accuracy",
            "direction": "maximize",
            "evaluator": "external_trusted",
            "answer_sha256": evaluation["answer_sha256"],
            "registry_version": evaluation["registry_version"],
            "row_count": evaluation.get("row_count"),
            "entrypoint": gate["entrypoint"],
            "entrypoint_sha256": gate["entrypoint_sha256"],
            "source_sha256": gate.get("source_sha256"),
            "policy_version": gate.get("policy_version"),
            "policy_sha256": gate.get("policy_sha256"),
            "submitted_source_sha256": submitted_sha,
            "source_identity_verified": True,
            "worker_cleanup_verified": True,
            "prediction_snapshot": snapshot,
            "prediction_sha256": prediction_sha256,
            "cleanup": cleanup_evidence,
        }
        return FinalExecution(
            status="completed",
            job_id=result.job_id,
            score=float(score),
            prediction_bytes=prediction_bytes,
            prediction_sha256=prediction_sha256,
            cleanup_verified=True,
            reason=None,
            evidence=completed_evidence,
        )


def sandbox_final_executor(
    repo_root: Path,
    publication_record: Mapping[str, Any],
    *,
    state_dir: Path,
    timeout_seconds: float = JOB_TIMEOUT_SECONDS,
    deadline_fn: Callable[[], float | None] | None = None,
) -> SandboxFinalExecutor:
    """Build the real executor bound to the verified publication's PHASE
    registry (explicit selection; the tracked search registry is never
    used for final evaluation). The phase registry file must exist and
    hash-match the publication record. ``state_dir`` is REQUIRED: durable
    submission/live-job tracking is never disabled. ``deadline_fn`` returns
    the ORIGINAL experiment deadline epoch (ledger-anchored) so network and
    poll waits stay bound to it, distinct from the ordinary job timeout."""
    root_raw = publication_record.get("publication_root")
    rel = publication_record.get("registry_file")
    if not isinstance(root_raw, str) or not isinstance(rel, str) or not rel:
        raise FinalPhaseError("final_registry_required")
    path = Path(root_raw) / rel
    if path.is_symlink() or not path.exists():
        raise FinalPhaseError("final_registry_missing")
    files = publication_record.get("files")
    expected = files.get(rel) if isinstance(files, Mapping) else None
    if not isinstance(expected, str) or _sha256(path.read_bytes()) != expected:
        raise FinalPhaseError("final_registry_mismatch")
    return SandboxFinalExecutor(
        repo_root,
        registry_path=path,
        timeout_seconds=timeout_seconds,
        state_dir=state_dir,
        deadline_fn=deadline_fn,
    )


def publish_final_data(
    orch: Orchestrator,
    ledger: ComparisonLedger,
    barrier: Mapping[str, Any],
    *,
    now_fn: Callable[[], float] = time.time,
    **publish_kwargs: Any,
) -> dict[str, Any]:
    """Barrier-gated final data publication (idempotent, hash-bound).

    The barrier is RE-COMPUTED from disk and must equal the supplied
    evidence exactly; the ledger then DURABLY enters the irreversible
    final phase (bound to the barrier identity) BEFORE any final input is
    copied; publication bytes are hash-bound to the frozen manifest and
    written ONLY to the ignored runtime publication root by
    ``data.publish_final_refit``.
    """
    try:
        fresh = orch.verify_final_barrier(ledger)
    except OrchestrationError as exc:
        ledger.stop(f"final_barrier_unverified:{exc}")
        raise FinalPhaseError(str(exc)) from exc
    if dict(barrier) != fresh:
        raise FinalPhaseError("final_barrier_stale")
    # Irreversible transition, bound to THIS barrier, BEFORE any final
    # input is copied: afterwards every model reservation/generation entry
    # refuses. Idempotent for the same barrier (crash-safe resume); a
    # different barrier is an integrity failure that stops the ledger.
    try:
        ledger.enter_final_phase(str(fresh["barrier_identity"]))
    except ComparisonBudgetError as exc:
        ledger.stop(f"final_phase_transition_failed:{exc}")
        raise FinalPhaseError(f"final_phase_transition_failed:{exc}") from exc
    try:
        publish_kwargs.pop("evidence_root", None)
        publish_kwargs.pop("ledger_evidence_path", None)
        publication = publish_final_refit(
            orchestrator=orch,
            ledger=ledger,
            barrier_evidence=fresh,
            repo_root=orch.repo_root,
            **publish_kwargs,
        )
    except ComparisonDataError as exc:
        # A publication failure is evidence-grade: stop the ledger durably.
        ledger.stop(f"final_publication_failed:{exc.reason}")
        raise FinalPhaseError(f"final_publication_failed:{exc.reason}") from exc
    except OSError as exc:
        # A raw durable-write I/O failure inside publication is equally
        # evidence-grade: stop the original ledger before propagating.
        ledger.stop(f"final_publication_failed:write_io")
        raise FinalPhaseError("final_publication_failed:write_io") from exc
    record_path = orch.runtime_dir / "final" / FINAL_PUBLICATION_RECORD
    record = dict(publication)
    record["recorded_at"] = float(now_fn())
    try:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        _write_durable(record_path, canonical_json(record) + "\n")
    except Exception as exc:
        # A durable-write failure of the publication record is
        # evidence-grade: stop the original ledger before propagating.
        ledger.stop("final_publication_failed:record_write")
        raise FinalPhaseError("final_publication_failed:record_write") from exc
    return publication


def _load_publication_record(
    orch: Orchestrator,
    barrier: Mapping[str, Any],
    *,
    public_root: Path | None = None,
    evaluator_root: Path | None = None,
) -> dict[str, Any]:
    """Load the durable publication record AND re-prove the original inputs.

    Re-hashes every published file (inputs, trusted answers, phase registry)
    against the record, then INDEPENDENTLY anchors the publication to the
    frozen data: test.csv/final_answer.csv hashes must equal the frozen
    cell task bindings, and each train.csv must be the EXACT refit
    recomputed from the binding-pinned train/validation artifacts —
    rehashing files only against a freely replaced publication record never
    proves the original inputs.
    """
    path = orch.runtime_dir / "final" / FINAL_PUBLICATION_RECORD
    if path.is_symlink() or not path.exists():
        raise FinalPhaseError("final_data_not_published")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FinalPhaseError("final_data_not_published") from exc
    if (
        not isinstance(record, dict)
        or record.get("schema") != "istratde-comparison-final-publication.v1"
        or record.get("barrier_identity") != barrier.get("barrier_identity")
        or not isinstance(record.get("registry_version"), str)
        or not record["registry_version"]
    ):
        raise FinalPhaseError("final_publication_mismatch")
    root_raw = record.get("publication_root")
    files = record.get("files")
    if not isinstance(root_raw, str) or not isinstance(files, dict):
        raise FinalPhaseError("final_publication_mismatch")
    root = Path(root_raw)
    for rel, expected in files.items():
        if not isinstance(rel, str) or not isinstance(expected, str):
            raise FinalPhaseError("final_publication_mismatch")
        file_path = root / rel
        if file_path.is_symlink() or not file_path.exists():
            raise FinalPhaseError(f"final_publication_file_missing:{rel}")
        try:
            actual = _sha256(file_path.read_bytes())
        except OSError as exc:
            raise FinalPhaseError(f"final_publication_file_missing:{rel}") from exc
        if actual != expected:
            raise FinalPhaseError(f"final_publication_file_tampered:{rel}")
    # Independent frozen-data anchor (same derivation as publication).
    bindings_by_task: dict[str, Any] = {}
    for cell in orch.cells:
        existing = bindings_by_task.setdefault(cell.task_id, cell.task_binding)
        if existing != cell.task_binding:
            raise FinalPhaseError("final_publication_mismatch")
    public_base = (
        Path(public_root)
        if public_root is not None
        else orch.repo_root / "tasks" / "comparison"
    )
    evaluator_base = (
        Path(evaluator_root)
        if evaluator_root is not None
        else orch.repo_root / "research" / "evaluator"
    )
    for task_id, binding in bindings_by_task.items():
        if (
            files.get(f"{task_id}/data/final/test.csv")
            != binding.final_features_sha256
            or files.get(f"answers/comparison_{task_id}/final_answer.csv")
            != binding.final_labels_sha256
        ):
            raise FinalPhaseError("final_publication_inputs_mismatch")
        train_rel = f"{task_id}/data/final/train.csv"
        published_train = files.get(train_rel)
        if not isinstance(published_train, str):
            raise FinalPhaseError("final_publication_inputs_mismatch")
        try:
            train = _read_bound(
                public_base / task_id / "data" / "public" / "train.csv",
                binding.train_csv_sha256,
            )
            validation_features = _read_bound(
                public_base / task_id / "data" / "public" / "test.csv",
                binding.validation_features_sha256,
            )
            validation_answer = _read_bound(
                evaluator_base
                / "tasks"
                / f"comparison_{task_id}"
                / "validation_answer.csv",
                binding.validation_labels_sha256,
            )
            derived = _refit_trainval_bytes(
                train, validation_features, validation_answer
            )
        except ComparisonDataError as exc:
            raise FinalPhaseError("final_publication_inputs_mismatch") from exc
        if _sha256(derived) != published_train:
            raise FinalPhaseError("final_publication_inputs_mismatch")
    return record


def verify_final_context(
    orch: Orchestrator,
    snapshot: Mapping[str, Any],
    *,
    public_root: Path | None = None,
    evaluator_root: Path | None = None,
) -> dict[str, Any]:
    """The verified final context SHARED by the offline audit and report.

    A persisted final outcome counts ONLY when the complete chain
    revalidates against the ORIGINAL validated ledger:

    1. the all-27-cell barrier recomputes from the actual terminal
       evidence (read-only, no live-ledger gates);
    2. the ledger holds an ENTERED final phase whose barrier_identity
       equals that recomputed barrier — an absent phase (rolled-back
       field/event) or a mismatched identity refuses every final claim,
       even when publication and result files are internally coherent;
    3. the hash-bound publication record revalidates against that barrier
       and the frozen data pins.

    Raises OrchestrationError/FinalPhaseError otherwise; a caller must
    treat a failed context as ZERO verified finals — the populated
    barrier/publication never leaks downstream. Read-only: no phase
    activation, no ledger stop, no mutation. The deadline is deliberately
    NOT re-enforced here: already-executed final results remain auditable
    after the original deadline as long as their original final-phase
    context still validates.
    """
    barrier = orch.compute_barrier_payload(snapshot)
    final_phase = snapshot.get("final_phase")
    if not isinstance(final_phase, dict):
        raise FinalPhaseError("final_phase_not_entered")
    if final_phase.get("barrier_identity") != barrier["barrier_identity"]:
        raise FinalPhaseError("final_phase_barrier_mismatch")
    publication = _load_publication_record(
        orch, barrier, public_root=public_root, evaluator_root=evaluator_root
    )
    return {
        "barrier": barrier,
        "final_phase": final_phase,
        "publication": publication,
    }


def _read_bound(path: Path, expected_sha256: str) -> bytes:
    """Read original input bytes pinned to an exact hash (never symlink)."""
    if path.is_symlink() or not path.exists():
        raise ComparisonDataError("final_source_missing", str(path))
    data = path.read_bytes()
    if _sha256(data) != expected_sha256:
        raise ComparisonDataError("manifest_binding_mismatch", str(path))
    return data


def _check_deadline(ledger: ComparisonLedger, cell_key: str) -> None:
    """The original 24h deadline is enforced at every final-phase boundary."""
    deadline = ledger.deadline_epoch
    if deadline is not None and ledger.now_fn() > deadline:
        ledger.stop("global_exhausted:deadline_expired_final")
        raise FinalPhaseError("deadline_expired")


def run_final_phase(
    orch: Orchestrator,
    ledger: ComparisonLedger,
    barrier: Mapping[str, Any],
    executor: FinalExecutor,
    *,
    now_fn: Callable[[], float] = time.time,
    sealed_root: Path | None = None,
    public_root: Path | None = None,
    evaluator_root: Path | None = None,
) -> dict[str, Any]:
    """Execute every frozen winner at most once; record truthful results.

    Resume-safe: a cell with a persisted final result is skipped ONLY after
    full re-validation (publication re-hash AND independent frozen-input
    anchor, intent re-hash, frozen winner re-hash, prediction bytes re-hash,
    pinned answer re-hash, independent accuracy recomputation); a durable
    intent WITHOUT a result means the process may have submitted —
    duplicates are forbidden and resume fails closed. Candidate failures
    record raw null / utility 0; integrity, cleanup, publication or
    infrastructure failures durably stop the ledger. No interrupted final
    submission is ever retried or replaced. ``public_root``/
    ``evaluator_root`` locate the binding-pinned train/validation artifacts
    for the publication anchor (default: the standard trees under
    ``orch.repo_root``).
    """
    if barrier.get("schema") != BARRIER_SCHEMA:
        raise FinalPhaseError("final_barrier_invalid")
    cells_map = barrier.get("cells")
    if not isinstance(cells_map, dict):
        raise FinalPhaseError("final_barrier_invalid")
    # Re-verify the ACTUAL search barrier from disk: a caller-supplied
    # mapping is never sufficient evidence on its own. Verification
    # failure is tamper-grade evidence: stop the ledger durably.
    try:
        fresh = orch.verify_final_barrier(ledger)
    except OrchestrationError as exc:
        ledger.stop(f"final_barrier_unverified:{exc}")
        raise FinalPhaseError(str(exc)) from exc
    if dict(barrier) != fresh:
        raise FinalPhaseError("final_barrier_stale")
    if ledger.stopped:
        raise FinalPhaseError("ledger_stopped")
    # The final phase must have been irreversibly entered (publication
    # does this); resume can never revert it.
    phase = ledger.final_phase
    if not isinstance(phase, dict) or phase.get("barrier_identity") != fresh[
        "barrier_identity"
    ]:
        ledger.stop("final_phase_not_entered")
        raise FinalPhaseError("final_phase_not_entered")
    try:
        publication = _load_publication_record(
            orch,
            fresh,
            public_root=public_root,
            evaluator_root=evaluator_root,
        )
    except FinalPhaseError as exc:
        # Publication read/parse/integrity failures durably stop the ledger.
        ledger.stop(f"final_publication_invalid:{exc}")
        raise
    from research.comparison.data import SEALED_ROOT

    sealed = sealed_root or SEALED_ROOT
    results: dict[str, Any] = {}
    for cell in orch.cells:
        cell_key = cell.cell_key()
        _check_deadline(ledger, cell_key)
        final_dir = orch.runtime_dir / "final" / cell_key.replace(":", "-")
        result_path = final_dir / "final.json"
        intent_path = final_dir / "intent.json"
        barrier_entry = cells_map.get(cell_key)
        if not isinstance(barrier_entry, dict):
            raise FinalPhaseError("final_barrier_invalid")
        if result_path.exists():
            try:
                results[cell_key] = _validate_final_result(
                    orch,
                    cell_key,
                    result_path,
                    ledger=ledger,
                    cell=cell,
                    barrier_entry=barrier_entry,
                    barrier_identity=str(fresh["barrier_identity"]),
                    publication=publication,
                    sealed_root=sealed,
                )
            except FinalPhaseError as exc:
                # Fabricated/missing/tampered persisted final evidence is an
                # integrity failure: durably stop the original ledger.
                ledger.stop(f"final_result_invalid:{cell_key}:{exc}")
                raise
            continue
        if intent_path.exists() or intent_path.is_symlink():
            # Crashed after (or during) submission: never execute twice.
            ledger.stop(f"final_interrupted:{cell_key}")
            raise FinalPhaseError(f"final_interrupted:{cell_key}")
        final_dir.mkdir(parents=True, exist_ok=True)
        if barrier_entry["terminal_status"] == "failed_no_valid_program":
            result = {
                "schema": FINAL_RESULT_SCHEMA,
                "cell_key": cell_key,
                "status": "no_valid_program",
                "barrier_identity": fresh["barrier_identity"],
                "raw_score": None,
                "utility": 0.0,
                "job_id": None,
                "prediction_sha256": None,
                "recomputed_accuracy": None,
                "finished_at": float(now_fn()),
            }
            try:
                _write_durable(result_path, canonical_json(result) + "\n")
            except OSError as exc:
                ledger.stop(f"final_write_failed:{cell_key}")
                raise FinalPhaseError(f"final_write_failed:{cell_key}") from exc
            results[cell_key] = result
            continue
        winner = barrier_entry["winner"]
        if not isinstance(winner, dict):
            raise FinalPhaseError("final_barrier_invalid")
        winner_path = (
            orch.runtime_dir / "cells" / cell_key.replace(":", "-") / "winner.py"
        )
        if winner_path.is_symlink() or not winner_path.exists():
            ledger.stop(f"final_winner_missing:{cell_key}")
            raise FinalPhaseError(f"final_winner_missing:{cell_key}")
        try:
            winner_bytes = winner_path.read_bytes()
        except OSError as exc:
            ledger.stop(f"final_winner_missing:{cell_key}")
            raise FinalPhaseError(f"final_winner_missing:{cell_key}") from exc
        if hashlib.sha256(winner_bytes).hexdigest() != winner["source_sha256"]:
            ledger.stop(f"final_winner_tampered:{cell_key}")
            raise FinalPhaseError(f"final_winner_tampered:{cell_key}")
        answer_rel = f"answers/comparison_{cell.task_id}/final_answer.csv"
        answer_sha256 = publication["files"].get(answer_rel)
        if not isinstance(answer_sha256, str):
            raise FinalPhaseError(f"final_answer_unbound:{cell_key}")
        # Final execution timeout is bounded by the ORIGINAL remaining
        # deadline (checked before AND immediately after execution). A
        # sub-second remainder is never rounded up to a full second: the
        # submission is refused instead.
        deadline = ledger.deadline_epoch
        remaining = (
            None
            if deadline is None
            else float(deadline) - float(ledger.now_fn())
        )
        if remaining is not None and remaining < 1.0:
            ledger.stop("global_exhausted:deadline_expired_final")
            raise FinalPhaseError("deadline_expired")
        timeout = (
            float(JOB_TIMEOUT_SECONDS)
            if remaining is None
            else min(float(JOB_TIMEOUT_SECONDS), remaining)
        )
        intent = {
            "schema": FINAL_INTENT_SCHEMA,
            "cell_key": cell_key,
            "task_id": cell.task_id,
            "final_task_id": final_task_id(cell.task_id),
            "winner": dict(winner),
            "barrier_identity": fresh["barrier_identity"],
            "answer_sha256": answer_sha256,
            "timeout_seconds": timeout,
            "submitted_at": float(now_fn()),
        }
        # Durable BEFORE-submit intent: a crash after this point can never
        # silently re-execute the frozen program.
        intent_text = canonical_json(intent) + "\n"
        _write_durable(intent_path, intent_text)
        intent_sha256 = hashlib.sha256(intent_text.encode("utf-8")).hexdigest()
        try:
            execution = executor.execute(
                task_id=final_task_id(cell.task_id),
                source_bytes=winner_bytes,
                cell_key=cell_key,
                timeout_seconds=timeout,
            )
        except Exception as exc:
            ledger.stop(f"final_infrastructure_failure:{cell_key}")
            raise FinalPhaseError(f"final_infrastructure_failure:{cell_key}") from exc
        # A job that FINISHES after the original deadline is not accepted:
        # re-check the ledger clock immediately after execution, before
        # recording anything.
        _check_deadline(ledger, cell_key)
        result = _translate_execution(
            orch,
            ledger,
            cell,
            winner,
            execution,
            now_fn,
            sealed,
            intent_sha256=intent_sha256,
            barrier_identity=str(fresh["barrier_identity"]),
            answer_sha256=answer_sha256,
            registry_version=str(publication["registry_version"]),
            winner_bytes=winner_bytes,
        )
        try:
            _write_durable(result_path, canonical_json(result) + "\n")
        except (OSError, OrchestrationError) as exc:
            # A durable result-write failure durably stops the ledger.
            ledger.stop(f"final_write_failed:{cell_key}")
            raise FinalPhaseError(f"final_write_failed:{cell_key}") from exc
        results[cell_key] = result
    return results


def _validate_final_result(
    orch: Orchestrator,
    cell_key: str,
    result_path: Path,
    *,
    ledger: ComparisonLedger | ReadOnlyComparisonLedger | None = None,
    cell: Any = None,
    barrier_entry: Mapping[str, Any] | None = None,
    barrier_identity: str | None = None,
    publication: Mapping[str, Any] | None = None,
    sealed_root: Path | None = None,
) -> dict[str, Any]:
    """Full re-verification of a persisted final result on resume.

    Re-hashes the durable intent, the frozen winner bytes, the persisted
    prediction bytes and the pinned sealed answer; independently recomputes
    accuracy; binds no_valid_program rows to the genuine barrier state.
    WITHOUT the complete barrier/publication/ledger context nothing is
    accepted as proven (fail closed). Fabricated, missing, cross-job or
    tampered artifacts are rejected, never accepted as cached truth.
    """

    def _bad(reason: str) -> FinalPhaseError:
        return FinalPhaseError(f"{reason}:{cell_key}")

    if result_path.is_symlink():
        raise _bad("final_result_invalid")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _bad("final_result_invalid") from exc
    if (
        not isinstance(result, dict)
        or result.get("schema") != FINAL_RESULT_SCHEMA
        or result.get("cell_key") != cell_key
    ):
        raise _bad("final_result_invalid")
    if (
        ledger is None
        or cell is None
        or barrier_entry is None
        or barrier_identity is None
        or publication is None
        or sealed_root is None
    ):
        # A bare score/hash claim without the full verification context is
        # unproven — resume NEVER accepts it.
        raise _bad("final_result_unproven")
    if result.get("barrier_identity") != barrier_identity:
        raise _bad("final_result_invalid")
    status = result.get("status")
    utility = result.get("utility")
    raw = result.get("raw_score")
    if status == "no_valid_program":
        # Bound to the GENUINE barrier state, not a caller claim.
        if (
            barrier_entry.get("terminal_status") != "failed_no_valid_program"
            or barrier_entry.get("winner") is not None
            or raw is not None
            or utility != 0.0
        ):
            raise _bad("final_result_invalid")
        return result
    if status not in ("completed", "failed"):
        raise _bad("final_result_invalid")
    # Executed results bind the durable pre-submit intent BY RE-HASH.
    intent_path = (
        orch.runtime_dir / "final" / cell_key.replace(":", "-") / "intent.json"
    )
    if intent_path.is_symlink() or not intent_path.exists():
        raise _bad("final_intent_missing")
    try:
        intent_bytes = intent_path.read_bytes()
        intent = json.loads(intent_bytes)
    except (OSError, ValueError) as exc:
        raise _bad("final_intent_missing") from exc
    if hashlib.sha256(intent_bytes).hexdigest() != result.get("intent_sha256"):
        raise _bad("final_intent_mismatch")
    winner = intent.get("winner") if isinstance(intent, dict) else None
    if (
        not isinstance(intent, dict)
        or intent.get("schema") != FINAL_INTENT_SCHEMA
        or intent.get("cell_key") != cell_key
        or intent.get("task_id") != cell.task_id
        or intent.get("final_task_id") != final_task_id(cell.task_id)
        or intent.get("barrier_identity") != barrier_identity
        or not isinstance(winner, dict)
        or winner != barrier_entry.get("winner")
        or result.get("winner_source_sha256") != winner.get("source_sha256")
        or not isinstance(result.get("winner_source_sha256"), str)
    ):
        raise _bad("final_intent_mismatch")
    # Frozen winner bytes re-hash (publication-independent source of truth).
    winner_path = (
        orch.runtime_dir / "cells" / cell_key.replace(":", "-") / "winner.py"
    )
    if winner_path.is_symlink() or not winner_path.exists():
        raise _bad("final_winner_missing")
    winner_bytes = winner_path.read_bytes()
    if hashlib.sha256(winner_bytes).hexdigest() != winner["source_sha256"]:
        raise _bad("final_winner_tampered")
    # The pinned answer hash must match the (re-hashed) publication record.
    answer_rel = f"answers/comparison_{cell.task_id}/final_answer.csv"
    published_answer = publication.get("files", {}).get(answer_rel)
    if (
        not isinstance(published_answer, str)
        or result.get("answer_sha256") != published_answer
        or intent.get("answer_sha256") != published_answer
    ):
        raise _bad("final_answer_unbound")
    answer_path = sealed_root / cell.task_id / "final_labels.csv"
    if answer_path.is_symlink() or not answer_path.exists():
        raise _bad("final_answer_missing")
    try:
        answer_bytes = answer_path.read_bytes()
    except OSError as exc:
        raise _bad("final_answer_missing") from exc
    if hashlib.sha256(answer_bytes).hexdigest() != published_answer:
        raise _bad("final_answer_mismatch")
    # The answer bytes must ALSO be the ORIGINAL frozen final labels of this
    # cell's task binding — a freely replaced publication record alone never
    # proves the original inputs.
    if (
        hashlib.sha256(answer_bytes).hexdigest()
        != cell.task_binding.final_labels_sha256
    ):
        raise _bad("final_answer_mismatch")
    if status == "failed":
        if raw is not None or utility != 0.0:
            raise _bad("final_result_invalid")
        try:
            _verify_execution_evidence(
                cell,
                status="failed",
                job_id=result.get("job_id"),
                prediction_sha256=None,
                winner_source_sha256=str(winner["source_sha256"]),
                answer_sha256=published_answer,
                evidence=result.get("evidence"),
                winner_bytes=winner_bytes,
                registry_version=str(publication["registry_version"]),
            )
        except FinalPhaseError as exc:
            raise _bad("final_evidence_invalid") from exc
        return result
    # status == "completed"
    prediction_sha256 = result.get("prediction_sha256")
    job_id = result.get("job_id")
    recomputed = result.get("recomputed_accuracy")
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not 0.0 <= float(raw) <= 1.0
        or utility != raw
        or not isinstance(prediction_sha256, str)
        or len(prediction_sha256) != 64
        or not isinstance(job_id, str)
        or not job_id
        or not isinstance(recomputed, (int, float))
        or isinstance(recomputed, bool)
    ):
        raise _bad("final_result_invalid")
    # Prediction BYTES must persist and re-hash (no hash-only claims).
    prediction_path = result_path.with_name(PREDICTION_FILENAME)
    if prediction_path.is_symlink() or not prediction_path.exists():
        raise _bad("final_prediction_missing")
    try:
        prediction_bytes = prediction_path.read_bytes()
    except OSError as exc:
        raise _bad("final_prediction_missing") from exc
    if hashlib.sha256(prediction_bytes).hexdigest() != prediction_sha256:
        raise _bad("final_prediction_mismatch")
    # Independent recomputation against the trusted sealed final labels.
    accuracy, rows = recompute_accuracy(prediction_bytes, answer_bytes)
    if (
        abs(accuracy - float(raw)) > _SCORE_TOLERANCE
        or abs(float(recomputed) - accuracy) > _SCORE_TOLERANCE
        or result.get("row_count") != rows
    ):
        raise _bad("final_score_recompute_mismatch")
    # The SAME full provenance verification as initial acceptance: exact
    # source bytes/tree/policy, trusted evaluator/registry/answer/row
    # identity, consistent job id and empty cleanup sets.
    try:
        _verify_execution_evidence(
            cell,
            status="completed",
            job_id=job_id,
            prediction_sha256=prediction_sha256,
            winner_source_sha256=str(winner["source_sha256"]),
            answer_sha256=published_answer,
            evidence=result.get("evidence"),
            winner_bytes=winner_bytes,
            registry_version=str(publication["registry_version"]),
            rows=rows,
        )
    except FinalPhaseError as exc:
        raise _bad("final_evidence_invalid") from exc
    return result


def _translate_execution(
    orch: Orchestrator,
    ledger: ComparisonLedger,
    cell: Any,
    winner: Mapping[str, Any],
    execution: FinalExecution,
    now_fn: Callable[[], float],
    sealed_root: Path,
    *,
    intent_sha256: str,
    barrier_identity: str,
    answer_sha256: str,
    registry_version: str,
    winner_bytes: bytes,
) -> dict[str, Any]:
    cell_key = cell.cell_key()
    if not isinstance(execution, FinalExecution):
        ledger.stop(f"final_executor_invalid:{cell_key}")
        raise FinalPhaseError(f"final_executor_invalid:{cell_key}")
    if execution.cleanup_verified is not True or not isinstance(
        execution.evidence, Mapping
    ):
        # Unresolved cleanup or missing provenance is a global integrity
        # stop, never a null row.
        ledger.stop(f"final_cleanup_unverified:{cell_key}")
        raise FinalPhaseError(f"final_cleanup_unverified:{cell_key}")
    if _sha256(winner_bytes) != winner["source_sha256"]:
        ledger.stop(f"final_winner_tampered:{cell_key}")
        raise FinalPhaseError(f"final_winner_tampered:{cell_key}")
    final_dir = orch.runtime_dir / "final" / cell_key.replace(":", "-")
    base = {
        "schema": FINAL_RESULT_SCHEMA,
        "cell_key": cell_key,
        "winner_source_sha256": winner["source_sha256"],
        "barrier_identity": barrier_identity,
        "intent_sha256": intent_sha256,
        "answer_sha256": answer_sha256,
        "finished_at": float(now_fn()),
    }
    if execution.status != "completed":
        try:
            _verify_execution_evidence(
                cell,
                status="failed",
                job_id=execution.job_id,
                prediction_sha256=None,
                winner_source_sha256=str(winner["source_sha256"]),
                answer_sha256=answer_sha256,
                evidence=execution.evidence,
                winner_bytes=winner_bytes,
                registry_version=registry_version,
            )
        except FinalPhaseError as exc:
            # Explicit contradictions (cleanup live jobs, source identity,
            # unproven cleanup sets) on a FAILED job are global stops.
            ledger.stop(f"final_evidence_invalid:{cell_key}")
            raise FinalPhaseError(f"final_evidence_invalid:{cell_key}") from exc
        return {
            **base,
            "status": "failed",
            "reason": execution.reason or "execution_failed",
            "raw_score": None,
            "utility": 0.0,
            "job_id": execution.job_id,
            "prediction_sha256": None,
            "recomputed_accuracy": None,
            "evidence": dict(execution.evidence),
        }
    score = execution.score
    if (
        execution.prediction_bytes is None
        or not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
        or execution.prediction_sha256
        != hashlib.sha256(execution.prediction_bytes).hexdigest()
        or execution.job_id is None
    ):
        ledger.stop(f"final_evidence_invalid:{cell_key}")
        raise FinalPhaseError(f"final_evidence_invalid:{cell_key}")
    # Independent recomputation against the TRUSTED sealed final labels
    # (the pinned answer bytes re-hash to the publication record).
    answer_path = sealed_root / cell.task_id / "final_labels.csv"
    if answer_path.is_symlink() or not answer_path.exists():
        ledger.stop(f"final_answer_missing:{cell_key}")
        raise FinalPhaseError(f"final_answer_missing:{cell_key}")
    answer_bytes = answer_path.read_bytes()
    if hashlib.sha256(answer_bytes).hexdigest() != answer_sha256:
        ledger.stop(f"final_answer_mismatch:{cell_key}")
        raise FinalPhaseError(f"final_answer_mismatch:{cell_key}")
    recomputed, rows = recompute_accuracy(execution.prediction_bytes, answer_bytes)
    if abs(recomputed - float(score)) > _SCORE_TOLERANCE:
        ledger.stop(f"final_score_recompute_mismatch:{cell_key}")
        raise FinalPhaseError(f"final_score_recompute_mismatch:{cell_key}")
    try:
        _verify_execution_evidence(
            cell,
            status="completed",
            job_id=execution.job_id,
            prediction_sha256=execution.prediction_sha256,
            winner_source_sha256=str(winner["source_sha256"]),
            answer_sha256=answer_sha256,
            evidence=execution.evidence,
            winner_bytes=winner_bytes,
            registry_version=registry_version,
            rows=rows,
        )
    except FinalPhaseError as exc:
        ledger.stop(f"final_evidence_invalid:{cell_key}")
        raise FinalPhaseError(f"final_evidence_invalid:{cell_key}") from exc
    # Persist the ACTUAL prediction bytes durably (audit evidence, not a
    # hash-only claim); a durable-write failure stops the original ledger.
    try:
        _write_durable_bytes(final_dir / PREDICTION_FILENAME, execution.prediction_bytes)
    except Exception as exc:
        ledger.stop(f"final_write_failed:{cell_key}")
        raise FinalPhaseError(f"final_write_failed:{cell_key}") from exc
    return {
        **base,
        "status": "completed",
        "raw_score": float(score),
        "utility": float(score),
        "job_id": execution.job_id,
        "prediction_sha256": execution.prediction_sha256,
        "recomputed_accuracy": recomputed,
        "row_count": rows,
        "evidence": dict(execution.evidence),
    }
