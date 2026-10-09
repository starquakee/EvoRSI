"""Validated config-level run index for the acceptance outer loop (US-009).

One COMPLETE, verified inner Evo run per config identity. The acceptance
design reuses the standalone first inner run as the outer iStratDE loop's
first evaluation (zero new model calls), so a lookup must only ever succeed
for a run whose evidence is complete, consistent and still intact ON EVERY
get — not only at put/load:

- identity binds seed/model/task/metric/policy version AND content
  (policy_sha256), the pinned public-data identity, the trusted registry
  content identity, the config hash and the runner commit;
- the bound artifacts must exist and match: operator trace, request trace,
  the search journal (generated candidate code) and the run result — the
  run result is hash-pinned AND re-parsed, and its contents must be
  consistent with the record (config/policy/status/score, unstopped ledger
  with no unresolved reservations, job evidence binding every submitted job
  to its source gate hash and trusted prediction/answer hashes);
- the run must actually cover the search: at least 3 inner generations and
  at least 6 main candidates (draft/improve/crossover; debug attempts never
  count toward the six), with a trusted, evidence-verified best result.

Anything else fails closed with a machine-readable reason.

This is deliberately separate from research.contracts.cache.ResultCache,
which caches single evaluation RESULTS; the index records WHOLE inner runs.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from research.contracts.persist import atomic_write_json

SCHEMA = "acceptance-run-index.v1"

_HEX64 = set("0123456789abcdef")

#: The pinned acceptance search shape (3 inner generations x 2 candidates).
MIN_INNER_GENERATIONS = 3
MIN_MAIN_CANDIDATES = 6
MAIN_OPERATORS = ("draft", "improve", "crossover")


class RunIndexError(RuntimeError):
    """Index corruption or an invalid record; the message is machine-readable."""


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX64 for c in value)


@dataclass(frozen=True)
class RunIdentity:
    """What makes an inner run the evaluation of ONE outer config vector.

    Binds the policy CONTENT (not just its version), the pinned public-data
    identity and the trusted registry content identity in addition to
    seed/model/task/metric/config/commit.
    """

    seed: int
    model: str
    task_id: str
    metric: str
    policy_version: str
    config_hash: str
    runner_commit: str
    policy_sha256: str = ""
    data_identity: str = ""
    registry_identity: str = ""

    def canonical(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def key(self) -> str:
        import hashlib

        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "model": self.model,
            "task_id": self.task_id,
            "metric": self.metric,
            "policy_version": self.policy_version,
            "config_hash": self.config_hash,
            "runner_commit": self.runner_commit,
            "policy_sha256": self.policy_sha256,
            "data_identity": self.data_identity,
            "registry_identity": self.registry_identity,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "RunIdentity":
        try:
            return RunIdentity(
                seed=int(data["seed"]),
                model=str(data["model"]),
                task_id=str(data["task_id"]),
                metric=str(data["metric"]),
                policy_version=str(data["policy_version"]),
                config_hash=str(data["config_hash"]),
                runner_commit=str(data["runner_commit"]),
                policy_sha256=str(data.get("policy_sha256") or ""),
                data_identity=str(data.get("data_identity") or ""),
                registry_identity=str(data.get("registry_identity") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunIndexError("run_identity_malformed") from exc


@dataclass(frozen=True)
class RunRecord:
    """One completed, verified inner run. ``put``/load/``get`` all enforce
    the invariants, including that the bound artifact FILES still exist with
    matching content — digest-shaped strings alone are not evidence."""

    identity: RunIdentity
    status: str  # "completed" | "failed"
    score: float | None  # trusted evaluator accuracy in [0, 1] when completed
    run_dir: str  # relative to the index file's directory
    run_result_sha256: str = ""
    journal_sha256: str = ""
    request_trace_sha256: str = ""
    main_candidates: int = 0
    generations_completed: int = 0
    best_job_id: str = ""
    job_ids: tuple[str, ...] = field(default_factory=tuple)
    prediction_artifacts: Mapping[str, str] = field(default_factory=dict)
    operator_trace_sha256: str = ""
    worker_cleanup_verified: bool = False
    operator_counts: Mapping[str, int] = field(default_factory=dict)
    termination: str = "completed"
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "status": self.status,
            "score": self.score,
            "run_dir": self.run_dir,
            "run_result_sha256": self.run_result_sha256,
            "journal_sha256": self.journal_sha256,
            "request_trace_sha256": self.request_trace_sha256,
            "main_candidates": self.main_candidates,
            "generations_completed": self.generations_completed,
            "best_job_id": self.best_job_id,
            "job_ids": sorted(self.job_ids),
            "prediction_artifacts": dict(self.prediction_artifacts),
            "operator_trace_sha256": self.operator_trace_sha256,
            "worker_cleanup_verified": self.worker_cleanup_verified,
            "operator_counts": dict(self.operator_counts),
            "termination": self.termination,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "RunRecord":
        try:
            score = data.get("score")
            cleanup = data["worker_cleanup_verified"]
            created_at = data.get("created_at", 0.0)
            if not isinstance(cleanup, bool):
                raise RunIndexError("run_record_malformed")
            if score is not None and (
                isinstance(score, bool) or not isinstance(score, (int, float))
            ):
                raise RunIndexError("run_record_malformed")
            if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
                raise RunIndexError("run_record_malformed")
            return RunRecord(
                identity=RunIdentity.from_dict(data["identity"]),
                status=str(data["status"]),
                score=None if score is None else float(score),
                run_dir=str(data["run_dir"]),
                run_result_sha256=str(data["run_result_sha256"]),
                journal_sha256=str(data["journal_sha256"]),
                request_trace_sha256=str(data["request_trace_sha256"]),
                main_candidates=int(data["main_candidates"]),
                generations_completed=int(data["generations_completed"]),
                best_job_id=str(data["best_job_id"]),
                job_ids=tuple(str(j) for j in data["job_ids"]),
                prediction_artifacts={
                    str(k): str(v) for k, v in dict(data.get("prediction_artifacts") or {}).items()
                },
                operator_trace_sha256=str(data["operator_trace_sha256"]),
                worker_cleanup_verified=cleanup,
                operator_counts={str(k): int(v) for k, v in dict(data["operator_counts"]).items()},
                termination=str(data.get("termination", "completed")),
                created_at=float(created_at),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunIndexError("run_record_malformed") from exc


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_artifact(root: Path, run_dir: Path, name: str, expected_sha256: str) -> None:
    if not _is_hex64(expected_sha256):
        raise RunIndexError(f"run_artifact_hash_invalid:{name}")
    path = root / run_dir / name
    try:
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            raise RunIndexError(f"run_artifact_mismatch:{name}")
    except OSError as exc:
        raise RunIndexError(f"run_artifact_unreadable:{name}") from exc


def validate_run_result_consistency(parsed: Mapping[str, Any], record: RunRecord) -> None:
    """The re-parsed run result must agree with the record and carry the
    trusted per-job evidence (source gate + prediction/answer hashes)."""
    if parsed.get("schema") != "acceptance-inner-result.v1":
        raise RunIndexError("run_result_schema_mismatch")
    if parsed.get("config_hash") != record.identity.config_hash:
        raise RunIndexError("run_result_config_mismatch")
    if parsed.get("policy_version") != record.identity.policy_version:
        raise RunIndexError("run_result_policy_mismatch")
    if parsed.get("status") != "completed" or record.status != "completed":
        raise RunIndexError("run_incomplete_not_indexable")
    if parsed.get("best_score") != record.score:
        raise RunIndexError("run_result_score_mismatch")
    budget = parsed.get("budget")
    if not isinstance(budget, Mapping):
        raise RunIndexError("run_result_budget_missing")
    if budget.get("stopped") is not None:
        raise RunIndexError("run_ledger_stopped")
    if budget.get("open_reservations"):
        raise RunIndexError("run_ledger_unresolved_reservations")
    jobs = parsed.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise RunIndexError("run_jobs_missing")
    best = None
    for job in jobs:
        if not isinstance(job, Mapping):
            raise RunIndexError("run_jobs_missing")
        job_id = job.get("job_id")
        if job_id is None:
            continue  # admission-denied candidates never reached the sandbox
        evaluation = job.get("evaluation")
        if job.get("evidence_verified") is not True:
            continue  # individual failed/rejected candidates may occur; they
            # are recorded as failures and carry no trusted score to bind
        if not isinstance(evaluation, Mapping) or not evaluation:
            raise RunIndexError("run_job_evidence_missing")
        if not _is_hex64(evaluation.get("prediction_sha256")):
            raise RunIndexError("run_job_prediction_unbound")
        if not _is_hex64(evaluation.get("answer_sha256")):
            raise RunIndexError("run_job_answer_unbound")
        if job_id == record.best_job_id:
            # The recorded best job must be a trusted, evidence-verified
            # evaluation with exactly the record's score (ties included).
            if job.get("score") != record.score:
                raise RunIndexError("run_best_result_untrusted")
            best = job
    if best is None:
        raise RunIndexError("run_best_result_untrusted")
    # Every evidence-verified job must have its trusted prediction snapshot
    # bytes bound into the run artifacts.
    artifacts = parsed.get("prediction_artifacts")
    if not isinstance(artifacts, Mapping):
        raise RunIndexError("run_prediction_artifacts_missing")
    for job in jobs:
        if job.get("evidence_verified") is True:
            job_id = str(job.get("job_id"))
            if not _is_hex64(artifacts.get(job_id)):
                raise RunIndexError("run_prediction_artifacts_missing")
            if artifacts[job_id] != job["evaluation"]["prediction_sha256"]:
                raise RunIndexError("run_prediction_artifacts_mismatch")
    if dict(artifacts) != dict(record.prediction_artifacts):
        raise RunIndexError("run_prediction_artifacts_mismatch")


def validate_completed_record(record: RunRecord, root: Path) -> None:
    """Fail closed unless the record is complete, verification-grade, covers
    the full inner search and its bound artifacts still exist and match."""
    if record.status != "completed":
        raise RunIndexError("run_incomplete_not_indexable")
    score = record.score
    if (
        score is None
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        raise RunIndexError("run_score_invalid")
    if not record.job_ids or any(not j for j in record.job_ids):
        raise RunIndexError("run_job_ids_missing")
    if record.worker_cleanup_verified is not True:
        raise RunIndexError("run_cleanup_unverified")
    identity = record.identity
    for name, value in (
        ("policy_sha256", identity.policy_sha256),
        ("data_identity", identity.data_identity),
        ("registry_identity", identity.registry_identity),
    ):
        if not _is_hex64(value):
            raise RunIndexError(f"run_identity_unbound:{name}")
    if record.main_candidates < MIN_MAIN_CANDIDATES:
        raise RunIndexError("run_main_candidates_missing")
    if record.generations_completed < MIN_INNER_GENERATIONS:
        raise RunIndexError("run_generations_incomplete")
    main_operator_calls = sum(
        int(record.operator_counts.get(name, 0)) for name in MAIN_OPERATORS
    )
    if main_operator_calls < MIN_MAIN_CANDIDATES:
        raise RunIndexError("run_operator_coverage_insufficient")
    if not record.best_job_id or record.best_job_id not in record.job_ids:
        raise RunIndexError("run_best_job_unbound")
    run_dir = Path(record.run_dir)
    if not record.run_dir or run_dir.is_absolute() or ".." in run_dir.parts:
        raise RunIndexError("run_dir_invalid")
    _check_artifact(root, run_dir, "operator-trace.jsonl", record.operator_trace_sha256)
    _check_artifact(root, run_dir, "request-trace.jsonl", record.request_trace_sha256)
    _check_artifact(root, run_dir, "journal.jsonl", record.journal_sha256)
    for artifact_job_id, artifact_sha in record.prediction_artifacts.items():
        # job ids are server-generated slugs; reject anything path-like.
        if not artifact_job_id or any(c in artifact_job_id for c in ("/", "\\", "..")):
            raise RunIndexError("run_prediction_artifact_path_invalid")
        _check_artifact(root, run_dir, f"predictions/{artifact_job_id}.csv", artifact_sha)
    result_path = root / run_dir / "run-result.json"
    if not _is_hex64(record.run_result_sha256):
        raise RunIndexError("run_artifact_hash_invalid:run-result.json")
    try:
        raw = result_path.read_bytes()
    except OSError as exc:
        raise RunIndexError("run_artifact_unreadable:run-result.json") from exc
    import hashlib

    if hashlib.sha256(raw).hexdigest() != record.run_result_sha256:
        raise RunIndexError("run_artifact_mismatch:run-result.json")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RunIndexError("run_result_malformed") from exc
    if not isinstance(parsed, dict):
        raise RunIndexError("run_result_malformed")
    validate_run_result_consistency(parsed, record)


class RunIndex:
    """Persistent validated run index; corrupt or unbound state fails closed.

    ``root`` (the index file's directory) anchors the artifact binding. Every
    ``get`` revalidates the record AND its bound artifacts, so tampering
    after indexing invalidates the hit.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._root = self._path.resolve().parent
        self._records: dict[str, RunRecord] = {}
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RunIndexError("run_index_corrupt") from exc
            if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
                raise RunIndexError("run_index_schema_mismatch")
            entries = raw.get("records")
            if not isinstance(entries, dict):
                raise RunIndexError("run_index_corrupt")
            for key, entry in entries.items():
                record = RunRecord.from_dict(entry)
                if record.identity.key() != key:
                    raise RunIndexError("run_index_identity_mismatch")
                # A loaded record must be as verified as a fresh put: a
                # failed, unproven or artifact-unbound run is never a hit.
                validate_completed_record(record, self._root)
                self._records[key] = record

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        return len(self._records)

    def get(self, identity: RunIdentity) -> RunRecord | None:
        """Return the verified record for this exact identity, or None.

        Revalidation happens on EVERY get: post-index tampering with any
        bound artifact (trace, request trace, journal, run result) fails
        closed instead of yielding a stale hit.
        """
        record = self._records.get(identity.key())
        if record is None:
            return None
        validate_completed_record(record, self._root)
        return record

    def put(self, record: RunRecord) -> None:
        """Index one completed, verified run (validated; atomic persist)."""
        validate_completed_record(record, self._root)
        key = record.identity.key()
        existing = self._records.get(key)
        if existing is not None and existing.to_dict() != record.to_dict():
            raise RunIndexError("run_index_conflict")
        self._records[key] = record
        atomic_write_json(self._path, {
            "schema": SCHEMA,
            "records": {k: r.to_dict() for k, r in sorted(self._records.items())},
        })
