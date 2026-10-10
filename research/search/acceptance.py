"""US-009 real-acceptance parent controller (runs under the legacy JAX venv).

Two modes:

- ``preflight`` — OFFLINE validation of everything the acceptance depends on.
  Never creates the persistent ledger, never reads credentials, never sends
  any model request. Safe to run any time.
- ``live`` — the one real acceptance. Refuses (before ledger/auth/model
  activity) unless the ignored supervisor review file binds the current git
  HEAD; then opens/creates the ONE persistent ledger (200k tokens / 30
  requests / 5400s, no reset) and orchestrates: standalone inner Evo run on
  the first decoded preflight vector -> validated run index -> native seeded
  iStratDE ask -> reuse the standalone run as the outer loop's first verified
  evaluation -> three more real inner runs -> one explicit tell (fitness =
  -accuracy; iStratDE minimizes natively) -> redacted summary.

The parent holds no credentials at all: the Evo child reads the sandbox key
and the managed Kimi token itself. The parent never holds the ledger flock
while a child runs (the child reopens the same path without reset).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from research.contracts.budget import BudgetError, BudgetLedger, BudgetLimits
from research.contracts.persist import atomic_write_json
from research.evaluator.service import EvaluatorError, EvaluatorRegistry
from research.search import acceptance_config as ac
from research.search.acceptance_config import (
    RoundAuthorizationRefused,
    RunnerReviewRefused,
    check_runner_review,
)
from research.search.acceptance_inner import (
    AcceptanceError,
    InnerRunSpec,
    config_identity_hash,
    load_sandbox_api_key,
)
from research.search.run_index import (
    MIN_MAIN_CANDIDATES,
    RunIdentity,
    RunIndex,
    RunIndexError,
    RunRecord,
)
from research.search.search_vector import RETAINED_DIMS, decode_retained, safety_gate

SUMMARY_SCHEMA = "us009-acceptance-summary.v1"
PREFLIGHT_SCHEMA = "us009-preflight.v1"

EVO_VENV_PYTHON = Path("OpenMLE-Evo") / ".venv" / "bin" / "python"
AIRA_EVO_SRC = Path("OpenMLE-Evo") / "third_party" / "aira-evo" / "src"
VENDORED_ISTRATDE_SRC = Path("research") / "vendor" / "istratde" / "src"


# --------------------------------------------------------------------------
# Preflight checks (offline; each returns a machine-readable check result)
# --------------------------------------------------------------------------

def _check(name: str, fn: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
    try:
        evidence = dict(fn())
        return {"ok": True, "reason": None, "evidence": evidence}
    except Exception as exc:
        reason = str(exc) or type(exc).__name__
        return {"ok": False, "reason": reason, "evidence": {}}


def check_validation_configs(repo_root: Path) -> Mapping[str, Any]:
    """The pinned Evo experiment/litellm YAMLs satisfy the US-007 contract."""
    import yaml  # noqa: PLC0415 - legacy/Evo venvs provide pyyaml

    from research.search.validation_config import validate_inner_evo_config

    experiment_path = repo_root / ac.VALIDATION_EXPERIMENT_YAML
    litellm_path = repo_root / ac.LITELLM_YAML
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    litellm = yaml.safe_load(litellm_path.read_text(encoding="utf-8"))
    if not isinstance(experiment, dict) or not isinstance(litellm, dict):
        raise AcceptanceError("config_file_malformed")
    solver = experiment.get("search", {}).get("runner", {}).get("solver", {})
    models = litellm.get("model_list") or []
    if len(models) != 1:
        raise AcceptanceError("litellm_model_list_must_have_one_entry")
    params = models[0].get("litellm_params", {})

    def need(mapping: Mapping[str, Any], key: str) -> Any:
        if key not in mapping:
            raise AcceptanceError(f"config_field_missing:{key}")
        return mapping[key]

    validate_inner_evo_config({
        "num_generations": need(solver, "num_generations"),
        "individuals_per_generation": need(solver, "individuals_per_generation"),
        "step_limit": min(need(experiment, "max_steps"), need(solver, "step_limit")),
        "num_islands": need(solver, "num_islands"),
        "num_generations_till_crossover": need(solver, "num_generations_till_crossover"),
        "llm_concurrency": need(experiment, "llm_concurrency"),
        "sandbox_concurrency": need(experiment.get("sandbox", {}), "concurrency"),
        "experience_enabled": need(solver.get("experience", {}), "enabled"),
        "prompt_memory_enabled": need(solver.get("experience", {}).get("prompt_memory", {}), "enabled"),
        "stream": need(params, "stream"),
        "max_output_tokens": need(params, "max_tokens"),
        "reasoning_effort": need(params, "reasoning_effort"),
    })
    if type(solver.get("max_llm_call_retries")) is not int or solver["max_llm_call_retries"] != 1:
        raise AcceptanceError("operator_generation_attempts_must_be_one")
    if params.get("model") != f"openai/{ac.MODEL_ID}":
        raise AcceptanceError("litellm_model_mismatch")
    if params.get("api_key") != ac.API_KEY_PLACEHOLDER:
        # A real credential in a tracked config is a hard failure.
        raise AcceptanceError("litellm_api_key_must_be_placeholder")
    if params.get("base_url") != ac.MODEL_BASE_URL:
        raise AcceptanceError("litellm_base_url_mismatch")
    if params.get("num_retries") != 0 or params.get("max_retries") != 0:
        raise AcceptanceError("litellm_retries_must_be_zero")
    return {
        "experiment": str(ac.VALIDATION_EXPERIMENT_YAML),
        "litellm": str(ac.LITELLM_YAML),
        "model": params.get("model"),
        "stream": params.get("stream"),
    }


def _ensure_native_env(repo_root: Path) -> None:
    """CPU JAX + vendored istratde precedence for the outer controller."""
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    vendor = str(repo_root / VENDORED_ISTRATDE_SRC)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def _native_istratde_vectors(repo_root: Path) -> list[list[float]]:
    """First native vendored-iStratDE ask (CPU JAX, seed 42, pop 4).

    Fails closed unless the imported package resolves under THIS repository
    (the explicit sys.path entry must override any installed/editable copy).
    """
    _ensure_native_env(repo_root)
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import istratde  # noqa: PLC0415
    from istratde.algorithms.jax import IStratDE  # noqa: PLC0415

    module_file = Path(istratde.__file__ or "").resolve()
    if not module_file.is_relative_to(repo_root.resolve()):
        raise AcceptanceError("istratde_not_vendored")
    if jax.default_backend() != "cpu":
        raise AcceptanceError("jax_platform_must_be_cpu")
    lb = jnp.array([dim.lower for dim in RETAINED_DIMS], dtype=jnp.float32)
    ub = jnp.array([dim.upper for dim in RETAINED_DIMS], dtype=jnp.float32)
    algo = IStratDE(lb=lb, ub=ub, pop_size=ac.POP_SIZE)
    state = algo.init(jax.random.PRNGKey(ac.SEED))
    population, _state = algo.ask(state)
    return [[float(v) for v in row] for row in population]


def check_preflight_vectors(repo_root: Path) -> Mapping[str, Any]:
    """Derive the first native vendored-iStratDE ask directly from seed 42 /
    pop 4 / vendored code. The supervisor's recorded preflight JSON is an
    OPTIONAL cross-check: compared when present (mismatch fails closed),
    never required and never a substitute for the actual derivation."""
    vectors = _native_istratde_vectors(repo_root)
    if len(vectors) != ac.POP_SIZE:
        raise AcceptanceError("outer_ask_population_mismatch")
    comparison = "no_prior_preflight_file"
    preflight_path = repo_root / ac.PREFLIGHT_VECTORS_FILE
    if preflight_path.exists():
        try:
            recorded = json.loads(preflight_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AcceptanceError("outer_ask_preflight_unavailable") from exc
        if recorded.get("seed") != ac.SEED or recorded.get("pop_size") != ac.POP_SIZE:
            raise AcceptanceError("outer_ask_preflight_mismatch")
        if vectors != recorded.get("vectors"):
            raise AcceptanceError("outer_ask_mismatch")
        comparison = "verified_against_recorded_preflight"
    for vector in vectors:
        safety_gate(decode_retained(vector))
    return {"pop_size": ac.POP_SIZE, "seed": ac.SEED, "preflight_comparison": comparison}


def check_public_data(repo_root: Path) -> Mapping[str, Any]:
    evidence: dict[str, str] = {}
    for relative, expected in ac.PUBLIC_DATA_HASHES.items():
        path = repo_root / relative
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise AcceptanceError(f"public_data_missing:{relative}") from exc
        if digest != expected:
            raise AcceptanceError(f"public_data_hash_mismatch:{relative}")
        evidence[relative] = digest
    return evidence


def check_evaluator_registry(repo_root: Path) -> Mapping[str, Any]:
    try:
        registry = EvaluatorRegistry.load(repo_root / ac.EVALUATOR_REGISTRY)
        spec = registry.task(ac.TASK_ID)
    except EvaluatorError as exc:
        raise AcceptanceError(f"evaluator_registry_unusable:{exc.reason}") from exc
    return {"task_id": ac.TASK_ID, "metric": spec.metric, "split": spec.split}


def check_gateway(url: str | None = None) -> Mapping[str, Any]:
    """Gateway answers on loopback (401 without auth proves API+auth up)."""
    gateway = url or ac.GATEWAY_URL
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(gateway + "/api/v1/jobs", method="GET")
    try:
        with opener.open(request, timeout=10) as response:
            code = response.status
    except urllib.error.HTTPError as exc:
        code = exc.code
    except (urllib.error.URLError, OSError) as exc:
        raise AcceptanceError("gateway_unreachable") from exc
    if code not in (200, 401, 403, 404, 405, 422):
        raise AcceptanceError(f"gateway_unexpected_status:{code}")
    return {"gateway": gateway, "status": code}


def check_auth_file(repo_root: Path) -> Mapping[str, Any]:
    """Private auth file exists with owner-only permissions (contents never
    printed; only the presence of the SANDBOX_API_KEYS entry is checked)."""
    path = repo_root / ac.AUTH_FILE
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise AcceptanceError("sandbox_auth_unavailable") from exc
    if mode & 0o077:
        raise AcceptanceError("sandbox_auth_permissions_too_open")
    content = path.read_text(encoding="utf-8")
    if "SANDBOX_API_KEYS=" not in content:
        raise AcceptanceError("sandbox_auth_unavailable")
    return {"auth_file": str(ac.AUTH_FILE), "mode": oct(mode & 0o777)}


def check_ledger_absent(repo_root: Path) -> Mapping[str, Any]:
    """Prep-phase property: the one persistent acceptance ledger must NOT
    exist yet (live mode creates/continues it; preflight never does)."""
    path = repo_root / ac.RUNTIME_DIR / ac.LEDGER_FILENAME
    if path.exists():
        raise AcceptanceError("acceptance_ledger_already_exists")
    return {"ledger": str(path.relative_to(repo_root)), "exists": False}


def runner_review_status(repo_root: Path) -> Mapping[str, Any]:
    """Informational only: whether the live gate is currently satisfied."""
    try:
        check_runner_review(repo_root)
    except RunnerReviewRefused as exc:
        return {"bound": False, "reason": str(exc)}
    return {"bound": True, "reason": None}


def run_preflight_checks(repo_root: Path, *, include_ledger_absence: bool = True) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "validation_configs": _check("validation_configs", lambda: check_validation_configs(repo_root)),
        "preflight_vectors": _check("preflight_vectors", lambda: check_preflight_vectors(repo_root)),
        "public_data": _check("public_data", lambda: check_public_data(repo_root)),
        "evaluator_registry": _check("evaluator_registry", lambda: check_evaluator_registry(repo_root)),
        "gateway": _check("gateway", check_gateway),
        "auth_file": _check("auth_file", lambda: check_auth_file(repo_root)),
    }
    if include_ledger_absence:
        checks["acceptance_ledger_absent"] = _check(
            "acceptance_ledger_absent", lambda: check_ledger_absent(repo_root)
        )
    return checks


def preflight(repo_root: Path, output: Path | None = None) -> int:
    checks = run_preflight_checks(repo_root)
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "offline_preflight_only": True,
        "model_calls": 0,
        "ledger_created": False,
        "model_credentials_read": False,
        "sandbox_auth_checked": True,
        "checks": checks,
        "runner_review": runner_review_status(repo_root),
        "ok": all(c["ok"] for c in checks.values()),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0 if report["ok"] else 1


# --------------------------------------------------------------------------
# Live orchestration
# --------------------------------------------------------------------------

#: Bounded grace after the shared ledger deadline for the child to finish
#: cancellation/checkpointing — no new model calls are possible then (the
#: ledger is exhausted/stopped at the deadline).
CHILD_CLEANUP_GRACE_SECONDS = 120.0


def _default_cleanup_client(repo_root: Path) -> Any:
    """Real sandbox client for emergency cancellation (controller side)."""
    from research.adapters.sandbox_eval_client import SandboxEvalClient

    return SandboxEvalClient(
        endpoint=ac.GATEWAY_URL,
        api_key=load_sandbox_api_key(repo_root / ac.AUTH_FILE),
        require_cleanup=True,
    )


def cleanup_child_jobs(
    run_dir: Path,
    *,
    repo_root: Path,
    client_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Cancel a child's recorded live jobs with explicit cleanup proof.

    Reads the child's persisted live-jobs registry (submission intents are
    persisted BEFORE the POST, so a killed child cannot hide an in-flight
    submission). Unknown submission identities are never claimed clean.
    Writes emergency-cleanup.json evidence into the run dir.
    """
    evidence: dict[str, Any] = {
        "schema": "acceptance-emergency-cleanup.v1",
        "run_dir": run_dir.name,
        "jobs": [],
        "unknown_identities": [],
        "cleanup_verified": True,
        "reason": None,
    }
    registry_path = run_dir / "live-jobs.json"
    registry: Mapping[str, Any] | None = None
    try:
        loaded = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and loaded.get("schema") == "acceptance-live-jobs.v1":
            registry = loaded
    except (OSError, ValueError):
        registry = None
    if registry is None:
        # The child creates the registry before ANY submission is possible,
        # so a missing registry after a kill means the in-flight state is
        # unknown: fail closed.
        evidence["cleanup_verified"] = False
        evidence["reason"] = "live_job_registry_missing"
        atomic_write_json(run_dir / "emergency-cleanup.json", evidence)
        return evidence
    lists: dict[str, list[str]] = {}
    for field_name in ("live_jobs", "uncertain_submissions", "submission_intents"):
        value = registry.get(field_name)
        # A missing/malformed list is UNKNOWN state, never an empty one.
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            evidence["cleanup_verified"] = False
            evidence["reason"] = "live_job_registry_malformed"
            atomic_write_json(run_dir / "emergency-cleanup.json", evidence)
            return evidence
        lists[field_name] = list(value)
    live_jobs = lists["live_jobs"]
    unknown = lists["uncertain_submissions"] + lists["submission_intents"]
    if live_jobs or unknown:
        try:
            factory = client_factory or (lambda: _default_cleanup_client(repo_root))
            client = factory()
        except Exception as exc:
            evidence["cleanup_verified"] = False
            evidence["reason"] = f"cleanup_client_unavailable:{type(exc).__name__}"
            atomic_write_json(run_dir / "emergency-cleanup.json", evidence)
            return evidence
        for job_id in live_jobs:
            cancel = client.cancel_job(job_id)
            evidence["jobs"].append(cancel)
            if cancel.get("cleanup_verified") is not True:
                evidence["cleanup_verified"] = False
        for trace_id in unknown:
            # Unknown submission identity: never claimed clean.
            evidence["unknown_identities"].append({
                "trace_id": trace_id,
                "cleanup_verified": False,
                "cancel_error": "submission_identity_unknown",
            })
            evidence["cleanup_verified"] = False
        if evidence["cleanup_verified"] is not True:
            evidence["reason"] = "child_cleanup_unproven"
    atomic_write_json(run_dir / "emergency-cleanup.json", evidence)
    return evidence


def _default_child_runner(
    spec: InnerRunSpec,
    spec_path: Path,
    *,
    repo_root: Path,
    cleanup_client_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Run one inner Evo acceptance child in the Evo venv (clean environment,
    no proxy variables; credentials are read by the child itself).

    The subprocess timeout is the SHARED ledger deadline plus a bounded
    cleanup grace (the child cannot start new model calls past the deadline).
    On timeout, ONLY this child process is killed and its recorded live jobs
    are cancelled through the real API client with explicit cleanup proof;
    unproven cleanup refuses continuation.
    """
    evo_python = repo_root / EVO_VENV_PYTHON
    if not evo_python.exists():
        raise AcceptanceError("evo_venv_missing")
    run_dir = repo_root / spec.run_dir
    ledger_path = repo_root / spec.ledger_path
    with BudgetLedger.open(ledger_path) as ledger:
        deadline = ledger.deadline_epoch
    remaining = deadline - time.time()
    if remaining <= 0:
        raise AcceptanceError("acceptance_deadline_exceeded")
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", str(repo_root)),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(repo_root), str(repo_root / "OpenMLE-Evo"), str(repo_root / AIRA_EVO_SRC)]
        ),
    }
    process = subprocess.Popen(
        [
            str(evo_python), "-m", "research.search.acceptance_inner",
            "run", "--spec", str(spec_path), "--repo-root", str(repo_root),
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _stdout, stderr = process.communicate(
            timeout=remaining + CHILD_CLEANUP_GRACE_SECONDS
        )
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # The child's own deadline stopped new model calls at the ledger
        # deadline; the grace elapsed too (or the parent was interrupted).
        # Kill ONLY this child process and cancel its recorded live jobs
        # through the real API client with explicit cleanup proof.
        process.kill()
        process.communicate()
        cleanup_child_jobs(
            run_dir, repo_root=repo_root, client_factory=cleanup_client_factory
        )
        raise AcceptanceError("child_deadline_exceeded")
    if stderr:
        (run_dir / "child-stderr.log").write_text(stderr[-8000:], encoding="utf-8")
    # Reconcile: even a normally exiting child must leave no live/unknown
    # submissions behind; cleanup must be explicitly proven.
    evidence = cleanup_child_jobs(
        run_dir, repo_root=repo_root, client_factory=cleanup_client_factory
    )
    if evidence["cleanup_verified"] is not True:
        raise AcceptanceError("child_cleanup_unproven")
    if process.returncode != 0:
        raise AcceptanceError(f"inner_run_exited:{process.returncode}")
    result_path = run_dir / "run-result.json"
    try:
        raw = result_path.read_bytes()
        result = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise AcceptanceError("inner_run_result_missing") from exc
    if not isinstance(result, dict) or result.get("schema") != "acceptance-inner-result.v1":
        raise AcceptanceError("inner_run_result_malformed")
    result["_run_result_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


class _NativeOuterAlgorithm:
    """Native vendored iStratDE ask/tell (CPU JAX); tell minimizes natively."""

    def __init__(self, repo_root: Path):
        self._repo_root = repo_root
        self._state: Any = None
        self._algo: Any = None

    def ask(self) -> list[list[float]]:
        _ensure_native_env(self._repo_root)
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415
        from istratde.algorithms.jax import IStratDE  # noqa: PLC0415

        lb = jnp.array([dim.lower for dim in RETAINED_DIMS], dtype=jnp.float32)
        ub = jnp.array([dim.upper for dim in RETAINED_DIMS], dtype=jnp.float32)
        self._algo = IStratDE(lb=lb, ub=ub, pop_size=ac.POP_SIZE)
        self._state = self._algo.init(jax.random.PRNGKey(ac.SEED))
        population, self._state = self._algo.ask(self._state)
        return [[float(v) for v in row] for row in population]

    def tell(self, fitness: Sequence[float]) -> None:
        import jax.numpy as jnp  # noqa: PLC0415

        if self._algo is None or self._state is None:
            raise AcceptanceError("outer_tell_before_ask")
        # Native iStratDE minimizes; accuracy maximizes -> fitness=-accuracy.
        self._state = self._algo.tell(self._state, jnp.array(fitness, dtype=jnp.float32))


def _record_from_result(
    result: Mapping[str, Any],
    spec: InnerRunSpec,
    *,
    run_result_sha256: str,
    repo_root: Path,
) -> RunRecord:
    """Validate a child run-result against the spec and build the record."""
    if result.get("run_id") != spec.run_id:
        raise AcceptanceError("inner_run_id_mismatch")
    if result.get("decoded") != dict(spec.decoded):
        raise AcceptanceError("inner_run_decoded_mismatch")
    if result.get("config_hash") != config_identity_hash(spec.decoded):
        raise AcceptanceError("inner_run_config_hash_mismatch")
    status = result.get("status")
    if status not in ("completed", "failed"):
        raise AcceptanceError("inner_run_status_invalid")
    score = result.get("best_score")
    if score is not None and (
        isinstance(score, bool) or not isinstance(score, (int, float))
    ):
        raise AcceptanceError("inner_run_score_invalid")
    return RunRecord(
        identity=RunIdentity(
            seed=ac.SEED,
            model=ac.MODEL_ID,
            task_id=ac.TASK_ID,
            metric=ac.METRIC,
            policy_version=str(result.get("policy_version") or ""),
            config_hash=str(result.get("config_hash") or ""),
            runner_commit=spec.runner_commit,
            policy_sha256=str(result.get("policy_sha256") or ""),
            data_identity=ac.public_data_identity(),
            registry_identity=ac.registry_identity(repo_root),
        ),
        status=status,
        score=None if score is None else float(score),
        run_dir=str(Path(spec.run_dir).relative_to(Path(spec.runtime_dir))),
        run_result_sha256=run_result_sha256,
        journal_sha256=str(result.get("journal_sha256") or ""),
        request_trace_sha256=str(result.get("request_trace_sha256") or ""),
        main_candidates=int(result.get("main_candidates") or 0),
        generations_completed=int(result.get("generations_completed") or 0),
        best_job_id=str(result.get("best_job_id") or ""),
        job_ids=tuple(str(j) for j in (result.get("job_ids") or [])),
        prediction_artifacts={
            str(k): str(v) for k, v in dict(result.get("prediction_artifacts") or {}).items()
        },
        operator_trace_sha256=str(result.get("operator_trace_sha256") or ""),
        worker_cleanup_verified=result.get("worker_cleanup_verified") is True,
        operator_counts={str(k): int(v) for k, v in dict(result.get("operator_counts") or {}).items()},
        termination=str(result.get("termination") or "unknown"),
        created_at=float(result.get("finished_at") or 0.0),
    )


def _prior_round_committed(repo_root: Path) -> dict[str, int]:
    """Committed totals of the ORIGINAL (immutable) round-1 ledger, for
    aggregate prior+new cost reporting in authorized retry rounds.

    Read-only by construction (plain file read, no ledger open): the original
    bytes never change. A retry must account for the prior cost BEFORE
    spending anything new, so unknown/unfinished prior usage (open
    reservations) is a refusal, never silently dropped from the aggregate.
    """
    path = repo_root / ac.RUNTIME_DIR / ac.LEDGER_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AcceptanceError("prior_round_ledger_unreadable") from exc
    committed = data.get("committed") if isinstance(data, dict) else None
    if not isinstance(committed, dict):
        raise AcceptanceError("prior_round_ledger_malformed")
    if data.get("stopped") is None:
        # An active/nonterminal prior ledger can still spend; aggregate cost
        # accounting requires a terminal, fully settled prior round.
        raise AcceptanceError("prior_round_ledger_active")
    reservations = data.get("reservations")
    if not isinstance(reservations, list):
        raise AcceptanceError("prior_round_ledger_malformed")
    if reservations:
        raise AcceptanceError("prior_round_open_reservations")
    totals: dict[str, int] = {}
    for key in ("requests", "tokens"):
        value = committed.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AcceptanceError("prior_round_ledger_malformed")
        totals[key] = value
    return totals


def _check_remaining_main_budget(ledger_path: Path, upcoming_fresh_runs: int) -> None:
    """Lower bound BEFORE starting any request of the next child run.

    The minimum is MAIN generations only (3x2 per inner run, never lowered):
    ``MIN_MAIN_CANDIDATES * upcoming_fresh_runs`` requests must remain, or
    the acceptance stops with ``remaining_budget_insufficient`` instead of
    wasting the remaining calls on a run that cannot complete."""
    with BudgetLedger.open(ledger_path) as ledger:
        snapshot = ledger.snapshot()
    stopped = snapshot["stopped"]
    if stopped is not None:
        raise AcceptanceError(
            f"acceptance_ledger_stopped:{stopped.get('reason')}"
        )
    needed = MIN_MAIN_CANDIDATES * upcoming_fresh_runs
    remaining = snapshot["remaining"]["requests"]
    if remaining < needed:
        raise AcceptanceError(
            f"remaining_budget_insufficient:need_main={needed},remaining={remaining}"
        )


def _limits_within_original_pins(limits: BudgetLimits) -> None:
    """No caller may expand the original acceptance pins, for ANY round."""
    pins = ac.acceptance_budget_limits()
    if (
        limits.max_tokens > pins.max_tokens
        or limits.max_requests > pins.max_requests
        or limits.max_elapsed_seconds > pins.max_elapsed_seconds
    ):
        raise AcceptanceError("budget_limits_exceed_acceptance_pins")


def run_acceptance(
    repo_root: Path,
    runtime_dir: Path,
    *,
    algo_factory: Callable[[], Any] | None = None,
    child_runner: Callable[..., dict[str, Any]] | None = None,
    summary_out: Path | None = None,
    budget_limits: BudgetLimits | None = None,
    round_id: str = ac.DEFAULT_ROUND_ID,
) -> dict[str, Any]:
    """The one bounded acceptance. Caller (live mode) has already passed the
    review gate; this function opens/creates the persistent ledger and never
    resets it. The runtime directory is FIXED (round1: the one acceptance
    location; retry rounds: the derived directory of a supervisor-authorized
    round identity — the authorization is verified HERE, before any new
    ledger exists, never implicitly). The preflight vector file is an
    optional cross-check only. Any failure checkpoints and propagates (STOP
    semantics)."""
    repo_root = repo_root.resolve()
    limits = budget_limits or ac.acceptance_budget_limits()
    _limits_within_original_pins(limits)
    if not runtime_dir.is_absolute():
        runtime_dir = repo_root / runtime_dir
    expected_runtime = repo_root / ac.round_runtime_dir(round_id)
    if Path(os.path.abspath(runtime_dir)) != Path(os.path.abspath(expected_runtime)):
        # The one fixed acceptance ledger/run location (or the derived
        # authorized round location); alternates would create a fresh budget.
        raise AcceptanceError("acceptance_runtime_dir_fixed")
    runtime_dir = Path(os.path.abspath(runtime_dir))
    if runtime_dir.resolve() != runtime_dir:
        # A symlinked/aliased runtime could silently modify the immutable
        # original tree; refuse before touching it.
        raise AcceptanceError("acceptance_runtime_dir_symlink")
    if round_id != ac.DEFAULT_ROUND_ID:
        # A retry round runs ONLY under the supervisor-written authorization
        # bound to this commit and this exact round identity; caps come from
        # the authorization and can never exceed the original pins.
        auth_round, auth_limits = ac.check_round_authorization(repo_root)
        if auth_round != round_id:
            raise AcceptanceError("round_authorization_round_mismatch")
        _limits_within_original_pins(auth_limits)
        if budget_limits is not None and budget_limits != auth_limits:
            raise AcceptanceError("round_authorization_caps_mismatch")
        limits = auth_limits
        # The prior (immutable) cost must be fully accounted BEFORE anything
        # new is spent; missing or unresolved prior usage is a refusal.
        prior_committed = _prior_round_committed(repo_root)
    else:
        prior_committed = None
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = runtime_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = runtime_dir / ac.LEDGER_FILENAME
    algo_factory = algo_factory or (lambda: _NativeOuterAlgorithm(repo_root))
    child_runner = child_runner or _default_child_runner
    runner_commit = ac.git_head_commit(repo_root)

    with BudgetLedger.open(ledger_path, limits=limits) as ledger:
        start_snapshot = ledger.snapshot()
    if start_snapshot["stopped"] is not None:
        # Default live mode must REFUSE the old stopped ledger (no
        # resume/reset). This refusal happens BEFORE the checkpointing
        # block, so prior checkpoint/evidence bytes stay untouched.
        raise AcceptanceError(
            f"acceptance_ledger_stopped:{start_snapshot['stopped'].get('reason')}"
        )

    records: list[RunRecord] = []
    job_counts_by_run: dict[str, Any] = {}
    try:
        # Native derivation is the source of truth; the recorded preflight
        # file is only an optional cross-check.
        algo = algo_factory()
        vectors = algo.ask()
        if len(vectors) != ac.POP_SIZE:
            raise AcceptanceError("outer_ask_population_mismatch")
        preflight_path = repo_root / ac.PREFLIGHT_VECTORS_FILE
        if preflight_path.exists():
            try:
                recorded = json.loads(preflight_path.read_text(encoding="utf-8"))
                expected_vectors = recorded["vectors"]
            except (OSError, ValueError, KeyError) as exc:
                raise AcceptanceError("outer_ask_preflight_unavailable") from exc
            if vectors != expected_vectors:
                raise AcceptanceError("outer_ask_mismatch")

        index = RunIndex(runtime_dir / ac.RUN_INDEX_FILENAME)

        def run_one(position: int) -> RunRecord:
            decoded = decode_retained(vectors[position])
            run_id = f"cfg{position}"
            run_dir_rel = (runtime_dir / "runs" / run_id).relative_to(repo_root).as_posix()
            ledger_rel = ledger_path.relative_to(repo_root).as_posix()
            spec = InnerRunSpec(
                run_id=run_id,
                run_dir=run_dir_rel,
                decoded=decoded,
                ledger_path=ledger_rel,
                runner_commit=runner_commit,
                runtime_dir=runtime_dir.relative_to(repo_root).as_posix(),
                future_main_requests=MIN_MAIN_CANDIDATES * (ac.POP_SIZE - position - 1),
            )
            spec.validate()
            actual_run_dir = runtime_dir / "runs" / run_id
            actual_run_dir.mkdir(parents=True, exist_ok=True)
            spec_path = actual_run_dir / "run-spec.json"
            atomic_write_json(spec_path, spec.to_dict())
            result = child_runner(spec, spec_path, repo_root=repo_root)
            # Bind the on-disk result bytes; the returned payload must be
            # exactly the persisted file (fail closed on divergence).
            result_file = actual_run_dir / "run-result.json"
            try:
                result_bytes = result_file.read_bytes()
                persisted = json.loads(result_bytes.decode("utf-8"))
            except (OSError, ValueError) as exc:
                raise AcceptanceError("inner_run_result_missing") from exc
            returned = {k: v for k, v in result.items() if k != "_run_result_sha256"}
            if persisted != returned:
                raise AcceptanceError("inner_run_result_diverged")
            job_counts_by_run[run_id] = persisted.get("job_counts") or {}
            record = _record_from_result(
                result,
                spec,
                run_result_sha256=hashlib.sha256(result_bytes).hexdigest(),
                repo_root=repo_root,
            )
            if record.status == "completed":
                index.put(record)  # validated: verified complete runs only
            return record

        # 1) standalone inner run on the first decoded config.
        _check_remaining_main_budget(ledger_path, ac.POP_SIZE)
        standalone = run_one(0)
        records.append(standalone)
        if standalone.status != "completed":
            raise AcceptanceError("standalone_run_incomplete")

        # 2) native outer loop: re-initialize the same seeded algorithm, ask
        #    (identical vectors), reuse the standalone run as the first
        #    verified evaluation, evaluate the rest, then one explicit tell.
        outer = algo_factory()
        outer_vectors = outer.ask()
        if outer_vectors != vectors:
            raise AcceptanceError("outer_ask_not_reproducible")
        fitness: list[float] = []
        cache_events: list[dict[str, Any]] = []
        for position in range(ac.POP_SIZE):
            decoded = decode_retained(vectors[position])
            reference = records[0].identity
            identity = RunIdentity(
                seed=ac.SEED,
                model=ac.MODEL_ID,
                task_id=ac.TASK_ID,
                metric=ac.METRIC,
                policy_version=reference.policy_version,
                config_hash=config_identity_hash(decoded),
                runner_commit=runner_commit,
                policy_sha256=reference.policy_sha256,
                data_identity=reference.data_identity,
                registry_identity=reference.registry_identity,
            )
            if position == 0:
                # No child is active here: a short ledger lock makes the
                # zero-incremental-cost cache observation explicit.
                with BudgetLedger.open(ledger_path) as cache_ledger:
                    cache_before = cache_ledger.snapshot()
                    record = index.get(identity)
                    cache_after = cache_ledger.snapshot()
                if record is None:
                    raise AcceptanceError("first_run_not_indexed")
                if cache_after["committed"] != cache_before["committed"]:
                    raise AcceptanceError("cache_hit_incurred_model_cost")
                cache_events.append({
                    "outer_position": position,
                    "origin_run": record.run_dir,
                    "identity_key": identity.key(),
                    "run_result_sha256": record.run_result_sha256,
                    "incremental_model_cost": {"requests": 0, "tokens": 0},
                    "verified_committed_before": cache_before["committed"],
                    "verified_committed_after": cache_after["committed"],
                })
            else:
                _check_remaining_main_budget(ledger_path, ac.POP_SIZE - position)
                record = run_one(position)
                records.append(record)
                if record.status != "completed":
                    # A failed child run can never form an accepted outer
                    # tell: stop BEFORE launching the remaining child runs
                    # (the checkpoint preserves the incomplete state).
                    raise AcceptanceError(f"outer_run_incomplete:cfg{position}")
            accuracy = record.score if record.status == "completed" else None
            fitness.append(-accuracy if accuracy is not None else 0.0)
        outer.tell(fitness)
        outer_steps = 1  # native tell does not increment its count field

        with BudgetLedger.open(ledger_path) as ledger:
            end_snapshot = ledger.snapshot()
        coverage = {"draft": 0, "improve": 0, "debug": 0, "crossover": 0}
        for record in records:
            for name in coverage:
                coverage[name] += int(record.operator_counts.get(name, 0))
        all_completed = len(records) == ac.POP_SIZE and all(
            record.status == "completed" for record in records
        )
        # The ledger must still be healthy and within the original deadline
        # at completion; a stopped or exhausted ledger caps the verdict.
        budget_healthy = (
            end_snapshot["stopped"] is None
            and end_snapshot["remaining"]["elapsed_seconds"] > 0
        )
        verdict = (
            "complete"
            if all_completed
            and budget_healthy
            and coverage["improve"] >= 1
            and coverage["crossover"] >= 1
            else "incomplete"
        )
        jobs_total: dict[str, int] = {}
        for counts in job_counts_by_run.values():
            if isinstance(counts, Mapping):
                for key, value in counts.items():
                    if isinstance(value, int) and not isinstance(value, bool):
                        jobs_total[key] = jobs_total.get(key, 0) + value
        summary = {
            "schema": SUMMARY_SCHEMA,
            "verdict": verdict,
            "runner_commit": runner_commit,
            "round_id": round_id,
            "budget": {
                "limits": {
                    "max_tokens": limits.max_tokens,
                    "max_requests": limits.max_requests,
                    "max_elapsed_seconds": limits.max_elapsed_seconds,
                },
                "started_at": start_snapshot["started_at"],
                "start_committed": start_snapshot["committed"],
                "end_committed": end_snapshot["committed"],
                "end_remaining": end_snapshot["remaining"],
                "stopped": end_snapshot["stopped"],
                "healthy_at_completion": budget_healthy,
            },
            "outer": {
                "algorithm": "istratde",
                "seed": ac.SEED,
                "pop_size": ac.POP_SIZE,
                "outer_steps_completed": outer_steps,
                "fitness_mapping": "fitness=-accuracy (native minimize)",
                "fitness": fitness,
            },
            "runs": [
                {
                    "identity": record.identity.to_dict(),
                    "status": record.status,
                    "score": record.score,
                    "metric": ac.METRIC,
                    "metric_direction": "maximize",
                    "job_ids": sorted(record.job_ids),
                    "job_counts": job_counts_by_run.get(f"cfg{run_position}", {}),
                    "operator_counts": dict(record.operator_counts),
                    "worker_cleanup_verified": record.worker_cleanup_verified,
                    "termination": record.termination,
                }
                for run_position, record in enumerate(records)
            ],
            "jobs_total": jobs_total,
            "cache_events": cache_events,
            # "fresh" child invocations are NOT model-active or completed
            # runs: the three are counted separately.
            "inner_evaluations": {
                "requested": 1 + ac.POP_SIZE,
                "fresh_child_invocations": len(records),
                "model_active_runs": sum(
                    1 for record in records
                    if sum(int(v) for v in record.operator_counts.values()) > 0
                ),
                "completed_runs": sum(
                    1 for record in records if record.status == "completed"
                ),
                "cached": len(cache_events),
            },
            "operator_coverage": coverage,
            "no_superiority_claim": True,
        }
        if round_id != ac.DEFAULT_ROUND_ID:
            # Aggregate prior+new cost is preserved across authorized rounds;
            # the original ledger stays the immutable prior-cost source (it
            # was validated before any new request was allowed).
            assert prior_committed is not None
            summary["aggregate_cost"] = {
                "prior_round_committed": prior_committed,
                "this_round_committed": end_snapshot["committed"],
                "total_committed": {
                    "requests": prior_committed["requests"] + end_snapshot["committed"]["requests"],
                    "tokens": prior_committed["tokens"] + end_snapshot["committed"]["tokens"],
                },
            }
        atomic_write_json(runtime_dir / ac.SUMMARY_FILENAME, summary)
        if summary_out is not None:
            atomic_write_json(summary_out, summary)
        return summary
    except BaseException as exc:
        # Any abort (budget, acceptance error, child timeout, interrupt)
        # stops the ledger and persists a checkpoint before propagating.
        try:
            with BudgetLedger.open(ledger_path) as ledger:
                try:
                    ledger.stop(f"acceptance_abort:{type(exc).__name__}")
                except BudgetError:
                    pass  # already stopped is the desired end state
                snapshot = ledger.snapshot()
        except BudgetError:
            snapshot = {"unavailable": True}
        emergency_cleanup: list[dict[str, Any]] = []
        for cleanup_file in sorted(runs_dir.glob("*/emergency-cleanup.json")):
            try:
                emergency_cleanup.append(json.loads(cleanup_file.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                emergency_cleanup.append({"run_dir": cleanup_file.parent.name,
                                          "cleanup_verified": False,
                                          "reason": "cleanup_evidence_unreadable"})
        checkpoint: dict[str, Any] = {
            "schema": "us009-acceptance-checkpoint.v1",
            "reason": f"{type(exc).__name__}: {exc}",
            "runs": [
                {"identity": r.identity.to_dict(), "status": r.status}
                for r in records
            ],
            # Failed records are never presented as completed runs.
            "completed_runs": [
                r.identity.to_dict() for r in records if r.status == "completed"
            ],
            "emergency_cleanup": emergency_cleanup,
            "budget": snapshot,
            "checkpointed_at": time.time(),
        }
        if round_id != ac.DEFAULT_ROUND_ID and prior_committed is not None:
            # Preserve prior+new KNOWN committed cost (and any unresolved new
            # reservations) on failure too — unknown cost is never coerced
            # to zero.
            aggregate: dict[str, Any] = {"prior_round_committed": prior_committed}
            if isinstance(snapshot, dict) and isinstance(snapshot.get("committed"), dict):
                aggregate["this_round_committed"] = snapshot["committed"]
                aggregate["this_round_open_reservations"] = snapshot.get("open_reservations")
                aggregate["total_committed"] = {
                    "requests": prior_committed["requests"] + snapshot["committed"]["requests"],
                    "tokens": prior_committed["tokens"] + snapshot["committed"]["tokens"],
                }
            else:
                aggregate["this_round_unavailable"] = True
            checkpoint["aggregate_cost"] = aggregate
        atomic_write_json(runtime_dir / ac.CHECKPOINT_FILENAME, checkpoint)
        raise


def live(repo_root: Path, *, summary_out: Path | None = None) -> int:
    """Live entrypoint: review gate FIRST, then offline checks, then the run.

    Without a supervisor-written round authorization the target is always the
    one fixed default location (which refuses the old stopped ledger). With a
    valid authorization bound to this commit, the derived retry-round
    directory and its exact caps are used instead — never an automatic new
    budget."""
    check_runner_review(repo_root)
    round_auth = ac.round_authorization_if_present(repo_root)
    checks = run_preflight_checks(repo_root, include_ledger_absence=False)
    failed = {name: c["reason"] for name, c in checks.items() if not c["ok"]}
    if failed:
        print(json.dumps({"refused": "preflight_failed", "failed": failed}, sort_keys=True))
        return 2
    try:
        if round_auth is None:
            summary = run_acceptance(repo_root, ac.RUNTIME_DIR, summary_out=summary_out)
        else:
            round_id, round_limits = round_auth
            summary = run_acceptance(
                repo_root,
                ac.round_runtime_dir(round_id),
                summary_out=summary_out,
                budget_limits=round_limits,
                round_id=round_id,
            )
    except (AcceptanceError, BudgetError, RunIndexError) as exc:
        print(json.dumps({"stopped": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2
    print(json.dumps({"verdict": summary["verdict"]}, sort_keys=True))
    return 0 if summary["verdict"] == "complete" else 3


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research.search.acceptance")
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="offline acceptance preflight (no ledger/auth/model)")
    pre.add_argument("--repo-root", type=Path, default=Path.cwd())
    pre.add_argument("--output", type=Path, default=None)
    live_parser = sub.add_parser("live", help="the one real bounded acceptance (gated)")
    live_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    live_parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.command == "preflight":
            return preflight(repo_root, output=args.output)
        if args.command == "live":
            return live(repo_root, summary_out=args.summary_out)
    except (RunnerReviewRefused, RoundAuthorizationRefused) as exc:
        print(json.dumps({"refused": str(exc)}, sort_keys=True))
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
