"""US-009 inner Evo acceptance child (runs under OpenMLE-Evo/.venv).

One invocation = ONE inner Evo run of one outer config vector against the
isolated 6581 stack. The child:

1. enforces the supervisor review gate BEFORE touching ledger/auth/model;
2. opens the EXISTING shared acceptance ledger (never creates/resets it);
3. assembles the real Evolutionary solver from the vendored dojo YAML
   operator/memory configs with the US-007 validation pins plus the decoded
   retained search dims (credentials stay a placeholder in every config);
4. refreshes the managed Kimi access token in memory before EVERY operator
   call (``prepare_operator`` hook) — tokens never touch disk artifacts;
5. evaluates candidates only through SandboxAcceptanceTask (real sandbox,
   trusted external evaluator), with one budget reservation per provider
   attempt via run_guarded_search;
6. persists a redacted run-result.json evidence record.

Heavy third-party imports (yaml/dojo) are function-local so the spec/result
data classes stay importable in the plain research venv for offline tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.contracts.budget import (
    BudgetError,
    BudgetLedger,
    CancellationToken,
    Cancelled,
)
from research.contracts.persist import atomic_write_json
from research.contracts.results import hash_config
from research.contracts.source_gate import load_policy
from research.search import acceptance_config as ac
from research.search.acceptance_config import (
    RunnerReviewRefused,
    check_runner_review,
)
from research.search.budget_control import MainBudgetControl
from research.search.credentials import (
    CredentialError,
    KimiAuthHelper,
    ManagedKimiCredentials,
    assert_redacted,
)
from research.search.generation_outcome import (
    build_outcome_record,
    classify_extraction_failure,
    feedback_for_failure,
    strip_thinking,
)
from research.search.sandbox_task import SandboxAcceptanceTask
from research.search.search_vector import RETAINED_DIMS, safety_gate
from research.search.validation_config import InnerEvoValidationConfig

INNER_SPEC_SCHEMA = "acceptance-inner-spec.v1"
INNER_RESULT_SCHEMA = "acceptance-inner-result.v1"

_RETAINED_NAMES = tuple(dim.name for dim in RETAINED_DIMS)

#: Real dojo operator YAMLs (relative to the vendored solver configs root).
#: experience.enabled pins the *_experience operator variants; analyze comes
#: from the aide operator set (same as the native single_task_runner).
_SOLVER_CONFIGS = Path("OpenMLE-Evo") / "third_party" / "aira-evo" / "src" / "dojo" / "configs" / "solver"
_OPERATOR_YAMLS = {
    "draft": "mlebench/aira_operators/draft.yaml",
    "improve": "mlebench/aira_operators/improve_experience.yaml",
    "debug": "mlebench/aira_operators/debug_experience.yaml",
    "crossover": "mlebench/aira_operators/crossover_experience.yaml",
    "analyze": "mlebench/aide_operators/analyze.yaml",
}
_SOLVER_DEFAULTS_YAML = "mlebench/evo.yaml"

#: Acceptance sampling pins (managed k3 profile). The builder, the preflight
#: and the transport guard all enforce these exact values.
PROFILE_TEMPERATURE = 1.0
PROFILE_TOP_P = 0.95

TASK_DESCRIPTION = (
    "hello_synth (synthetic acceptance task). A public training CSV and a "
    "public test CSV are provided. Train a classifier predicting the binary "
    "label and write predictions for every test row. Scored by accuracy "
    "(higher is better) against a private answer key that is never visible."
)
DATA_DESCRIPTION = (
    "The environment variable DATA_DIR points to a read-only public data "
    "directory containing train.csv (columns: id,f1,f2,label) and test.csv "
    "(columns: id,f1,f2). No other data files exist or may be accessed."
)
PUBLIC_USER_PROMPT = (
    "Write ONE complete, concise Python program (at most about 120 lines) "
    "that reads os.environ['DATA_DIR']/train.csv and test.csv, fits a single "
    "reasonable classifier (numpy/pandas/scikit-learn are installed), and "
    "writes submission.csv in the current working directory with header "
    "'id,label' and exactly one row per test id, in test order, with label 0 "
    "or 1. Respond with exactly one complete ```python code block and nothing "
    "else: no essay, no explanation before or after the code, no complex "
    "ensemble or multi-stage pipeline, and no partial or truncated code — a "
    "short complete program is required, an incomplete one is discarded. Do "
    "not access any path outside DATA_DIR and the current working directory; "
    "do not use the network."
)


class AcceptanceError(RuntimeError):
    """Acceptance orchestration failure; the message is machine-readable."""


def config_identity_hash(decoded: Mapping[str, Any]) -> str:
    """Canonical hash of one decoded config plus the fixed inner-run pins."""
    pins = InnerEvoValidationConfig()
    return hash_config({
        "decoded": {name: decoded[name] for name in _RETAINED_NAMES},
        "pins": {
            "num_generations": pins.inner_num_generations,
            "individuals_per_generation": pins.individuals_per_generation,
            "step_limit": pins.step_limit,
            "num_islands": pins.num_islands,
            "num_generations_till_crossover": pins.num_generations_till_crossover,
            "experience_enabled": pins.experience_enabled,
            "prompt_memory_enabled": pins.prompt_memory_enabled,
            "stream": pins.stream_enabled,
            "max_output_tokens": pins.max_output_tokens,
            "reasoning_effort": pins.reasoning_effort,
            # litellm forwards reasoning_effort for openai/* only via this
            # per-request allowlist (drop_params stays False) — recorded so
            # the compatibility choice is visible in the config identity.
            "allowed_openai_params": ["reasoning_effort"],
            "operator_generation_attempts": 1,
        },
    })


@dataclass(frozen=True)
class InnerRunSpec:
    """What the parent asks the child to execute (plain JSON on disk)."""

    run_id: str
    run_dir: str  # repo-relative, under the acceptance runtime dir
    decoded: Mapping[str, Any]
    ledger_path: str  # repo-relative; must already exist (parent creates)
    runner_commit: str
    resource_type: str = ac.RESOURCE_TYPE
    job_timeout_seconds: int = ac.JOB_TIMEOUT_SECONDS
    runtime_dir: str = str(ac.RUNTIME_DIR)  # default round; retry rounds need auth
    future_main_requests: int = 0  # main calls reserved for later outer configs

    def validate(self) -> None:
        if not self.run_id or not all(c.isalnum() or c in "-_" for c in self.run_id):
            raise AcceptanceError("spec_run_id_invalid")
        if set(self.decoded.keys()) != set(_RETAINED_NAMES):
            raise AcceptanceError("spec_decoded_dims_mismatch")
        safety_gate(dict(self.decoded))
        for rel in (self.run_dir, self.ledger_path, self.runtime_dir):
            path = Path(rel)
            if path.is_absolute() or ".." in path.parts:
                raise AcceptanceError("spec_path_not_relative")
        runtime = Path(self.runtime_dir)
        if not ac.is_valid_round_runtime_dir(runtime):
            # Only the one fixed acceptance location or a well-formed
            # retry-round directory — never an arbitrary fresh runtime.
            raise AcceptanceError("spec_runtime_dir_invalid")
        run_dir = Path(self.run_dir)
        if not run_dir.is_relative_to(runtime / "runs"):
            raise AcceptanceError("spec_run_dir_outside_runtime")
        if run_dir.name != self.run_id:
            raise AcceptanceError("spec_run_dir_outside_runtime")
        if Path(self.ledger_path) != runtime / ac.LEDGER_FILENAME:
            raise AcceptanceError("spec_ledger_path_mismatch")
        if not self.runner_commit:
            raise AcceptanceError("spec_runner_commit_missing")
        if self.resource_type not in ("cpu", "gpu"):
            raise AcceptanceError("spec_resource_type_invalid")
        if isinstance(self.job_timeout_seconds, bool) or not 30 <= int(self.job_timeout_seconds) <= 1800:
            raise AcceptanceError("spec_job_timeout_invalid")
        if (
            isinstance(self.future_main_requests, bool)
            or not isinstance(self.future_main_requests, int)
            or not 0 <= self.future_main_requests <= 24
        ):
            raise AcceptanceError("spec_future_main_requests_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INNER_SPEC_SCHEMA,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "decoded": dict(self.decoded),
            "ledger_path": self.ledger_path,
            "runner_commit": self.runner_commit,
            "resource_type": self.resource_type,
            "job_timeout_seconds": self.job_timeout_seconds,
            "runtime_dir": self.runtime_dir,
            "future_main_requests": self.future_main_requests,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "InnerRunSpec":
        if not isinstance(data, Mapping) or data.get("schema") != INNER_SPEC_SCHEMA:
            raise AcceptanceError("spec_schema_mismatch")
        try:
            spec = InnerRunSpec(
                run_id=str(data["run_id"]),
                run_dir=str(data["run_dir"]),
                decoded=dict(data["decoded"]),
                ledger_path=str(data["ledger_path"]),
                runner_commit=str(data["runner_commit"]),
                resource_type=str(data.get("resource_type", ac.RESOURCE_TYPE)),
                job_timeout_seconds=int(data.get("job_timeout_seconds", ac.JOB_TIMEOUT_SECONDS)),
                runtime_dir=str(data.get("runtime_dir", str(ac.RUNTIME_DIR))),
                future_main_requests=data.get("future_main_requests", 0),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AcceptanceError("spec_malformed") from exc
        spec.validate()
        return spec

    @staticmethod
    def from_json_file(path: Path) -> "InnerRunSpec":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AcceptanceError("spec_unreadable") from exc
        return InnerRunSpec.from_dict(raw)


def load_sandbox_api_key(auth_file: Path) -> str:
    """Read the sandbox key from the private auth file; never printed/logged."""
    try:
        for line in Path(auth_file).read_text(encoding="utf-8").splitlines():
            if line.startswith("SANDBOX_API_KEYS="):
                key = line.split("=", 1)[1].split(",")[0].strip()
                if key:
                    return key
    except OSError as exc:
        raise AcceptanceError("sandbox_auth_unavailable") from exc
    raise AcceptanceError("sandbox_auth_unavailable")


def operator_clients(solver: Any) -> list[Any]:
    """Every operator's LLM client (for in-memory token refresh)."""
    clients = []
    for name in ("draft", "improve", "debug", "crossover", "analyze", "rich_memory_summary"):
        operator = getattr(solver, name + "_fn", None)
        if operator is None:
            continue
        args = getattr(operator, "args", ())
        llm = args[0] if args else None
        client = getattr(llm, "client", None)
        if client is not None:
            clients.append(client)
    return clients


def make_prepare_operator(
    credentials: ManagedKimiCredentials,
    clients: Sequence[Any],
    secrets_sink: list[str],
) -> Any:
    """Refresh the managed token in memory before EVERY operator attempt.

    The token is read fresh from the credential file (short OAuth TTL) and
    assigned only to the live client objects; configs, journals and traces
    keep the ``runtime`` placeholder. Fail closed when no valid token exists.
    """

    def prepare_operator() -> None:
        token = credentials.access_token()
        secrets_sink.append(token)
        for client in clients:
            client.api_key = token

    return prepare_operator


def _atomic_write_text(path: Path, text: str) -> None:
    """Durably persist exact bytes (tmp + replace), never a partial file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def acceptance_request_profile() -> dict[str, Any]:
    """The exact transport kwargs every actual operator request must carry.

    Mirrors the credential-free litellm profile enforced by
    build_acceptance_solver; the budget guard refuses any off-profile request
    BEFORE reserving budget.
    """
    pins = InnerEvoValidationConfig()
    return {
        # The transport carries the provider-prefixed id (litellm profile:
        # openai/k3); any other model is refused before reservation.
        "model": f"openai/{ac.MODEL_ID}",
        "stream": True,
        "max_tokens": pins.max_output_tokens,
        "temperature": PROFILE_TEMPERATURE,
        "top_p": PROFILE_TOP_P,
        "reasoning_effort": pins.reasoning_effort,
    }


def check_cooperative_stop(stop_file: Path) -> str | None:
    """Run-scoped cooperative stop request (a touched file next to the run
    artifacts). Checked ONLY at safe boundaries — never mid-request and never
    a host-wide process action."""
    try:
        if stop_file.is_file():
            return "stop_requested_file"
    except OSError:
        return "stop_check_unreadable"
    return None


def make_generation_outcome_sink(
    *,
    run_dir: Path,
    run_id: str,
    secrets: Sequence[str],
    stop_file: Path,
    solver: Any,
) -> Any:
    """Persist every generation's source bytes and precise outcome BEFORE the
    operator trace and any candidate evaluation.

    For each provider attempt (each separately budgeted) this writes, in
    order: the raw visible response text, the extracted program (when one
    exists), and one generation-outcomes.jsonl record binding
    finish_reason, provider token counts (numeric only), visible/extracted
    length+hash and the extraction/truncation classification. A failed
    extraction is never collapsed into an unexplained ``empty_candidate_code``
    and never salvaged into fabricated code. After persisting, the
    run-scoped cooperative stop file is honored: the run stops BETWEEN
    generation and scoring with complete artifacts instead of a
    signal-time gap.
    """
    outcomes_path = run_dir / "generation-outcomes.jsonl"
    generated_dir = run_dir / "generated"
    state = {"seq": 0}

    def sink(operator: str, completion_text: Any, extracted_code: Any, metrics: Any) -> None:
        state["seq"] += 1
        sequence = state["seq"]
        visible = strip_thinking(completion_text if isinstance(completion_text, str) else "")
        extracted = extracted_code if isinstance(extracted_code, str) else ""
        classification = "ok" if extracted.strip() else classify_extraction_failure(visible)
        artifacts: dict[str, str] = {}
        generated_dir.mkdir(exist_ok=True)
        # Raw visible response (never reasoning content), redaction-checked.
        assert_redacted(visible, *secrets)
        response_name = f"{sequence:03d}-{operator}.response.txt"
        _atomic_write_text(generated_dir / response_name, visible)
        artifacts["response"] = f"generated/{response_name}"
        if extracted.strip():
            assert_redacted(extracted, *secrets)
            code_name = f"{sequence:03d}-{operator}.py"
            _atomic_write_text(generated_dir / code_name, extracted)
            artifacts["code"] = f"generated/{code_name}"
        record = build_outcome_record(
            run_id=run_id,
            sequence=sequence,
            operator=operator,
            visible_text=visible,
            extracted_code=extracted,
            classification=classification,
            metrics=metrics,
            artifacts=artifacts,
        )
        assert_redacted(record, *secrets)
        with open(outcomes_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        solver.last_generation_feedback = (
            None if classification == "ok" else feedback_for_failure(record)
        )
        stop_reason = check_cooperative_stop(stop_file)
        if stop_reason is not None:
            raise Cancelled(f"cooperative_stop:{stop_reason}")

    return sink


def job_outcome_counts(jobs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Separate submitted/executed/scored/failed job counts.

    Source/cleanup proof alone never means scored: a job only counts as
    ``scored`` with fully verified evaluator evidence; a failed execution
    keeps its own bucket.
    """
    counts = {
        "submitted": 0,
        "executed_and_cleaned": 0,
        "scored": 0,
        "execution_failed": 0,
        "scoring_failed": 0,
        "admission_denied": 0,
        "timeout": 0,
        "cancelled": 0,
        "unscored_other": 0,
    }
    for job in jobs:
        status = job.get("status")
        if status == "admission_denied":
            counts["admission_denied"] += 1
            continue
        if not job.get("job_id"):
            continue  # never reached the sandbox
        counts["submitted"] += 1
        if (
            job.get("worker_cleanup_verified") is True
            and job.get("execution_never_started") is not True
        ):
            counts["executed_and_cleaned"] += 1
        if job.get("evidence_verified") is True:
            counts["scored"] += 1
        elif status == "cancelled":
            counts["cancelled"] += 1
        elif status == "timeout":
            counts["timeout"] += 1
        elif status == "failed" and job.get("scoring_result") == "scoring_failed":
            # A trusted-scoring rejection is not a candidate code crash.
            counts["scoring_failed"] += 1
        elif status == "failed":
            counts["execution_failed"] += 1
        else:
            counts["unscored_other"] += 1
    return counts


def _generation_summary(outcomes_path: Path) -> dict[str, int]:
    summary: dict[str, int] = {}
    if not outcomes_path.exists():
        return summary
    for line in outcomes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        classification = str(json.loads(line).get("classification") or "unknown")
        summary[classification] = summary.get(classification, 0) + 1
    return summary


def build_acceptance_solver(spec: InnerRunSpec, *, repo_root: Path) -> Any:
    """Assemble the REAL Evolutionary solver from vendored dojo YAML configs.

    Mirrors the native single_task_runner assembly, reduced to the minimal
    guarded loop: real operator prompt templates, memory configs and solver
    dataclasses; api_key stays the placeholder; stream/max_tokens forced into
    every operator's generation kwargs.
    """
    import yaml  # noqa: PLC0415 - Evo venv only

    from dojo.config_dataclasses.client.base import ClientConfig  # noqa: PLC0415
    from dojo.config_dataclasses.llm.generic_llm import GenericLLMConfig  # noqa: PLC0415
    from dojo.config_dataclasses.llm.jinjaprompt import JinjaPromptConfig  # noqa: PLC0415
    from dojo.config_dataclasses.operators.base import OperatorConfig  # noqa: PLC0415
    from dojo.config_dataclasses.operators.memory import MemoryOpConfig  # noqa: PLC0415
    from dojo.config_dataclasses.solver.evo import EvolutionarySolverConfig  # noqa: PLC0415
    from dojo.solvers.evo.evo import Evolutionary  # noqa: PLC0415

    spec.validate()
    pins = InnerEvoValidationConfig()
    decoded = dict(spec.decoded)
    run_dir = repo_root / spec.run_dir

    solver_root = repo_root / _SOLVER_CONFIGS

    def load_yaml(path: Path) -> dict[str, Any]:
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AcceptanceError(f"dojo_config_unreadable:{path.name}") from exc
        if not isinstance(loaded, dict):
            raise AcceptanceError(f"dojo_config_malformed:{path.name}")
        return dict(loaded)

    def strip_target(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in dict(payload).items() if k != "_target_"}

    def prompt_config(payload: Any) -> Any:
        if payload is None:
            return JinjaPromptConfig()
        return JinjaPromptConfig(**strip_target(payload))

    client_cfg = ClientConfig(
        api="litellm",
        model_id=ac.MODEL_ID,
        base_url=ac.MODEL_BASE_URL,
        api_key=ac.API_KEY_PLACEHOLDER,
        use_azure_client=False,
        provider="openai",
    )
    solver_defaults = load_yaml(solver_root / _SOLVER_DEFAULTS_YAML)
    # The credential-free managed-k3 profile pins the actual sampling
    # parameters; operator/solver defaults must not override them.
    profile = load_yaml(repo_root / ac.LITELLM_YAML)
    models = profile.get("model_list") or []
    if len(models) != 1:
        raise AcceptanceError("litellm_model_list_must_have_one_entry")
    profile_params = models[0].get("litellm_params") or {}
    if profile_params.get("model") != f"openai/{ac.MODEL_ID}":
        raise AcceptanceError("litellm_model_mismatch")
    if profile_params.get("api_key") != ac.API_KEY_PLACEHOLDER:
        raise AcceptanceError("litellm_api_key_must_be_placeholder")
    if profile_params.get("base_url") != ac.MODEL_BASE_URL:
        raise AcceptanceError("litellm_base_url_mismatch")
    if profile_params.get("stream") is not True:
        raise AcceptanceError("stream_must_be_enabled")
    if profile_params.get("max_tokens") != pins.max_output_tokens:
        raise AcceptanceError("litellm_max_tokens_mismatch")
    if profile_params.get("reasoning_effort") != pins.reasoning_effort:
        # k3 default is high; disabling thinking silently routes to K2.8 —
        # both are forbidden. The profile must pin low explicitly.
        raise AcceptanceError("litellm_reasoning_effort_mismatch")
    for parameter in ("temperature", "top_p"):
        value = profile_params.get(parameter)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AcceptanceError(f"litellm_{parameter}_missing")
        expected = PROFILE_TEMPERATURE if parameter == "temperature" else PROFILE_TOP_P
        if float(value) != expected:
            raise AcceptanceError(f"litellm_{parameter}_mismatch")

    def build_operator(name: str, relative: str) -> Any:
        operator_yaml = load_yaml(solver_root / "operators" / relative)
        payload = operator_yaml.get(name)
        if not isinstance(payload, dict):
            raise AcceptanceError(f"dojo_operator_missing:{name}")
        llm_payload = strip_target(payload.get("llm") or {})
        generation_kwargs: dict[str, Any] = dict(llm_payload.get("generation_kwargs") or {})
        defaults_kwargs = (
            dict(solver_defaults.get("operators", {}).get(name, {})
                 .get("llm", {}).get("generation_kwargs", {}) or {})
        )
        generation_kwargs.update(defaults_kwargs)
        # Acceptance transport pins (managed k3 hangs on nonstream requests;
        # provider usage is mandatory for the budget guard) and the profile's
        # pinned sampling parameters.
        generation_kwargs["stream"] = True
        generation_kwargs["max_tokens"] = pins.max_output_tokens
        generation_kwargs["reasoning_effort"] = pins.reasoning_effort
        # The installed litellm rejects reasoning_effort for openai/* custom
        # models unless allowlisted per request; drop_params stays False so
        # an unforwardable parameter always fails loudly (never silently
        # dropped, never a hidden fallback).
        generation_kwargs["allowed_openai_params"] = ["reasoning_effort"]
        generation_kwargs["temperature"] = float(profile_params["temperature"])
        generation_kwargs["top_p"] = float(profile_params["top_p"])
        return OperatorConfig(
            llm=GenericLLMConfig(client=client_cfg, generation_kwargs=generation_kwargs),
            system_message_prompt_template=prompt_config(payload.get("system_message_prompt_template")),
            init_user_message_prompt_template=prompt_config(payload.get("init_user_message_prompt_template")),
            user_message_prompt_template=prompt_config(payload.get("user_message_prompt_template")),
        )

    operators = {name: build_operator(name, rel) for name, rel in _OPERATOR_YAMLS.items()}
    memory_cfg = MemoryOpConfig(**strip_target(load_yaml(solver_root / "memory" / "simple_memory.yaml")))
    debug_memory_cfg = MemoryOpConfig(**strip_target(load_yaml(solver_root / "memory" / "debug_memory.yaml")))

    cfg = EvolutionarySolverConfig(
        step_limit=pins.step_limit,
        available_packages=["numpy", "pandas", "scikit-learn"],
        operators=operators,
        memory=memory_cfg,
        debug_memory=debug_memory_cfg,
        exp_name="us009_acceptance",
        execution_timeout=spec.job_timeout_seconds,
        time_limit_secs=3000,
        export_search_results=False,
        checkpoint_path=str(run_dir / "checkpoint"),
        use_test_score=False,
        use_complexity=False,
        max_llm_call_retries=1,  # one generation per operator; fresh auth before each request
        num_islands=pins.num_islands,
        max_island_size=50,
        crossover_prob=float(decoded["crossover_prob"]),
        migration_prob=0.0,
        initial_temp=1.0,
        final_temp=1.0,
        num_generations_till_migration=999,
        num_generations_till_crossover=pins.num_generations_till_crossover,
        few_shot={"improve": 1, "crossover": 2},
        num_generations=pins.inner_num_generations,
        individuals_per_generation=pins.individuals_per_generation,
        max_debug_time=600,
        max_debug_depth=int(decoded["max_debug_depth"]),
        fresh_draft_prob=0.0,
        max_wall_time_secs=0.0,
        data_preview=False,
        experience={
            "enabled": True,
            "parent_selection": {
                "enabled": True,
                "weights": {
                    "score": float(decoded["score_weight"]),
                    "delta": float(decoded["delta_weight"]),
                    "novelty": float(decoded["novelty_weight"]),
                },
            },
            "prompt_memory": {"enabled": False},
        },
        execution_mode="generation",
        async_workers=1,
    )
    task_info = {
        "task_description": TASK_DESCRIPTION,
        "data_description": DATA_DESCRIPTION,
        "public_user_prompt": PUBLIC_USER_PROMPT,
        "lower_is_better": False,
    }
    return Evolutionary(cfg, task_info=task_info)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_prediction_snapshots(jobs: list[dict[str, Any]], run_dir: Path, repo_root: Path) -> dict[str, str]:
    """Copy every verified job's trusted prediction snapshot into the run dir
    (immutable per-run bytes), verifying content against the evaluator's
    recorded prediction_sha256. Fail closed on any mismatch/unreadable —
    evidence integrity is infrastructure, not a candidate outcome."""
    artifacts: dict[str, str] = {}
    storage_root = repo_root / ac.DISPATCHER_STORAGE_ROOT
    for job in jobs:
        evaluation = job.get("evaluation") or {}
        snapshot = evaluation.get("prediction_snapshot")
        job_id = job.get("job_id")
        if not job_id or not snapshot:
            continue
        if not isinstance(snapshot, str) or not snapshot.startswith(ac.CONTAINER_STORAGE_PREFIX):
            raise AcceptanceError("prediction_snapshot_path_invalid")
        relative = snapshot[len(ac.CONTAINER_STORAGE_PREFIX):]
        host_path = (storage_root / relative).resolve()
        if not host_path.is_relative_to(storage_root.resolve()):
            raise AcceptanceError("prediction_snapshot_path_invalid")
        try:
            data = host_path.read_bytes()
        except OSError as exc:
            raise AcceptanceError("prediction_snapshot_unreadable") from exc
        digest = hashlib.sha256(data).hexdigest()
        if digest != evaluation.get("prediction_sha256"):
            raise AcceptanceError("prediction_snapshot_mismatch")
        predictions_dir = run_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        (predictions_dir / f"{job_id}.csv").write_bytes(data)
        artifacts[str(job_id)] = digest
    return artifacts


def _operator_counts(trace_path: Path) -> dict[str, int]:
    counts = {"draft": 0, "improve": 0, "debug": 0, "crossover": 0}
    if not trace_path.exists():
        return counts
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        operator = event.get("operator")
        if operator in counts:
            counts[operator] += 1
    return counts


#: Allowlist of public request parameters recorded in the request trace.
#: Everything else (api_key, extra_headers, default_headers, ...) is dropped.
_TRACE_REQUEST_KEYS = (
    "model",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "temperature",
    "top_p",
    "reasoning_effort",
    "allowed_openai_params",
    "request_timeout",
    "stream_options",
    "num_retries",
    "max_retries",
)


def run_inner(spec_path: Path, *, repo_root: Path) -> int:
    """Execute one inner run; write run-result.json; return process exit code.

    Exit 0: run-result.json written (check its status field). Exit 2:
    refused/failed before any result could be produced (nothing started).
    """
    check_runner_review(repo_root)  # gate BEFORE ledger/auth/model activity
    spec = InnerRunSpec.from_json_file(spec_path)
    if spec.runner_commit != ac.git_head_commit(repo_root):
        raise AcceptanceError("spec_runner_commit_mismatch")
    if Path(spec.runtime_dir) != ac.RUNTIME_DIR:
        # A retry-round spec is only executable under a supervisor-written
        # authorization bound to this commit AND this exact round identity;
        # there is no automatic new directory or budget.
        try:
            round_id, _round_caps = ac.check_round_authorization(repo_root)
        except ac.RoundAuthorizationRefused as exc:
            raise AcceptanceError(f"round_authorization_required:{exc}") from exc
        if ac.round_runtime_dir(round_id) != Path(spec.runtime_dir):
            raise AcceptanceError("round_authorization_round_mismatch")
    ledger_path = repo_root / spec.ledger_path
    if not ledger_path.exists():
        # The child never creates or resets the one acceptance ledger.
        raise AcceptanceError("acceptance_ledger_missing")
    run_dir = repo_root / spec.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LOGGING_DIR", str(run_dir / "logs"))

    # Deferred imports: dojo needs LOGGING_DIR; keep them out of module scope.
    import random  # noqa: PLC0415

    import numpy  # noqa: PLC0415
    from dojo.core.interpreters.base import ExecutionResult  # noqa: PLC0415
    from dojo.solvers.evo.operator_trace import OperatorTraceWriter  # noqa: PLC0415
    from dojo.utils.logger import config_logger  # noqa: PLC0415
    from research.adapters.sandbox_eval_client import SandboxEvalClient  # noqa: PLC0415
    from research.search.budget_transport import TransportGuardConfig  # noqa: PLC0415
    from research.search.run_control import (  # noqa: PLC0415
        controlled_termination_reason,
        run_guarded_search,
    )

    config_logger(None)  # low-resource global logger: no file/wandb handlers

    # Search-side RNG seeding (the RunIdentity claims seed 42): the Evo solver
    # draws operator coin flips and parent selection from Python's random and
    # NumPy. This seeds the SEARCH only; provider generation remains
    # nondeterministic (no unsupported provider seed parameter is sent).
    random.seed(ac.SEED)
    numpy.random.seed(ac.SEED)
    rng_evidence = {
        "python_random_seed": ac.SEED,
        "numpy_random_seed": ac.SEED,
        "outer_jax_seed": ac.SEED,
        "scope": "inner solver branch/selection seeding only; provider generation is nondeterministic",
    }

    secrets: list[str] = []
    sandbox_key = load_sandbox_api_key(repo_root / ac.AUTH_FILE)
    secrets.append(sandbox_key)
    credentials = ManagedKimiCredentials()
    auth_helper = KimiAuthHelper()
    try:
        credentials.attach_helper(auth_helper)

        policy = load_policy()
        request_trace_path = run_dir / "request-trace.jsonl"
        attempt_counter = {"n": 0}

        def attempt_sink(
            reservation: Any, messages: Any, model_kwargs: Any, operator: Any = None
        ) -> None:
            """Link every provider attempt to its ledger reservation and persist
            the PUBLIC request for provenance: run/operator/attempt identity,
            reservation, allowlisted bounded request parameters (never headers or
            credentials). Fail closed: a redaction or write failure aborts the
            attempt before the provider is called (reservation retained + stop).
            """
            attempt_counter["n"] += 1
            record = {
                "run_id": spec.run_id,
                "attempt": attempt_counter["n"],
                "operator": operator or "unknown",
                "reservation_id": reservation.reservation_id,
                "reserved_tokens": reservation.tokens,
                "messages": messages,
                "request": {
                    key: model_kwargs[key]
                    for key in _TRACE_REQUEST_KEYS
                    if isinstance(model_kwargs, Mapping) and key in model_kwargs
                },
            }
            assert_redacted(record, *secrets)
            with open(request_trace_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

        guard_config = TransportGuardConfig(
            **InnerEvoValidationConfig().as_transport_guard_config(),
            attempt_sink=attempt_sink,
            required_request_profile=acceptance_request_profile(),
        )

        error: str | None = None
        best_score: float | None = None
        best_job_id = ""
        start_snapshot: dict[str, Any] = {}
        end_snapshot: dict[str, Any] = {}
        cleanup_verified = False
        with BudgetLedger.open(ledger_path) as ledger:
            start_snapshot = ledger.snapshot()
            if start_snapshot["stopped"] is not None:
                # The child never runs against a stopped ledger (e.g. the
                # old, stopped acceptance ledger): refuse BEFORE any provider
                # call or sandbox submission.
                raise AcceptanceError("acceptance_ledger_stopped")
            pins = InnerEvoValidationConfig()
            main_minimum = (
                pins.inner_num_generations * pins.individuals_per_generation
                + spec.future_main_requests
            )
            if start_snapshot["remaining"]["requests"] < main_minimum:
                # Lower bound: this run plus all FUTURE outer configs need at
                # least their 3x2 main generations; starting with less would
                # waste the remaining calls.
                raise AcceptanceError("remaining_budget_insufficient")
            # Anchor to the ORIGINAL persisted deadline (started_at + persisted
            # limits), never time.time()+remaining, so a restarted child cannot
            # silently extend the acceptance window.
            ledger_deadline = ledger.deadline_epoch
            token = CancellationToken(deadline_epoch=ledger_deadline)
            live_jobs_path = run_dir / "live-jobs.json"
            submission_intents: set[str] = set()

            def persist_live_jobs(_event: Any = None) -> None:
                """Persist submission intents, live job ids and unresolved
                submissions BEFORE polling; the parent uses this for emergency
                cleanup if this child dies."""
                if isinstance(_event, Mapping):
                    if _event.get("event") == "submission_intent":
                        submission_intents.add(str(_event.get("trace_id")))
                    elif _event.get("event") == "job_registered":
                        submission_intents.discard(str(_event.get("trace_id")))
                atomic_write_json(live_jobs_path, {
                    "schema": "acceptance-live-jobs.v1",
                    "run_id": spec.run_id,
                    "submission_intents": sorted(submission_intents),
                    "live_jobs": sorted(client.live_job_ids()),
                    "uncertain_submissions": sorted(client.uncertain_submission_ids()),
                    "updated_at": time.time(),
                })

            client = SandboxEvalClient(
                endpoint=ac.GATEWAY_URL,
                api_key=sandbox_key,
                require_cleanup=True,
                observer=persist_live_jobs,
            )
            persist_live_jobs()  # initial empty state: file must always exist
            task = SandboxAcceptanceTask(
                client,
                token,
                resource_type=spec.resource_type,
                job_timeout_seconds=spec.job_timeout_seconds,
                execution_result_cls=ExecutionResult,
            )
            solver = build_acceptance_solver(spec, repo_root=repo_root)
            trace_path = run_dir / "operator-trace.jsonl"
            solver.operator_tracer = OperatorTraceWriter(trace_path)
            # Generation artifacts + precise outcomes persist BEFORE the
            # operator trace / any evaluation; the run-scoped stop file is
            # honored at safe boundaries (before an operator and right after
            # a settled, persisted generation — never mid-request).
            stop_file = run_dir / "stop-request.json"
            solver.generation_outcome_sink = make_generation_outcome_sink(
                run_dir=run_dir,
                run_id=spec.run_id,
                secrets=secrets,
                stop_file=stop_file,
                solver=solver,
            )
            # Reserve current+future MAIN requests; optional debug attempts
            # proceed only with one extra slack request (debug is an upper
            # bound, never a required minimum).
            solver.budget_control = MainBudgetControl(
                ledger,
                total_main=pins.inner_num_generations * pins.individuals_per_generation,
                future_main=spec.future_main_requests,
                control_log_path=run_dir / "budget-control.jsonl",
                run_id=spec.run_id,
            )
            prepare = make_prepare_operator(credentials, operator_clients(solver), secrets)
            try:
                run_guarded_search(
                    solver,
                    task,
                    {},
                    ledger=ledger,
                    token=token,
                    transport_config=guard_config,
                    cancel_live_jobs=client.cancel_all_live,
                    checkpoint_path=run_dir / "search-control-checkpoint.json",
                    prepare_operator=prepare,
                    cooperative_stop=lambda: check_cooperative_stop(stop_file),
                )
            except BaseException as exc:  # ledger/token already stopped by control
                error = controlled_termination_reason(exc)
            if error is None:
                best_node = solver.journal.get_best_node()
                metric = getattr(best_node, "metric", None)
                value = getattr(metric, "value", None)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    best_score = float(value)
                    info = getattr(metric, "info", None)
                    if isinstance(info, Mapping):
                        best_job_id = str(info.get("job_id") or "")
            control_checkpoint = run_dir / "search-control-checkpoint.json"
            cancellation_ok = True
            if control_checkpoint.exists():
                try:
                    control = json.loads(control_checkpoint.read_text(encoding="utf-8"))
                    cancellation_ok = (
                        control.get("cleanup_error") is None
                        and all(
                            entry.get("cleanup_verified") is True
                            for entry in control.get("cancellation", [])
                        )
                    )
                except ValueError:
                    cancellation_ok = False
            else:
                cancellation_ok = False
            cleanup_verified = (
                cancellation_ok
                and not client.live_job_ids()
                and not client.uncertain_submission_ids()
            )
            end_snapshot = ledger.snapshot()

        # Bind the generated-candidate journal into the run artifacts.
        journal_src = run_dir / "checkpoint" / "journal.jsonl"
        journal_path = run_dir / "journal.jsonl"
        if journal_src.exists():
            journal_path.write_bytes(journal_src.read_bytes())
        counts = _operator_counts(trace_path)
        main_candidates = sum(counts[name] for name in ("draft", "improve", "crossover"))
        generations_completed = int(getattr(solver.state, "current_generation", 0))
        prediction_artifacts = bind_prediction_snapshots(task.jobs, run_dir, repo_root)
        outcomes_path = run_dir / "generation-outcomes.jsonl"
        status = "completed" if error is None and best_score is not None else "failed"
        result = {
            "schema": INNER_RESULT_SCHEMA,
            "run_id": spec.run_id,
            "decoded": dict(spec.decoded),
            "config_hash": config_identity_hash(spec.decoded),
            "policy_version": policy.policy_version,
            "policy_sha256": policy.policy_sha256,
            "runner_commit": spec.runner_commit,
            "status": status,
            "termination": error or "completed",
            "best_score": best_score,
            "best_job_id": best_job_id,
            "metric": ac.METRIC,
            "metric_direction": "maximize",
            "main_candidates": main_candidates,
            "generations_completed": generations_completed,
            "rng": rng_evidence,
            "operator_counts": counts,
            "operator_trace_sha256": (
                _sha256_file(trace_path) if trace_path.exists() else ""
            ),
            "request_trace_sha256": (
                _sha256_file(request_trace_path) if request_trace_path.exists() else ""
            ),
            "journal_sha256": (
                _sha256_file(journal_path) if journal_path.exists() else ""
            ),
            "prediction_artifacts": prediction_artifacts,
            "jobs": task.jobs,
            "job_counts": job_outcome_counts(task.jobs),
            "generation_outcomes_sha256": (
                _sha256_file(outcomes_path) if outcomes_path.exists() else ""
            ),
            "generation_summary": _generation_summary(outcomes_path),
            "job_ids": sorted({j["job_id"] for j in task.jobs if j.get("job_id")}),
            "worker_cleanup_verified": cleanup_verified,
            "model_usage": {
                "requests_delta": (
                    end_snapshot["committed"]["requests"] - start_snapshot["committed"]["requests"]
                ),
                "tokens_delta": (
                    end_snapshot["committed"]["tokens"] - start_snapshot["committed"]["tokens"]
                ),
            },
            "budget": {
                "committed": end_snapshot.get("committed"),
                "remaining": end_snapshot.get("remaining"),
                "stopped": end_snapshot.get("stopped"),
                "open_reservations": end_snapshot.get("open_reservations"),
                "deadline_epoch": ledger_deadline,
            },
            "started_at": start_snapshot.get("started_at"),
            "finished_at": time.time(),
        }
        assert_redacted(result, *secrets)
        atomic_write_json(run_dir / "run-result.json", result)
        return 0
    finally:
        auth_helper.close()  # only this invocation's owned helpers


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research.search.acceptance_inner")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="execute one inner Evo acceptance run")
    run_parser.add_argument("--spec", required=True, type=Path)
    run_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.command == "run":
            return run_inner(args.spec, repo_root=repo_root)
    except RunnerReviewRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except (AcceptanceError, CredentialError, BudgetError) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2  # unreachable: unknown command


if __name__ == "__main__":
    sys.exit(main())
