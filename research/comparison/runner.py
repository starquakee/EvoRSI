"""Comparison inner-search runner (EXP-005): one inner Evo run of one
comparison config against the isolated 6581 stack.

Separate from the US-009 acceptance runner (which is preserved unchanged):
this runner executes ONE inner search bound to a comparison
``InnerRunSpec`` (task, method, search seed, config index, paired inner
seed) with the decoded theta, the pinned K3 profile and the comparison
ledger as the ONLY accounting layer.

One invocation = ONE fresh memory-off inner search: 3 generations x 2 main
candidates, at most 2 debug requests total, real vendored dojo operators,
candidates executed/scored only through the trusted sandbox + external
evaluator (validation split; validation labels never leave the evaluator
tree). The generic ``train.csv``/``test.csv`` program contract is identical
for all three tasks, so the exact winning bytes can refit on
train+validation at the final phase (EXP-006).

Live gate: ``check_comparison_review`` (research/comparison/review.py)
refuses BEFORE any ledger/auth/model/sandbox activity unless the ignored
supervisor review — bound to the exact clean HEAD and the frozen immutable
experiment manifest — is present and re-verifies. Offline integration
tests inject fakes BELOW this entry point; the production entry never
bypasses the gate.
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

from research.comparison.budget import (
    ComparisonBudgetError,
    ComparisonLedger,
)
from research.comparison.control import (
    ComparisonMainBudgetControl,
    run_comparison_search,
)
from research.comparison.data import load_task_bindings
from research.comparison.protocol import (
    COMPARISON_DIMS,
    CellBudget,
    ComparisonSpecError,
    ModelProfile,
    canonical_protocol,
    identity_of,
    theta_gate,
)
from research.comparison.specs import (
    CellSpec,
    InnerRunSpec,
    TaskBinding,
    iter_cells,
)
from research.comparison.task import ComparisonSandboxTask
from research.contracts.budget import CancellationToken
from research.contracts.persist import atomic_write_json
from research.contracts.results import MetricDirection
from research.contracts.source_gate import load_policy
from research.evaluator.service import EvaluatorRegistry
from research.comparison import deploy as ac
from research.comparison.review import (
    LEDGER_FILENAME,
    RUNTIME_DIR,
    ComparisonReviewRefused,
    check_review_acceptance,
)
from research.search.acceptance_inner import (
    job_outcome_counts,
    load_sandbox_api_key,
    make_generation_outcome_sink,
    make_prepare_operator,
    operator_clients,
)
from research.search.credentials import (
    CredentialError,
    KimiAuthHelper,
    ManagedKimiCredentials,
    assert_redacted,
)
from research.search.sandbox_task import _single_file_tree_sha256
from research.search.validation_config import InnerEvoValidationConfig

RUN_SPEC_SCHEMA = "istratde-comparison-run.v1"
RUN_RESULT_SCHEMA = "istratde-comparison-inner-result.v1"
LIVE_JOBS_SCHEMA = "istratde-comparison-live-jobs.v1"

#: Runtime layout (ignored .runtime/, never committed): the one comparison
#: ledger and one controller-private run directory per inner run. The
#: constants are owned by research.comparison.review and re-exported here.
_COMMIT_HEX_LEN = 40
_HEX_DIGITS = frozenset("0123456789abcdef")

#: Public task display names (official UCI datasets; PRD References).
TASK_TITLES = {
    "banknote": "Banknote Authentication (UCI dataset 267)",
    "wdbc": "Breast Cancer Wisconsin Diagnostic (UCI dataset 17)",
    "spambase": "Spambase (UCI dataset 94)",
}

#: The worker data root for one comparison task inside the sandbox
#: (dispatcher validates it against the pinned registry grant leaf and
#: exports DATA_DIR as the pinned public dir).
WORKER_TASK_DATA_ROOT = "/mnt/rsi_data/comparison"


class ComparisonRunnerError(RuntimeError):
    """Comparison runner failure; the message is machine-readable."""


def check_comparison_review(repo_root: Path) -> None:
    """Live gate: the accepted, exactly-bound supervisor review (EXP-007).

    No model call, credential read, ledger creation or sandbox submission
    may happen before the comparison supervisor review — bound to the exact
    clean HEAD and the full immutable experiment manifest — verifies. The
    implementation NEVER writes the review file; while it is absent every
    live attempt refuses with ``comparison_review_absent``.
    """
    check_review_acceptance(repo_root)


def run_dir_name(run: InnerRunSpec) -> str:
    """Filesystem-safe run directory name for one inner run."""
    return run.run_key().replace(":", "-")


@dataclass(frozen=True)
class ComparisonRunSpec:
    """What the controller asks the runner to execute (plain JSON on disk).

    Binds the comparison InnerRunSpec (cell: task binding/method/search
    seed; config index; derived paired inner seed), the decoded theta, the
    runner source commit, the run directory and the EXISTING comparison
    ledger path (the runner never creates or resets the ledger).
    """

    run: InnerRunSpec
    decoded: Mapping[str, Any]
    run_dir: str  # repo-relative, under RUNTIME_DIR/runs
    ledger_path: str  # repo-relative; must already exist
    runner_commit: str
    resource_type: str = "cpu"
    job_timeout_seconds: int = 600

    def validate(self) -> None:
        if not isinstance(self.run, InnerRunSpec):
            raise ComparisonRunnerError("spec_run_invalid")
        if set(self.decoded.keys()) != {dim.name for dim in COMPARISON_DIMS}:
            raise ComparisonRunnerError("spec_decoded_dims_mismatch")
        try:
            theta_gate(dict(self.decoded))
        except ComparisonSpecError as exc:
            raise ComparisonRunnerError(f"spec_theta_invalid:{exc}") from exc
        for rel in (self.run_dir, self.ledger_path):
            path = Path(rel)
            if path.is_absolute() or ".." in path.parts:
                raise ComparisonRunnerError("spec_path_not_relative")
        run_dir = Path(self.run_dir)
        if run_dir.parent != RUNTIME_DIR / "runs":
            raise ComparisonRunnerError("spec_run_dir_outside_runtime")
        if run_dir.name != run_dir_name(self.run):
            raise ComparisonRunnerError("spec_run_dir_name_mismatch")
        if Path(self.ledger_path) != RUNTIME_DIR / LEDGER_FILENAME:
            raise ComparisonRunnerError("spec_ledger_path_mismatch")
        if (
            not isinstance(self.runner_commit, str)
            or len(self.runner_commit) != _COMMIT_HEX_LEN
            or not set(self.runner_commit) <= _HEX_DIGITS
        ):
            raise ComparisonRunnerError("spec_runner_commit_invalid")
        if self.runner_commit != self.run.cell.source_commit:
            # The runner commit IS the cell's bound implementation commit.
            raise ComparisonRunnerError("spec_runner_commit_mismatch")
        # The frozen protocol runs on the one CPU worker slot.
        if self.resource_type != "cpu":
            raise ComparisonRunnerError("spec_resource_type_invalid")
        if (
            isinstance(self.job_timeout_seconds, bool)
            or not isinstance(self.job_timeout_seconds, int)
            or not 30 <= self.job_timeout_seconds <= 1800
        ):
            raise ComparisonRunnerError("spec_job_timeout_invalid")

    def to_dict(self) -> dict[str, Any]:
        cell = self.run.cell
        return {
            "schema": RUN_SPEC_SCHEMA,
            "task_binding": cell.task_binding.to_dict(),
            "method": cell.method,
            "search_seed": cell.search_seed,
            "config_index": self.run.config_index,
            "inner_seed": self.run.inner_seed,
            "decoded": dict(self.decoded),
            "run_dir": self.run_dir,
            "ledger_path": self.ledger_path,
            "runner_commit": self.runner_commit,
            "resource_type": self.resource_type,
            "job_timeout_seconds": self.job_timeout_seconds,
            "run_identity": self.run.identity(),
            "cell_identity": cell.identity(),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "ComparisonRunSpec":
        if not isinstance(data, Mapping) or data.get("schema") != RUN_SPEC_SCHEMA:
            raise ComparisonRunnerError("spec_schema_mismatch")
        try:
            # Strict serialized types: a float stand-in like 11.0 compares
            # equal to the int pin but is a tampered/lossy encoding.
            if type(data["inner_seed"]) is not int:
                raise ComparisonRunnerError("spec_inner_seed_not_an_int")
            if type(data["config_index"]) is not int:
                raise ComparisonRunnerError("spec_config_index_not_an_int")
            binding = TaskBinding(**dict(data["task_binding"]))
            cell = CellSpec(
                task_binding=binding,
                method=str(data["method"]),
                search_seed=data["search_seed"],
                model_profile=ModelProfile(),
                source_commit=str(data["runner_commit"]),
                cell_budget=CellBudget(),
                protocol_identity=canonical_protocol().identity(),
            )
            run = cell.inner_run(data["config_index"])
            if run.inner_seed != data["inner_seed"]:
                raise ComparisonRunnerError("spec_inner_seed_mismatch")
            spec = ComparisonRunSpec(
                run=run,
                decoded=dict(data["decoded"]),
                run_dir=str(data["run_dir"]),
                ledger_path=str(data["ledger_path"]),
                runner_commit=str(data["runner_commit"]),
                resource_type=str(data.get("resource_type", "cpu")),
                job_timeout_seconds=data.get("job_timeout_seconds", 600),
            )
        except ComparisonSpecError:
            raise  # keep the precise fail-closed rule
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ComparisonRunnerError):
                raise
            raise ComparisonRunnerError("spec_malformed") from exc
        if spec.run.identity() != data.get("run_identity"):
            raise ComparisonRunnerError("spec_run_identity_mismatch")
        if spec.run.cell.identity() != data.get("cell_identity"):
            raise ComparisonRunnerError("spec_cell_identity_mismatch")
        spec.validate()
        return spec

    @staticmethod
    def from_json_file(path: Path) -> "ComparisonRunSpec":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ComparisonRunnerError("spec_unreadable") from exc
        return ComparisonRunSpec.from_dict(raw)


def comparison_config_identity(spec: ComparisonRunSpec) -> str:
    """Canonical identity of one decoded config plus the fixed inner pins."""
    pins = InnerEvoValidationConfig()
    profile = ModelProfile()
    return identity_of({
        "run_identity": spec.run.identity(),
        "decoded": {dim.name: spec.decoded[dim.name] for dim in COMPARISON_DIMS},
        "pins": {
            "num_generations": pins.inner_num_generations,
            "individuals_per_generation": pins.individuals_per_generation,
            "step_limit": pins.step_limit,
            "num_islands": pins.num_islands,
            "num_generations_till_crossover": pins.num_generations_till_crossover,
            "experience_enabled": pins.experience_enabled,
            "prompt_memory_enabled": pins.prompt_memory_enabled,
            "operator_generation_attempts": 1,
        },
        "model_profile": profile.to_dict(),
    })


def task_info_for(binding: TaskBinding) -> dict[str, Any]:
    """Generic train.csv/test.csv program contract for one task.

    Identical shape for all three tasks (only the feature count and the
    official dataset title differ), so the exact winning source bytes refit
    on train+validation at the final phase without code changes.
    """
    n_features = binding.features
    feature_cols = ",".join(["id", *[f"f{i}" for i in range(1, n_features + 1)]])
    title = TASK_TITLES[binding.task_id]
    task_description = (
        f"{title} binary classification (trusted comparison task "
        f"{binding.task_id}). A public training CSV and a public test CSV "
        "are provided. Train a classifier predicting the binary label and "
        "write predictions for every test row. Scored by accuracy (higher "
        "is better) against a private answer key that is never visible."
    )
    data_description = (
        "The environment variable DATA_DIR points to a read-only public "
        f"data directory containing train.csv (columns: {feature_cols},label) "
        f"and test.csv (columns: {feature_cols}). No other data files exist "
        "or may be accessed."
    )
    public_user_prompt = (
        "Write ONE complete, concise Python program (at most about 120 "
        "lines) that reads os.environ['DATA_DIR']/train.csv and test.csv, "
        "fits a single reasonable classifier (numpy/pandas/scikit-learn "
        "are installed), and writes submission.csv in the current working "
        "directory with header 'id,label' and exactly one row per test id, "
        "in test order, with label 0 or 1. Respond with exactly one "
        "complete ```python code block and nothing else: no essay, no "
        "explanation before or after the code, no complex ensemble or "
        "multi-stage pipeline, and no partial or truncated code — a short "
        "complete program is required, an incomplete one is discarded. Do "
        "not access any path outside DATA_DIR and the current working "
        "directory; do not use the network."
    )
    return {
        "task_description": task_description,
        "data_description": data_description,
        "public_user_prompt": public_user_prompt,
        "lower_is_better": False,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_task_data_identity(
    binding: TaskBinding,
    *,
    repo_root: Path,
    registry: Any,  # EvaluatorRegistry; duck-typed at this trusted boundary
) -> dict[str, Any]:
    """Bind the run to the EXACT trusted dataset before any model activity.

    Recomputes the worker-visible public file hashes and checks the pinned
    evaluator registry entry (validation split, answer hash, worker grant
    leaf). Any mismatch fails closed. Sealed final/refit data is never
    touched here; validation labels exist only in the evaluator tree.
    """
    task_id = binding.task_id
    public = repo_root / "tasks" / "comparison" / task_id / "data" / "public"
    for name, expected in (
        ("train.csv", binding.train_csv_sha256),
        ("test.csv", binding.validation_features_sha256),
    ):
        path = public / name
        if not path.is_file():
            raise ComparisonRunnerError(f"task_data_missing:{name}")
        if _sha256_file(path) != expected:
            raise ComparisonRunnerError(f"task_data_hash_mismatch:{name}")
    try:
        spec = registry.task(f"comparison_{task_id}")
    except Exception as exc:
        raise ComparisonRunnerError(f"evaluator_task_unregistered:{task_id}") from exc
    if spec.split != "validation":
        raise ComparisonRunnerError("evaluator_split_mismatch")
    if spec.metric != "accuracy":
        raise ComparisonRunnerError("evaluator_metric_mismatch")
    if spec.direction is not MetricDirection.MAXIMIZE:
        raise ComparisonRunnerError("evaluator_direction_mismatch")
    if spec.answer_sha256 != binding.validation_labels_sha256:
        raise ComparisonRunnerError("evaluator_answer_hash_mismatch")
    expected_grant = f"{WORKER_TASK_DATA_ROOT}/{task_id}/data/public"
    if spec.worker_data_public_path != expected_grant:
        raise ComparisonRunnerError("worker_data_grant_mismatch")
    return {
        "registry_task_id": f"comparison_{task_id}",
        "registry_version": registry.registry_version,
        "split": spec.split,
        "worker_data_public_path": spec.worker_data_public_path,
        "train_csv_sha256": binding.train_csv_sha256,
        "validation_features_sha256": binding.validation_features_sha256,
        "validation_labels_sha256": binding.validation_labels_sha256,
        "task_binding_identity": binding.identity(),
    }


def _load_pinned_profile_params(repo_root: Path) -> dict[str, Any]:
    """Load and pin-check the shared credential-free litellm profile YAML.

    Same transport profile as the acceptance stack (the K3 pins are the
    comparison ModelProfile); any drift fails closed.
    """
    import yaml  # noqa: PLC0415 - Evo venv only

    profile = ModelProfile()
    try:
        loaded = yaml.safe_load(
            (repo_root / ac.LITELLM_YAML).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ComparisonRunnerError("litellm_profile_unreadable") from exc
    if not isinstance(loaded, dict):
        raise ComparisonRunnerError("litellm_profile_malformed")
    models = loaded.get("model_list") or []
    if len(models) != 1:
        raise ComparisonRunnerError("litellm_model_list_must_have_one_entry")
    params = models[0].get("litellm_params") or {}
    if params.get("model") != f"openai/{profile.model_id}":
        raise ComparisonRunnerError("litellm_model_mismatch")
    if params.get("api_key") != ac.API_KEY_PLACEHOLDER:
        raise ComparisonRunnerError("litellm_api_key_must_be_placeholder")
    if params.get("base_url") != profile.base_url:
        raise ComparisonRunnerError("litellm_base_url_mismatch")
    if params.get("stream") is not True:
        raise ComparisonRunnerError("stream_must_be_enabled")
    if params.get("max_tokens") != profile.max_output_tokens:
        raise ComparisonRunnerError("litellm_max_tokens_mismatch")
    if params.get("reasoning_effort") != profile.reasoning_effort:
        raise ComparisonRunnerError("litellm_reasoning_effort_mismatch")
    for key, expected in (
        ("temperature", profile.temperature),
        ("top_p", profile.top_p),
    ):
        value = params.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ComparisonRunnerError(f"litellm_{key}_missing")
        if float(value) != float(expected):
            raise ComparisonRunnerError(f"litellm_{key}_mismatch")
    return dict(params)


def build_comparison_solver(spec: ComparisonRunSpec, *, repo_root: Path) -> Any:
    """Assemble the REAL Evolutionary solver from vendored dojo YAML configs.

    Real operator prompt templates, memory configs and solver dataclasses;
    api_key stays the placeholder; the pinned K3 profile is forced into
    every operator's generation kwargs. Fresh per run: no journal, memory,
    cache or candidate state carries over between configs or methods.
    """
    import yaml  # noqa: PLC0415 - Evo venv only

    from dojo.config_dataclasses.client.base import ClientConfig  # noqa: PLC0415
    from dojo.config_dataclasses.llm.generic_llm import GenericLLMConfig  # noqa: PLC0415
    from dojo.config_dataclasses.llm.jinjaprompt import JinjaPromptConfig  # noqa: PLC0415
    from dojo.config_dataclasses.operators.base import OperatorConfig  # noqa: PLC0415
    from dojo.config_dataclasses.operators.memory import MemoryOpConfig  # noqa: PLC0415
    from dojo.config_dataclasses.solver.evo import EvolutionarySolverConfig  # noqa: PLC0415
    from dojo.solvers.evo.evo import Evolutionary  # noqa: PLC0415

    from research.search.acceptance_inner import (  # noqa: PLC0415
        _OPERATOR_YAMLS,
        _SOLVER_CONFIGS,
        _SOLVER_DEFAULTS_YAML,
    )

    spec.validate()
    pins = InnerEvoValidationConfig()
    profile = ModelProfile()
    if pins.max_output_tokens != profile.max_output_tokens:
        raise ComparisonRunnerError("profile_pin_inconsistent")
    decoded = dict(spec.decoded)
    run_dir = repo_root / spec.run_dir
    solver_root = repo_root / _SOLVER_CONFIGS

    def load_yaml(path: Path) -> dict[str, Any]:
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ComparisonRunnerError(f"dojo_config_unreadable:{path.name}") from exc
        if not isinstance(loaded, dict):
            raise ComparisonRunnerError(f"dojo_config_malformed:{path.name}")
        return dict(loaded)

    def strip_target(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in dict(payload).items() if k != "_target_"}

    def prompt_config(payload: Any) -> Any:
        if payload is None:
            return JinjaPromptConfig()
        return JinjaPromptConfig(**strip_target(payload))

    profile_params = _load_pinned_profile_params(repo_root)
    client_cfg = ClientConfig(
        api="litellm",
        model_id=profile.model_id,
        base_url=profile.base_url,
        api_key=ac.API_KEY_PLACEHOLDER,
        use_azure_client=False,
        provider="openai",
    )
    solver_defaults = load_yaml(solver_root / _SOLVER_DEFAULTS_YAML)

    def build_operator(name: str, relative: str) -> Any:
        operator_yaml = load_yaml(solver_root / "operators" / relative)
        payload = operator_yaml.get(name)
        if not isinstance(payload, dict):
            raise ComparisonRunnerError(f"dojo_operator_missing:{name}")
        llm_payload = strip_target(payload.get("llm") or {})
        generation_kwargs: dict[str, Any] = dict(
            llm_payload.get("generation_kwargs") or {}
        )
        defaults_kwargs = dict(
            solver_defaults.get("operators", {})
            .get(name, {})
            .get("llm", {})
            .get("generation_kwargs", {})
            or {}
        )
        generation_kwargs.update(defaults_kwargs)
        # Comparison transport pins (managed k3 hangs on nonstream requests;
        # provider usage is mandatory for the ledger) plus the profile's
        # pinned sampling parameters.
        generation_kwargs["stream"] = True
        generation_kwargs["max_tokens"] = profile.max_output_tokens
        generation_kwargs["reasoning_effort"] = profile.reasoning_effort
        # litellm rejects reasoning_effort for openai/* custom models unless
        # allowlisted per request; drop_params stays False so an
        # unforwardable parameter always fails loudly.
        generation_kwargs["allowed_openai_params"] = list(
            profile.allowed_openai_params
        )
        generation_kwargs["temperature"] = float(profile_params["temperature"])
        generation_kwargs["top_p"] = float(profile_params["top_p"])
        return OperatorConfig(
            llm=GenericLLMConfig(
                client=client_cfg, generation_kwargs=generation_kwargs
            ),
            system_message_prompt_template=prompt_config(
                payload.get("system_message_prompt_template")
            ),
            init_user_message_prompt_template=prompt_config(
                payload.get("init_user_message_prompt_template")
            ),
            user_message_prompt_template=prompt_config(
                payload.get("user_message_prompt_template")
            ),
        )

    operators = {
        name: build_operator(name, rel) for name, rel in _OPERATOR_YAMLS.items()
    }
    memory_cfg = MemoryOpConfig(
        **strip_target(load_yaml(solver_root / "memory" / "simple_memory.yaml"))
    )
    debug_memory_cfg = MemoryOpConfig(
        **strip_target(load_yaml(solver_root / "memory" / "debug_memory.yaml"))
    )

    cfg = EvolutionarySolverConfig(
        step_limit=pins.step_limit,
        available_packages=["numpy", "pandas", "scikit-learn"],
        operators=operators,
        memory=memory_cfg,
        debug_memory=debug_memory_cfg,
        exp_name="istratde_comparison",
        execution_timeout=spec.job_timeout_seconds,
        time_limit_secs=3000,
        export_search_results=False,
        checkpoint_path=str(run_dir / "checkpoint"),
        use_test_score=False,
        use_complexity=False,
        max_llm_call_retries=1,  # one generation per operator; guard owns attempts
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
    return Evolutionary(cfg, task_info=task_info_for(spec.run.cell.task_binding))


def select_incumbent(journal: Any) -> Any | None:
    """Best successful validation candidate; tie -> earliest actual order.

    Unlike the acceptance runner (which reads the best node only when the
    run completed without error), the comparison must retain the best
    successful candidate even when a later request hit the cell cap or an
    ordinary candidate failed — raw failures stay separate and never
    fabricate an incumbent.
    """
    best: Any | None = None
    for node in journal.nodes:
        if journal.is_root_node(node):
            continue
        if bool(getattr(node, "is_buggy", True)):
            continue
        metric = getattr(node, "metric", None)
        value = getattr(metric, "value", None)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            continue
        if best is None or float(value) > float(getattr(best.metric, "value")):
            best = node
    return best


def _operator_counts(trace_path: Path) -> dict[str, int]:
    counts = {"draft": 0, "improve": 0, "debug": 0, "crossover": 0}
    if not trace_path.exists():
        return counts
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        operator = json.loads(line).get("operator")
        if operator in counts:
            counts[operator] += 1
    return counts


#: Allowlist of public request parameters recorded in the request trace.
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


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run_inner(spec_path: Path, *, repo_root: Path) -> int:
    """Execute one comparison inner run; write run-result.json.

    Exit 0: run-result.json written (check its status/termination fields).
    Exit 2: refused/failed before any result could be produced.
    """
    check_comparison_review(repo_root)  # gate BEFORE ledger/auth/model activity
    spec = ComparisonRunSpec.from_json_file(spec_path)
    if spec.runner_commit != ac.git_head_commit(repo_root):
        raise ComparisonRunnerError("runner_commit_mismatch")
    bindings = load_task_bindings()
    cells = iter_cells(bindings, spec.runner_commit)
    roster = {cell.identity(): cell for cell in cells}
    if spec.run.cell.identity() not in roster:
        raise ComparisonRunnerError("cell_not_in_roster")
    registry = EvaluatorRegistry.load()
    data_evidence = verify_task_data_identity(
        spec.run.cell.task_binding, repo_root=repo_root, registry=registry
    )
    ledger_path = repo_root / spec.ledger_path
    if ledger_path.is_symlink() or not ledger_path.exists():
        # The runner never creates or resets the one comparison ledger, and
        # never follows a symlinked ledger (foreign accounting target).
        raise ComparisonRunnerError("comparison_ledger_missing")
    run_dir = repo_root / spec.run_dir
    # Fresh controller-private run directory ONLY: an interrupted config is
    # never silently rerun (existing dir/checkpoint refused), and no
    # symlinked path component may redirect artifacts outside the runtime
    # (checked BEFORE any ledger/auth/dojo activity).
    if run_dir.exists() or run_dir.is_symlink():
        raise ComparisonRunnerError("run_dir_not_fresh")
    ancestor = run_dir.parent
    while ancestor != repo_root and ancestor != ancestor.parent:
        if ancestor.is_symlink():
            raise ComparisonRunnerError("run_dir_symlink_ancestor")
        ancestor = ancestor.parent
    try:
        run_dir.mkdir(parents=True)
    except FileExistsError as exc:
        raise ComparisonRunnerError("run_dir_not_fresh") from exc
    os.environ.setdefault("LOGGING_DIR", str(run_dir / "logs"))

    # Deferred imports: dojo needs LOGGING_DIR; keep them out of module
    # scope so the spec/result types stay importable in the research venv.
    # litellm must use its BUNDLED model-cost map: frozen comparison startup
    # never fetches mutable public pricing metadata.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import random  # noqa: PLC0415

    import numpy  # noqa: PLC0415
    from dojo.core.interpreters.base import ExecutionResult  # noqa: PLC0415
    from dojo.solvers.evo.operator_trace import OperatorTraceWriter  # noqa: PLC0415
    from dojo.utils.logger import config_logger  # noqa: PLC0415
    from research.adapters.sandbox_eval_client import SandboxEvalClient  # noqa: PLC0415

    config_logger(None)  # low-resource global logger: no file/wandb handlers

    # Seed the SEARCH RNGs with the paired inner seed (method-independent by
    # design). Provider generation stays nondeterministic: no unsupported
    # provider seed parameter is sent, and no result is cached or shared.
    random.seed(spec.run.inner_seed)
    numpy.random.seed(spec.run.inner_seed)
    rng_evidence = {
        "python_random_seed": spec.run.inner_seed,
        "numpy_random_seed": spec.run.inner_seed,
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
        debug_budget = 0 if int(spec.decoded["max_debug_depth"]) == 0 else 2

        incumbent: Any | None = None
        end_snapshot: dict[str, Any] = {}
        cleanup_verified = False
        with ComparisonLedger.open(ledger_path, cells=cells) as ledger:
            if ledger.stopped:
                # Never run against a stopped comparison ledger: refuse
                # BEFORE any provider call or sandbox submission.
                raise ComparisonRunnerError("comparison_ledger_stopped")
            if ledger.final_phase is not None:
                # The irreversible final phase is active: no inner run may
                # start generation afterwards — not even an unused
                # invalid/skipped config with remaining quota or an
                # alternative fresh run directory. Refuse BEFORE any
                # provider call or sandbox submission (the reservation
                # guard independently refuses, defense in depth).
                raise ComparisonRunnerError("comparison_final_phase_active")
            budget_control = ComparisonMainBudgetControl(
                ledger,
                spec.run,
                debug_budget=debug_budget,
                control_log_path=run_dir / "budget-control.jsonl",
            )

            def attempt_sink(
                reservation: Any,
                messages: Any,
                model_kwargs: Any,
                operator: Any = None,
            ) -> None:
                """Link every provider attempt to its ledger reservation and
                persist the PUBLIC request (never headers/credentials). A
                redaction or write failure aborts the attempt before the
                provider is called (reservation retained + global stop)."""
                attempt_counter["n"] += 1
                if operator == "debug":
                    budget_control.note_debug_attempt()
                record = {
                    "run_key": spec.run.run_key(),
                    "attempt": attempt_counter["n"],
                    "operator": operator or "unknown",
                    "kind": "debug" if operator == "debug" else "main",
                    "request_id": reservation.request_id,
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
                    handle.write(
                        json.dumps(record, sort_keys=True, default=str) + "\n"
                    )

            token = CancellationToken(deadline_epoch=ledger.deadline_epoch)
            live_jobs_path = run_dir / "live-jobs.json"
            submission_intents: set[str] = set()

            def persist_live_jobs(_event: Any = None) -> None:
                """Persist submission intents / live jobs / uncertain
                submissions BEFORE polling; survives interruption for
                emergency cleanup."""
                if isinstance(_event, Mapping):
                    if _event.get("event") == "submission_intent":
                        submission_intents.add(str(_event.get("trace_id")))
                    elif _event.get("event") == "job_registered":
                        submission_intents.discard(str(_event.get("trace_id")))
                atomic_write_json(live_jobs_path, {
                    "schema": LIVE_JOBS_SCHEMA,
                    "run_key": spec.run.run_key(),
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
            task = ComparisonSandboxTask(
                client,
                token,
                ledger=ledger,
                data_dir=str(data_evidence["worker_data_public_path"]),
                task_id=str(data_evidence["registry_task_id"]),
                resource_type=spec.resource_type,
                job_timeout_seconds=spec.job_timeout_seconds,
                name_prefix="istratde-comparison",
                execution_result_cls=ExecutionResult,
            )
            solver = build_comparison_solver(spec, repo_root=repo_root)
            trace_path = run_dir / "operator-trace.jsonl"
            solver.operator_tracer = OperatorTraceWriter(trace_path)
            stop_file = run_dir / "stop-request.json"
            solver.generation_outcome_sink = make_generation_outcome_sink(
                run_dir=run_dir,
                run_id=spec.run.run_key(),
                secrets=secrets,
                stop_file=stop_file,
                solver=solver,
            )
            prepare = make_prepare_operator(
                credentials, operator_clients(solver), secrets
            )
            termination = run_comparison_search(
                solver,
                task,
                {},
                ledger=ledger,
                run=spec.run,
                token=token,
                budget_control=budget_control,
                cancel_live_jobs=client.cancel_all_live,
                checkpoint_path=run_dir / "search-control-checkpoint.json",
                attempt_sink=attempt_sink,
                prepare_operator=prepare,
                cooperative_stop=lambda: _check_cooperative_stop(stop_file),
            )
            incumbent = select_incumbent(solver.journal)
            control_checkpoint = run_dir / "search-control-checkpoint.json"
            cancellation_ok = True
            if control_checkpoint.exists():
                try:
                    control = json.loads(
                        control_checkpoint.read_text(encoding="utf-8")
                    )
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
                and not submission_intents  # durable intent/registration gap
            )
            if not cleanup_verified:
                # Unresolved live/uncertain jobs after cancellation are a
                # GLOBAL stop condition, durably recorded inside the
                # ledger's ownership — never a silent evidence gap.
                ledger.stop("sandbox_cleanup_unproven")
                if termination == "completed":
                    termination = "cleanup_unproven"
            # Strict prediction binding INSIDE ledger ownership: every job's
            # snapshot hash is verified; a SCORED job without a proven
            # snapshot stops the ledger instead of being silently skipped.
            prediction_artifacts = bind_comparison_prediction_snapshots(
                task.jobs, run_dir, repo_root, ledger
            )
            incumbent_evidence = _bind_incumbent(
                incumbent, task.jobs, prediction_artifacts, ledger
            )
            end_snapshot = ledger.snapshot()
            deadline_epoch = ledger.deadline_epoch

        # Bind the generated-candidate journal into the run artifacts.
        journal_src = run_dir / "checkpoint" / "journal.jsonl"
        journal_path = run_dir / "journal.jsonl"
        if journal_src.exists():
            journal_path.write_bytes(journal_src.read_bytes())
        counts = _operator_counts(trace_path)
        main_candidates = sum(
            counts[name] for name in ("draft", "improve", "crossover")
        )
        generations_completed = int(getattr(solver.state, "current_generation", 0))
        outcomes_path = run_dir / "generation-outcomes.jsonl"

        run_requests = [
            rec
            for rec in end_snapshot.get("requests", [])
            if rec.get("run_identity") == spec.run.identity()
        ]
        stopped_globally = end_snapshot.get("stopped") is not None
        if (
            termination == "completed"
            and not stopped_globally
            and cleanup_verified
        ):
            status = "completed"
        elif (
            termination.startswith("CellBudgetExhausted")
            and not stopped_globally
        ):
            status = "cell_terminated"
        else:
            # A globally stopped ledger can never report a completed run,
            # and a refused winner never survives as evidence.
            status = "failed"
        result = {
            "schema": RUN_RESULT_SCHEMA,
            "run_key": spec.run.run_key(),
            "cell_key": spec.run.cell.cell_key(),
            "config_index": spec.run.config_index,
            "inner_seed": spec.run.inner_seed,
            "decoded": dict(spec.decoded),
            "config_identity": comparison_config_identity(spec),
            "run_identity": spec.run.identity(),
            "cell_identity": spec.run.cell.identity(),
            "protocol_identity": canonical_protocol().identity(),
            "model_profile_identity": ModelProfile().identity(),
            "policy_version": policy.policy_version,
            "policy_sha256": policy.policy_sha256,
            "runner_commit": spec.runner_commit,
            "dataset": data_evidence,
            "status": status,
            "termination": termination,
            "incumbent": incumbent_evidence,
            "metric": "accuracy",
            "metric_direction": "maximize",
            "debug_budget": debug_budget,
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
            "job_ids": sorted({j["job_id"] for j in task.jobs if j.get("job_id")}),
            "worker_cleanup_verified": cleanup_verified,
            "requests": {
                "main": sum(1 for r in run_requests if r["kind"] == "main"),
                "debug": sum(1 for r in run_requests if r["kind"] == "debug"),
                "charged_tokens": sum(int(r["charged_tokens"]) for r in run_requests),
            },
            "budget": {
                "global_committed": end_snapshot.get("global", {}).get("committed"),
                "stopped": end_snapshot.get("stopped"),
                "cell": end_snapshot.get("cells", {}).get(
                    spec.run.cell.cell_key()
                ),
                "deadline_epoch": deadline_epoch,
            },
            "started_at": end_snapshot.get("started_at"),
            "finished_at": time.time(),
        }
        assert_redacted(result, *secrets)
        atomic_write_json(run_dir / "run-result.json", result)
        return 0
    finally:
        auth_helper.close()  # only this invocation's owned helpers


def bind_comparison_prediction_snapshots(
    jobs: list[dict[str, Any]],
    run_dir: Path,
    repo_root: Path,
    ledger: ComparisonLedger,
) -> dict[str, str]:
    """Strictly bind every job's trusted prediction snapshot into the run.

    Unlike the acceptance helper (which silently skips jobs without a
    snapshot): the snapshot path must be EXACTLY the job's own storage
    snapshot (never another job's file), a SCORED job missing its snapshot
    is an integrity failure, and any unreadable/mismatched bytes or
    unwritable destination stops the ledger GLOBALLY (durably, inside its
    ownership) before raising. All jobs are verified, not only the
    incumbent.
    """
    artifacts: dict[str, str] = {}
    storage_root = repo_root / ac.DISPATCHER_STORAGE_ROOT
    for job in jobs:
        evaluation = job.get("evaluation") or {}
        snapshot = evaluation.get("prediction_snapshot")
        job_id = job.get("job_id")
        if not job_id:
            continue
        if not snapshot:
            if job.get("evidence_verified") is True:
                ledger.stop("prediction_evidence_missing")
                raise ComparisonRunnerError("prediction_snapshot_missing")
            continue
        reason: str | None = None
        try:
            try:
                # Only the job's OWN snapshot object is valid evidence:
                # validated container path bound to the exact job id, then
                # resolved inside the dispatcher storage root.
                host_path = ac.job_snapshot_host_path(
                    snapshot, job_id, storage_root=storage_root
                )
            except ComparisonSpecError:
                reason = "prediction_snapshot_path_invalid"
                raise ComparisonRunnerError(reason)
            data = host_path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != evaluation.get("prediction_sha256"):
                reason = "prediction_snapshot_mismatch"
                raise ComparisonRunnerError(reason)
            predictions_dir = run_dir / "predictions"
            predictions_dir.mkdir(exist_ok=True)
            (predictions_dir / f"{job_id}.csv").write_bytes(data)
        except OSError:
            reason = reason or "prediction_snapshot_unreadable"
            ledger.stop(f"prediction_evidence_invalid:{reason}")
            raise ComparisonRunnerError(reason) from None
        except ComparisonRunnerError:
            ledger.stop(f"prediction_evidence_invalid:{reason}")
            raise
        artifacts[str(job_id)] = digest
    return artifacts


def _bind_incumbent(
    incumbent: Any | None,
    jobs: list[dict[str, Any]],
    prediction_artifacts: Mapping[str, str],
    ledger: ComparisonLedger,
) -> dict[str, Any] | None:
    """Winner evidence bound to trusted job/prediction identity.

    The selected incumbent must trace to a VERIFIED job whose executed
    source hash equals the incumbent code bytes and whose prediction
    snapshot is bound; otherwise the winner is REFUSED and the ledger
    stops (corrupted evidence never produces a winner).
    """
    if incumbent is None:
        return None
    metric = getattr(incumbent, "metric", None)
    info = getattr(metric, "info", None)
    job_id = (
        str(info.get("job_id"))
        if isinstance(info, Mapping) and info.get("job_id")
        else ""
    )
    code = getattr(incumbent, "code", "") or ""
    code_sha256 = hashlib.sha256(
        code.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    job = next(
        (
            j
            for j in jobs
            if j.get("job_id") == job_id and j.get("evidence_verified") is True
        ),
        None,
    )
    score = float(getattr(metric, "value"))
    gate = (job or {}).get("source_gate") or {}
    evaluation = (job or {}).get("evaluation") or {}
    submitted = code.encode("utf-8", errors="surrogatepass")
    refusal: str | None = None
    job_score = (job or {}).get("score")
    eval_score = evaluation.get("score")
    scores_match = (
        isinstance(job_score, (int, float))
        and not isinstance(job_score, bool)
        and isinstance(eval_score, (int, float))
        and not isinstance(eval_score, bool)
        and float(job_score) == score
        and float(eval_score) == score
    )
    if not job_id or job is None:
        refusal = "job_unproven"
    elif not scores_match:
        # The winner cannot detach from its trusted evaluated score.
        refusal = "score_mismatch"
    elif job.get("submitted_source_sha256") != code_sha256:
        refusal = "executed_source_mismatch"
    elif (
        gate.get("entrypoint_sha256") != code_sha256
        or gate.get("source_sha256")
        != _single_file_tree_sha256(str(gate.get("entrypoint") or ""), submitted)
    ):
        refusal = "source_gate_mismatch"
    elif (
        job_id not in prediction_artifacts
        or prediction_artifacts.get(job_id) != evaluation.get("prediction_sha256")
    ):
        refusal = "prediction_mismatch"
    if refusal is not None:
        ledger.stop(f"incumbent_evidence_invalid:{refusal}")
        return None
    return {
        "node_id": str(getattr(incumbent, "id", "")),
        "validation_score": score,
        "code_sha256": code_sha256,
        "job_id": job_id,
        "prediction_sha256": prediction_artifacts[job_id],
        "operators_used": list(getattr(incumbent, "operators_used", [])),
    }


def _check_cooperative_stop(stop_file: Path) -> str | None:
    """Run-scoped cooperative stop (a touched file next to the run
    artifacts), checked ONLY at safe boundaries."""
    try:
        if stop_file.is_file():
            return "stop_requested_file"
    except OSError:
        return "stop_check_unreadable"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research.comparison.runner")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="execute one comparison inner run")
    run_parser.add_argument("--spec", required=True, type=Path)
    run_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.command == "run":
            return run_inner(args.spec, repo_root=repo_root)
    except ComparisonReviewRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except (
        ComparisonRunnerError,
        ComparisonBudgetError,
        ComparisonSpecError,
        CredentialError,
    ) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2  # unreachable: unknown command


if __name__ == "__main__":
    sys.exit(main())
