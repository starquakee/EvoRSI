"""Frozen protocol for the EvoRSI iStratDE controlled pilot comparison.

Stdlib-only on purpose: this package is imported by the parent controller,
the Evo child and the offline research tests.

Every value the human approved on 2026-10-10
(``tasks/prd-istratde-comparison.md``) is pinned here:

- 3 tasks (banknote UCI 267, wdbc UCI 17, spambase UCI 94) x 3 methods
  (FIXED, RANDOM, ISTRATDE) x 3 search seeds (11, 22, 33) = 27 cells;
- 8 independent inner searches per cell = 216 inner runs, each 3
  generations x 2 main candidates plus at most 2 debug requests;
- per-cell budget 64 requests / 200000 tokens (no quota borrowing);
- global budget 1728 requests / 5400000 tokens / 86400 seconds;
- FIXED theta = (0.5, 1.0, 0.4, 0.25, 1) over the pinned comparison dims;
- K3 model profile: reasoning_effort=low, max_tokens=8192, temperature=1,
  top_p=0.95, stream=true, allowed_openai_params=['reasoning_effort'],
  transport retries 0.

``ComparisonProtocol`` refuses any deviation from the approved values: the
type cannot be used to widen the design. Canonical identities are sha256
over canonical JSON (sorted keys, tight separators), so tampering with any
bound value changes the identity.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from research.search.search_vector import SearchDim

COMPARISON_SCHEMA = "istratde-comparison-protocol.v1"


class ComparisonSpecError(ValueError):
    """A comparison protocol/spec constraint was violated.

    The message IS the machine-readable constraint name (e.g.
    ``protocol_frozen:search_seeds``) so tests and later gates can assert
    on it directly.
    """


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Canonical JSON rendering (sorted keys, tight separators)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def identity_of(payload: Mapping[str, Any]) -> str:
    """Content identity (sha256) of a canonical-JSON payload."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _positive_int(value: Any, rule: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ComparisonSpecError(rule)
    return value


def _nonnegative_int(value: Any, rule: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ComparisonSpecError(rule)
    return value


def _positive_finite(value: Any, rule: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ComparisonSpecError(rule)
    return float(value)


# --- task pins (PRD References: official UCI shapes) -------------------------
@dataclass(frozen=True)
class TaskPin:
    """Approved identity of one comparison task (shape from the UCI page)."""

    task_id: str
    uci_id: int
    rows: int
    features: int

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ComparisonSpecError("task_id_invalid")
        _positive_int(self.uci_id, "uci_id_invalid")
        _positive_int(self.rows, "task_rows_invalid")
        _positive_int(self.features, "task_features_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "uci_id": self.uci_id,
            "rows": self.rows,
            "features": self.features,
        }


TASK_BANKNOTE = TaskPin("banknote", 267, 1372, 4)
TASK_WDBC = TaskPin("wdbc", 17, 569, 30)
TASK_SPAMBASE = TaskPin("spambase", 94, 4601, 57)

#: Canonical task order (PRD order); execution rotation is a separate
#: frozen artifact (EXP-006), never implied by enumeration order here.
TASKS: tuple[TaskPin, ...] = (TASK_BANKNOTE, TASK_WDBC, TASK_SPAMBASE)
TASKS_BY_ID = {task.task_id: task for task in TASKS}

#: One fixed five-fold grouped split per task, seed 20261010 + UCI id (FR1).
SPLIT_SEED_BASE = 20261010


def split_seed(task: TaskPin) -> int:
    """The fixed split seed for a task (20261010 + UCI id)."""
    return SPLIT_SEED_BASE + task.uci_id


# --- method / seed / shape pins ----------------------------------------------
#: Canonical method order (PRD order).
METHODS: tuple[str, str, str] = ("FIXED", "RANDOM", "ISTRATDE")
SEARCH_SEEDS: tuple[int, int, int] = (11, 22, 33)
CONFIGS_PER_CELL = 8
INNER_GENERATIONS = 3
MAIN_CANDIDATES_PER_GENERATION = 2
NUM_CELLS = 27
NUM_INNER_RUNS = 216

_SOLVER = "search.runner.solver"
_WEIGHTS = f"{_SOLVER}.experience.parent_selection.weights"

#: OUTER search dims for the comparison. Names/hydra paths match the US-007
#: retained dims, but max_debug_depth is bounded to [0, 2] here (PRD:
#: "debug integer[0,2]", at most two additional debug requests per inner
#: run) — tighter than the acceptance bound [0, 6].
COMPARISON_DIMS: tuple[SearchDim, ...] = (
    SearchDim(
        name="crossover_prob",
        lower=0.05,
        upper=0.95,
        kind="float",
        hydra_path=f"{_SOLVER}.crossover_prob",
        effect="operator coin flip improve-vs-crossover",
    ),
    SearchDim(
        name="score_weight",
        lower=0.0,
        upper=2.0,
        kind="float",
        hydra_path=f"{_WEIGHTS}.score",
        effect="experience parent-selection utility component (score)",
    ),
    SearchDim(
        name="delta_weight",
        lower=0.0,
        upper=2.0,
        kind="float",
        hydra_path=f"{_WEIGHTS}.delta",
        effect="experience parent-selection utility component (delta)",
    ),
    SearchDim(
        name="novelty_weight",
        lower=0.0,
        upper=2.0,
        kind="float",
        hydra_path=f"{_WEIGHTS}.novelty",
        effect="experience parent-selection utility component (novelty)",
    ),
    SearchDim(
        name="max_debug_depth",
        lower=0.0,
        upper=2.0,
        kind="int",
        hydra_path=f"{_SOLVER}.max_debug_depth",
        effect="debug-cycle attempt count (comparison cap: 2)",
    ),
)

#: The FIXED method's default theta (PRD: (.5, 1.0, .4, .25, 1)).
FIXED_THETA: tuple[float, float, float, float, float] = (0.5, 1.0, 0.4, 0.25, 1.0)


def decode_theta(vector: Sequence[float]) -> dict[str, float | int]:
    """Decode one comparison theta into pinned-dim values.

    Values are clipped to the dim bounds and the integer dim is rounded
    (round-half-even, matching the legacy ``jnp.rint`` decode) — the same
    encoding/rounding the US-007 acceptance vector uses, so all three
    methods decode identically. Non-finite or bool elements and wrong
    lengths raise ``ComparisonSpecError``.
    """
    if len(vector) != len(COMPARISON_DIMS):
        raise ComparisonSpecError("theta_length_mismatch")
    decoded: dict[str, float | int] = {}
    for dim, raw in zip(COMPARISON_DIMS, vector):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ComparisonSpecError("nonfinite_or_invalid_theta")
        if not math.isfinite(float(raw)):
            raise ComparisonSpecError("nonfinite_or_invalid_theta")
        value = min(max(float(raw), dim.lower), dim.upper)
        if dim.kind == "int":
            decoded[dim.name] = int(round(value))
        else:
            decoded[dim.name] = value
    return decoded


def theta_gate(decoded: Mapping[str, Any]) -> None:
    """Fixed gate over decoded comparison dims; methods cannot change it.

    Enforces the pinned ranges post-decode and the weight-sum > 0 rule
    (an all-zero parent-selection utility is rejected, never silently
    repaired).
    """
    for dim in COMPARISON_DIMS:
        value = decoded[dim.name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ComparisonSpecError("nonfinite_or_invalid_theta")
        if dim.kind == "int" and (
            not isinstance(value, int) or not dim.lower <= value <= dim.upper
        ):
            raise ComparisonSpecError("max_debug_depth_out_of_range")
        if dim.kind == "float" and not dim.lower <= value <= dim.upper:
            raise ComparisonSpecError("theta_dimension_out_of_range")
    weights = (
        float(decoded["score_weight"])
        + float(decoded["delta_weight"])
        + float(decoded["novelty_weight"])
    )
    if weights <= 0:
        raise ComparisonSpecError("empty_parent_selection_weights")


# --- model profile pin ---------------------------------------------------------
@dataclass(frozen=True)
class ModelProfile:
    """K3 request profile shared identically by all three methods.

    Field defaults ARE the approved pins; the frozen protocol and the specs
    additionally require equality with the defaults, so a well-formed but
    different profile can be constructed for validation tests yet never
    enters a comparison spec.
    """

    model_id: str = "k3"
    base_url: str = "https://api.kimi.com/coding/v1"
    reasoning_effort: str = "low"
    max_output_tokens: int = 8192
    temperature: float = 1.0
    top_p: float = 0.95
    stream: bool = True
    allowed_openai_params: tuple[str, ...] = ("reasoning_effort",)
    transport_retries: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ComparisonSpecError("model_id_invalid")
        if not isinstance(self.base_url, str) or not self.base_url.startswith(
            "https://"
        ):
            raise ComparisonSpecError("model_base_url_invalid")
        # Managed k3 supports low/high/max; the comparison pins the explicit
        # low effort (thinking is never disabled — that routes to K2.8).
        if self.reasoning_effort != "low":
            raise ComparisonSpecError("reasoning_effort_must_be_low")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or not 1 <= self.max_output_tokens <= 8192
        ):
            raise ComparisonSpecError("max_output_tokens_impractical")
        _temperature = self.temperature
        if (
            isinstance(_temperature, bool)
            or not isinstance(_temperature, (int, float))
            or not math.isfinite(float(_temperature))
            or not 0 < float(_temperature) <= 2
        ):
            raise ComparisonSpecError("temperature_out_of_range")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not math.isfinite(float(self.top_p))
            or not 0 < float(self.top_p) <= 1
        ):
            raise ComparisonSpecError("top_p_out_of_range")
        if self.stream is not True:
            raise ComparisonSpecError("stream_must_be_enabled")
        if (
            not isinstance(self.allowed_openai_params, tuple)
            or not all(isinstance(p, str) for p in self.allowed_openai_params)
            or not set(self.allowed_openai_params) <= {"reasoning_effort"}
        ):
            raise ComparisonSpecError("allowed_openai_params_invalid")
        _nonnegative_int(self.transport_retries, "transport_retries_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "base_url": self.base_url,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "stream": self.stream,
            "allowed_openai_params": list(self.allowed_openai_params),
            "transport_retries": self.transport_retries,
        }

    def identity(self) -> str:
        return identity_of(self.to_dict())


# --- budget pins -----------------------------------------------------------------
@dataclass(frozen=True)
class InnerBudget:
    """Per-inner-run request shape: 6 main + at most 2 debug requests."""

    max_main_requests: int = 6
    max_debug_requests: int = 2

    def __post_init__(self) -> None:
        _positive_int(self.max_main_requests, "inner_budget_invalid")
        _nonnegative_int(self.max_debug_requests, "inner_budget_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_main_requests": self.max_main_requests,
            "max_debug_requests": self.max_debug_requests,
        }


@dataclass(frozen=True)
class CellBudget:
    """Per-cell hard caps; no quota borrowing between cells."""

    max_requests: int = 64
    max_tokens: int = 200_000

    def __post_init__(self) -> None:
        _positive_int(self.max_requests, "cell_budget_invalid")
        _positive_int(self.max_tokens, "cell_budget_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_requests": self.max_requests,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True)
class GlobalBudget:
    """Global hard caps; the clock starts at the supervised live start."""

    max_requests: int = 1728
    max_tokens: int = 5_400_000
    max_elapsed_seconds: float = 86_400.0

    def __post_init__(self) -> None:
        _positive_int(self.max_requests, "global_budget_invalid")
        _positive_int(self.max_tokens, "global_budget_invalid")
        _positive_finite(self.max_elapsed_seconds, "global_budget_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_requests": self.max_requests,
            "max_tokens": self.max_tokens,
            "max_elapsed_seconds": float(self.max_elapsed_seconds),
        }


# --- frozen protocol -------------------------------------------------------------
def _freeze(rule: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ComparisonSpecError(rule)


@dataclass(frozen=True)
class ComparisonProtocol:
    """The approved 2026-10-10 comparison design, frozen.

    Every field defaults to the approved pin and ``__post_init__`` refuses
    any deviation with ``protocol_frozen:<field>``; the type exists so the
    pins are explicit, typed, serializable and identity-bound — not to be
    parameterized. Budget arithmetic is cross-checked so the request caps
    stay mutually consistent (8 configs x (6 main + 2 debug) == 64 per
    cell; 27 cells x 64 == 1728 globally).
    """

    schema: str = COMPARISON_SCHEMA
    tasks: tuple[TaskPin, ...] = TASKS
    methods: tuple[str, str, str] = METHODS
    search_seeds: tuple[int, int, int] = SEARCH_SEEDS
    configs_per_cell: int = CONFIGS_PER_CELL
    inner_generations: int = INNER_GENERATIONS
    main_candidates_per_generation: int = MAIN_CANDIDATES_PER_GENERATION
    fixed_theta: tuple[float, float, float, float, float] = FIXED_THETA
    inner_budget: InnerBudget = InnerBudget()
    cell_budget: CellBudget = CellBudget()
    global_budget: GlobalBudget = GlobalBudget()
    model_profile: ModelProfile = ModelProfile()
    split_seed_base: int = SPLIT_SEED_BASE

    def __post_init__(self) -> None:
        # Strict integer typing FIRST: float stand-ins like 8.0 or
        # (11.0, 22.0, 33.0) compare equal to the pins but would reach
        # range(), seed initializers and persisted identities with a
        # different JSON type.
        for int_field in (
            "configs_per_cell",
            "inner_generations",
            "main_candidates_per_generation",
            "split_seed_base",
        ):
            _positive_int(
                getattr(self, int_field), f"protocol_field_not_an_int:{int_field}"
            )
        if not isinstance(self.search_seeds, tuple) or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in self.search_seeds
        ):
            raise ComparisonSpecError("protocol_field_not_an_int:search_seeds")
        _freeze("protocol_frozen:schema", self.schema, COMPARISON_SCHEMA)
        _freeze("protocol_frozen:tasks", self.tasks, TASKS)
        _freeze("protocol_frozen:methods", self.methods, METHODS)
        _freeze("protocol_frozen:search_seeds", self.search_seeds, SEARCH_SEEDS)
        _freeze(
            "protocol_frozen:configs_per_cell",
            self.configs_per_cell,
            CONFIGS_PER_CELL,
        )
        _freeze(
            "protocol_frozen:inner_generations",
            self.inner_generations,
            INNER_GENERATIONS,
        )
        _freeze(
            "protocol_frozen:main_candidates_per_generation",
            self.main_candidates_per_generation,
            MAIN_CANDIDATES_PER_GENERATION,
        )
        # Strict element types (bool sneak-in like (…, True) == (…, 1.0))
        # and finiteness are enforced by decode_theta; the gate enforces the
        # weight-sum rule on the decoded FIXED theta. The container itself
        # must be an immutable tuple: a caller-owned list would stay mutable
        # after validation.
        if not isinstance(self.fixed_theta, tuple):
            raise ComparisonSpecError("fixed_theta_must_be_a_tuple")
        decoded = decode_theta(self.fixed_theta)
        theta_gate(decoded)
        _freeze("protocol_frozen:fixed_theta", tuple(self.fixed_theta), FIXED_THETA)
        _freeze("protocol_frozen:inner_budget", self.inner_budget, InnerBudget())
        _freeze("protocol_frozen:cell_budget", self.cell_budget, CellBudget())
        _freeze("protocol_frozen:global_budget", self.global_budget, GlobalBudget())
        _freeze(
            "protocol_frozen:model_profile", self.model_profile, ModelProfile()
        )
        _freeze(
            "protocol_frozen:split_seed_base", self.split_seed_base, SPLIT_SEED_BASE
        )
        if self.num_cells != NUM_CELLS or self.num_inner_runs != NUM_INNER_RUNS:
            raise ComparisonSpecError("protocol_frozen:cell_count")
        if (
            self.inner_budget.max_main_requests
            != self.inner_generations * self.main_candidates_per_generation
        ):
            raise ComparisonSpecError("budget_arithmetic_mismatch")
        if (
            self.configs_per_cell
            * (
                self.inner_budget.max_main_requests
                + self.inner_budget.max_debug_requests
            )
            != self.cell_budget.max_requests
        ):
            raise ComparisonSpecError("budget_arithmetic_mismatch")
        if self.num_cells * self.cell_budget.max_requests != (
            self.global_budget.max_requests
        ):
            raise ComparisonSpecError("budget_arithmetic_mismatch")

    @property
    def num_cells(self) -> int:
        return len(self.tasks) * len(self.methods) * len(self.search_seeds)

    @property
    def num_inner_runs(self) -> int:
        return self.num_cells * self.configs_per_cell

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tasks": [task.to_dict() for task in self.tasks],
            "methods": list(self.methods),
            "search_seeds": list(self.search_seeds),
            "configs_per_cell": self.configs_per_cell,
            "inner_generations": self.inner_generations,
            "main_candidates_per_generation": self.main_candidates_per_generation,
            "fixed_theta": [float(v) for v in self.fixed_theta],
            "dims": [
                {
                    "name": dim.name,
                    "lower": dim.lower,
                    "upper": dim.upper,
                    "kind": dim.kind,
                    "hydra_path": dim.hydra_path,
                }
                for dim in COMPARISON_DIMS
            ],
            "inner_budget": self.inner_budget.to_dict(),
            "cell_budget": self.cell_budget.to_dict(),
            "global_budget": self.global_budget.to_dict(),
            "model_profile": self.model_profile.to_dict(),
            "split_seed_base": self.split_seed_base,
        }

    def identity(self) -> str:
        """Canonical protocol identity (sha256 of the canonical rendering).

        Task DATA hashes are deliberately NOT part of this identity: they
        are bound per task by TaskBinding (EXP-002) and enter the cell
        identities through the binding identity.
        """
        return identity_of(self.to_dict())


def canonical_protocol() -> ComparisonProtocol:
    """The one frozen protocol instance (all pins)."""
    return ComparisonProtocol()
