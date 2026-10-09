"""Validation-config contract for the trustworthy acceptance run (US-007).

Stdlib-only on purpose: this module is imported from OpenMLE-Evo tests, so it
must not pull in jax/httpx or any other third-party dependency.

Terminology (kept explicit everywhere in this package):

- INNER generations are Evo *solver* generations
  (``search.runner.solver.num_generations``): the evolutionary loop that
  OpenMLE-Evo runs inside ONE evaluation of one config vector.
- OUTER generations are iStratDE DE generations over config vectors: the
  differential-evolution loop that proposes the vectors whose fitness is one
  full inner Evo run. Outer fitness is minimized natively by the DE, so the
  mapping is ``fitness = -accuracy``.

The constraints below pin the inner Evo validation experiment
(``trustworthy_validation.yaml``) so that every inner generation actually
completes within the step budget and no auxiliary model calls (prompt-memory rich summaries or analyze LLM)
can fire. Streaming is required for the managed k3 acceptance transport;
full stream consumption and provider usage remain inside the budget guard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# Keys accepted by validate_inner_evo_config. Keys NOT in this set are
# ignored (documented behavior): the validator only pins the constraints the
# acceptance evidence depends on and stays forward-compatible with extra
# experiment-config keys.
_ACCEPTED_KEYS = (
    "num_generations",
    "individuals_per_generation",
    "step_limit",
    "num_islands",
    "num_generations_till_crossover",
    "llm_concurrency",
    "sandbox_concurrency",
    "experience_enabled",
    "prompt_memory_enabled",
    "stream",
    "max_output_tokens",
)

_INT_DEFAULTS = {
    "num_generations": 3,
    "individuals_per_generation": 2,
    "step_limit": 12,
    "num_islands": 1,
    "num_generations_till_crossover": 1,
    "llm_concurrency": 1,
    "sandbox_concurrency": 1,
    "max_output_tokens": 4096,
}

_BOOL_DEFAULTS = {
    "experience_enabled": True,
    "prompt_memory_enabled": False,
    "stream": True,
}

MAX_OUTPUT_TOKENS_LIMIT = 8192


class ValidationConfigError(ValueError):
    """A validation-config constraint was violated.

    The message IS the machine-readable constraint name (e.g.
    ``step_limit_truncates_generations``) so tests and the Evo-side config
    check can assert on it directly.
    """


def _int_field(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key, _INT_DEFAULTS[key])
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationConfigError(f"{key}_must_be_an_int")
    return value


def _bool_field(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key, _BOOL_DEFAULTS[key])
    if not isinstance(value, bool):
        raise ValidationConfigError(f"{key}_must_be_a_bool")
    return value


def validate_inner_evo_config(mapping: Mapping[str, Any]) -> None:
    """Validate the INNER Evo validation experiment config.

    ``mapping`` is the experiment config (or a subset of it); absent keys
    fall back to the acceptance defaults and keys outside the accepted set
    are ignored. Raises :class:`ValidationConfigError` whose message is the
    machine-readable name of the first violated constraint.

    Note: "generations" here always means INNER Evo solver generations; the
    OUTER iStratDE loop is configured separately (see :func:`outer_de_bounds`).
    """
    if not isinstance(mapping, Mapping):
        raise ValidationConfigError("config_must_be_a_mapping")
    num_generations = _int_field(mapping, "num_generations")
    individuals = _int_field(mapping, "individuals_per_generation")
    step_limit = _int_field(mapping, "step_limit")
    num_islands = _int_field(mapping, "num_islands")
    gens_till_crossover = _int_field(mapping, "num_generations_till_crossover")
    llm_concurrency = _int_field(mapping, "llm_concurrency")
    sandbox_concurrency = _int_field(mapping, "sandbox_concurrency")
    max_output_tokens = _int_field(mapping, "max_output_tokens")
    experience_enabled = _bool_field(mapping, "experience_enabled")
    prompt_memory_enabled = _bool_field(mapping, "prompt_memory_enabled")
    stream_enabled = _bool_field(mapping, "stream")

    if num_generations < 3:
        # Fewer than 3 inner generations cannot show a search trajectory.
        raise ValidationConfigError("inner_generations_too_few")
    if individuals < 2:
        raise ValidationConfigError("individuals_too_few")
    if step_limit < 6:
        raise ValidationConfigError("step_limit_too_small")
    # The root consumes one journal step, and each evaluated candidate one.
    # Extra debug nodes need additional headroom; the global model ledger
    # remains the hard limit regardless of this local step allowance.
    if step_limit < 1 + num_generations * individuals:
        raise ValidationConfigError("step_limit_truncates_generations")
    if num_islands != 1:
        raise ValidationConfigError("num_islands_must_be_one")
    if not 0 <= gens_till_crossover <= num_generations - 1:
        # The crossover operator only activates once the inner generation
        # counter passes this threshold; if the threshold is never reached,
        # the crossover_prob search dim has no observable effect in the run.
        raise ValidationConfigError("crossover_threshold_unreachable")
    if llm_concurrency != 1:
        raise ValidationConfigError("llm_concurrency_must_be_one")
    if sandbox_concurrency != 1:
        raise ValidationConfigError("sandbox_concurrency_must_be_one")
    if not experience_enabled:
        # Disabling experience switches the evaluator to call the analyze
        # LLM for every candidate instead of using deterministic evaluator
        # metadata — an unbudgeted auxiliary model call path.
        raise ValidationConfigError("experience_must_be_enabled")
    if prompt_memory_enabled:
        # Prompt memory triggers auxiliary rich_memory_summary model calls;
        # the validation config disables it (and with it the prompt-memory
        # search dims; see research.search.search_vector.DROP_RATIONALE).
        raise ValidationConfigError("prompt_memory_must_be_disabled")
    if not stream_enabled:
        raise ValidationConfigError("stream_must_be_enabled")
    if not 1 <= max_output_tokens <= MAX_OUTPUT_TOKENS_LIMIT:
        raise ValidationConfigError("max_output_tokens_impractical")


@dataclass(frozen=True)
class InnerEvoValidationConfig:
    """Typed, self-validating view of the INNER Evo validation config.

    Field defaults mirror ``trustworthy_validation.yaml`` (3 INNER Evo solver
    generations x 2 individuals, step_limit 12). ``__post_init__`` delegates
    to :func:`validate_inner_evo_config`, so constructing an invalid config
    raises :class:`ValidationConfigError` naming the violated constraint.

    ``inner_num_generations`` is named explicitly to keep the INNER (Evo
    solver) vs OUTER (iStratDE over config vectors) generation distinction
    visible at every call site.
    """

    inner_num_generations: int = 3
    individuals_per_generation: int = 2
    step_limit: int = 12
    num_islands: int = 1
    num_generations_till_crossover: int = 1
    crossover_prob: float = 0.5
    max_debug_depth: int = 1
    llm_concurrency: int = 1
    sandbox_concurrency: int = 1
    experience_enabled: bool = True
    prompt_memory_enabled: bool = False
    stream_enabled: bool = True
    max_output_tokens: int = 4096
    estimated_input_tokens: int = 8000
    request_deadline_seconds: float = 600.0

    def __post_init__(self) -> None:
        validate_inner_evo_config(
            {
                "num_generations": self.inner_num_generations,
                "individuals_per_generation": self.individuals_per_generation,
                "step_limit": self.step_limit,
                "num_islands": self.num_islands,
                "num_generations_till_crossover": (
                    self.num_generations_till_crossover
                ),
                "llm_concurrency": self.llm_concurrency,
                "sandbox_concurrency": self.sandbox_concurrency,
                "experience_enabled": self.experience_enabled,
                "prompt_memory_enabled": self.prompt_memory_enabled,
                "stream": self.stream_enabled,
                "max_output_tokens": self.max_output_tokens,
            }
        )

    def as_transport_guard_config(self) -> dict[str, Any]:
        """Budget-guard parameters for one inner-run model request.

        Returned as a plain mapping so this module stays stdlib-only; callers
        in research.search.budget_transport unpack it into
        TransportGuardConfig.
        """
        return {
            "estimated_input_tokens": self.estimated_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "request_deadline_seconds": self.request_deadline_seconds,
        }


def outer_de_bounds() -> dict[str, Any]:
    """Documentation helper for the OUTER iStratDE loop over config vectors.

    The OUTER DE generations are distinct from the INNER Evo solver
    generations validated above: one outer fitness evaluation is one full
    inner Evo run. iStratDE minimizes fitness natively, so accuracy is mapped
    as ``fitness = -accuracy``.
    """
    return {
        "outer_algorithm": "istratde",
        "outer_pop_size_min": 4,
        "note": (
            "outer iStratDE generations are distinct from inner Evo "
            "generations; fitness minimized natively, map fitness=-accuracy"
        ),
    }
