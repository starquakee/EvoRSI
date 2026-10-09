"""Offline contract tests for research/search/validation_config.py (US-007).

Pins the machine-readable constraint names the Evo-side config check relies
on, the INNER-vs-OUTER generation terminology, and the step-limit boundary
that guarantees every INNER Evo generation completes.
"""
from __future__ import annotations

import pytest

from research.search.validation_config import (
    InnerEvoValidationConfig,
    ValidationConfigError,
    outer_de_bounds,
    validate_inner_evo_config,
)

DEFAULTS = {
    "num_generations": 3,
    "individuals_per_generation": 2,
    "step_limit": 12,
    "num_islands": 1,
    "num_generations_till_crossover": 1,
    "llm_concurrency": 1,
    "sandbox_concurrency": 1,
    "experience_enabled": True,
    "prompt_memory_enabled": False,
    "stream": True,
    "max_output_tokens": 4096,
}


def _config(**overrides):
    merged = {**DEFAULTS, **overrides}
    return merged


class TestConstraintMatrix:
    @pytest.mark.parametrize(
        ("overrides", "constraint"),
        [
            ({"num_generations": 2}, "inner_generations_too_few"),
            ({"num_generations": 0}, "inner_generations_too_few"),
            ({"individuals_per_generation": 1}, "individuals_too_few"),
            ({"step_limit": 5}, "step_limit_too_small"),
            # One root plus 3 x 2 candidates needs seven journal steps.
            ({"step_limit": 6}, "step_limit_truncates_generations"),
            ({"num_generations": 4, "step_limit": 8}, "step_limit_truncates_generations"),
            ({"num_islands": 2}, "num_islands_must_be_one"),
            ({"num_islands": 0}, "num_islands_must_be_one"),
            # threshold 3 is never reached within 3 generations (needs <= 2)
            ({"num_generations_till_crossover": 3}, "crossover_threshold_unreachable"),
            ({"llm_concurrency": 2}, "llm_concurrency_must_be_one"),
            ({"sandbox_concurrency": 0}, "sandbox_concurrency_must_be_one"),
            ({"experience_enabled": False}, "experience_must_be_enabled"),
            ({"prompt_memory_enabled": True}, "prompt_memory_must_be_disabled"),
            ({"stream": False}, "stream_must_be_enabled"),
            ({"max_output_tokens": 0}, "max_output_tokens_impractical"),
            ({"max_output_tokens": 8193}, "max_output_tokens_impractical"),
            ({"max_output_tokens": -1}, "max_output_tokens_impractical"),
        ],
    )
    def test_violation_raises_named_constraint(self, overrides, constraint):
        with pytest.raises(ValidationConfigError, match=constraint):
            validate_inner_evo_config(_config(**overrides))

    def test_error_message_is_exactly_the_constraint_name(self):
        with pytest.raises(ValidationConfigError) as excinfo:
            validate_inner_evo_config(_config(num_islands=3))
        assert str(excinfo.value) == "num_islands_must_be_one"

    def test_error_is_a_value_error(self):
        with pytest.raises(ValueError):
            validate_inner_evo_config(_config(stream=False))


class TestDefaultsAndAcceptedKeys:
    def test_defaults_pass(self):
        validate_inner_evo_config(_config())

    def test_empty_mapping_uses_safe_defaults(self):
        validate_inner_evo_config({})

    def test_partial_mapping_passes(self):
        validate_inner_evo_config({"num_generations": 5, "step_limit": 20})

    def test_unknown_keys_are_ignored(self):
        # Documented behavior: keys outside the accepted set (e.g.
        # crossover_prob, max_debug_depth, sandbox.image) are ignored so the
        # validator stays forward-compatible with extra experiment keys.
        validate_inner_evo_config(
            _config(crossover_prob=0.9, max_debug_depth=6, **{"sandbox.image": "x"})
        )

    def test_non_mapping_rejected(self):
        with pytest.raises(ValidationConfigError, match="config_must_be_a_mapping"):
            validate_inner_evo_config([("num_generations", 3)])


class TestStepLimitBoundary:
    def test_6_fails_for_3x2(self):
        with pytest.raises(
            ValidationConfigError, match="step_limit_truncates_generations"
        ):
            validate_inner_evo_config(_config(step_limit=6))

    def test_7_passes_for_3x2(self):
        validate_inner_evo_config(_config(step_limit=7))

    def test_boundary_scales_with_generations_and_individuals(self):
        # One root plus 4 x 3 candidates -> 13 steps required
        validate_inner_evo_config(
            _config(num_generations=4, individuals_per_generation=3, step_limit=13)
        )
        with pytest.raises(
            ValidationConfigError, match="step_limit_truncates_generations"
        ):
            validate_inner_evo_config(
                _config(
                    num_generations=4, individuals_per_generation=3, step_limit=12
                )
            )


class TestCrossoverThresholdBoundary:
    def test_threshold_at_num_generations_minus_one_passes(self):
        validate_inner_evo_config(_config(num_generations_till_crossover=2))

    def test_threshold_at_num_generations_fails(self):
        with pytest.raises(
            ValidationConfigError, match="crossover_threshold_unreachable"
        ):
            validate_inner_evo_config(_config(num_generations_till_crossover=3))


class TestInnerEvoValidationConfig:
    def test_defaults_round_trip(self):
        cfg = InnerEvoValidationConfig()
        assert cfg.inner_num_generations == 3
        assert cfg.individuals_per_generation == 2
        assert cfg.step_limit == 12
        assert cfg.num_islands == 1
        assert cfg.crossover_prob == 0.5
        assert cfg.max_debug_depth == 1
        assert cfg.experience_enabled is True
        assert cfg.prompt_memory_enabled is False
        assert cfg.stream_enabled is True
        assert cfg.max_output_tokens == 4096

    def test_defaults_match_trustworthy_validation_yaml(self):
        # The Evo-side experiment config (trustworthy_validation.yaml) uses
        # exactly these values; constructing the typed config must succeed.
        InnerEvoValidationConfig(
            inner_num_generations=3,
            individuals_per_generation=2,
            step_limit=12,
            num_islands=1,
            num_generations_till_crossover=1,
            crossover_prob=0.5,
            max_debug_depth=1,
            llm_concurrency=1,
            sandbox_concurrency=1,
            experience_enabled=True,
            prompt_memory_enabled=False,
        )

    def test_invalid_field_raises_named_constraint(self):
        with pytest.raises(
            ValidationConfigError, match="prompt_memory_must_be_disabled"
        ):
            InnerEvoValidationConfig(prompt_memory_enabled=True)
        with pytest.raises(
            ValidationConfigError, match="step_limit_truncates_generations"
        ):
            InnerEvoValidationConfig(step_limit=6)

    def test_transport_guard_config_mapping(self):
        cfg = InnerEvoValidationConfig()
        guard = cfg.as_transport_guard_config()
        assert guard == {
            "estimated_input_tokens": 8000,
            "max_output_tokens": 4096,
            "request_deadline_seconds": 600.0,
        }


class TestOuterDeBounds:
    def test_content(self):
        bounds = outer_de_bounds()
        assert bounds["outer_algorithm"] == "istratde"
        assert bounds["outer_pop_size_min"] == 4
        assert "fitness=-accuracy" in bounds["note"]
        assert "inner Evo generations" in bounds["note"]
