"""Offline contract tests for research/comparison/protocol.py (EXP-001).

Pins the frozen approved values (27 cells / 216 inner runs, budget
arithmetic, FIXED theta, K3 model profile), the machine-readable
constraint names, and the freeze behavior: the protocol type refuses any
deviation from the approved design.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from research.comparison.protocol import (
    COMPARISON_SCHEMA,
    COMPARISON_DIMS,
    CONFIGS_PER_CELL,
    FIXED_THETA,
    INNER_GENERATIONS,
    MAIN_CANDIDATES_PER_GENERATION,
    METHODS,
    NUM_CELLS,
    NUM_INNER_RUNS,
    SEARCH_SEEDS,
    SPLIT_SEED_BASE,
    TASKS,
    TASKS_BY_ID,
    CellBudget,
    ComparisonProtocol,
    ComparisonSpecError,
    GlobalBudget,
    InnerBudget,
    ModelProfile,
    TaskPin,
    canonical_protocol,
    decode_theta,
    split_seed,
    theta_gate,
)


class TestApprovedPins:
    def test_tasks_match_prd_order_and_shapes(self):
        assert [(t.task_id, t.uci_id, t.rows, t.features) for t in TASKS] == [
            ("banknote", 267, 1372, 4),
            ("wdbc", 17, 569, 30),
            ("spambase", 94, 4601, 57),
        ]

    def test_split_seeds_are_base_plus_uci_id(self):
        assert split_seed(TASKS_BY_ID["banknote"]) == 20261010 + 267
        assert split_seed(TASKS_BY_ID["wdbc"]) == 20261010 + 17
        assert split_seed(TASKS_BY_ID["spambase"]) == 20261010 + 94
        assert SPLIT_SEED_BASE == 20261010

    def test_methods_seeds_and_counts(self):
        assert METHODS == ("FIXED", "RANDOM", "ISTRATDE")
        assert SEARCH_SEEDS == (11, 22, 33)
        assert CONFIGS_PER_CELL == 8
        assert NUM_CELLS == 27
        assert NUM_INNER_RUNS == 216
        assert INNER_GENERATIONS == 3
        assert MAIN_CANDIDATES_PER_GENERATION == 2

    def test_budget_arithmetic_is_closed(self):
        inner = InnerBudget()
        cell = CellBudget()
        glob = GlobalBudget()
        assert inner.max_main_requests == INNER_GENERATIONS * (
            MAIN_CANDIDATES_PER_GENERATION
        )
        assert CONFIGS_PER_CELL * (
            inner.max_main_requests + inner.max_debug_requests
        ) == cell.max_requests == 64
        assert NUM_CELLS * cell.max_requests == glob.max_requests == 1728
        assert cell.max_tokens == 200_000
        assert glob.max_tokens == 5_400_000
        assert glob.max_elapsed_seconds == 86_400.0

    def test_comparison_dims_bounds(self):
        assert [d.name for d in COMPARISON_DIMS] == [
            "crossover_prob",
            "score_weight",
            "delta_weight",
            "novelty_weight",
            "max_debug_depth",
        ]
        bounds = {d.name: (d.lower, d.upper, d.kind) for d in COMPARISON_DIMS}
        assert bounds["crossover_prob"] == (0.05, 0.95, "float")
        assert bounds["score_weight"] == (0.0, 2.0, "float")
        assert bounds["delta_weight"] == (0.0, 2.0, "float")
        assert bounds["novelty_weight"] == (0.0, 2.0, "float")
        # Tighter than the US-007 acceptance bound [0, 6]: the comparison
        # caps debug at 2 (PRD: at most two debug requests per inner run).
        assert bounds["max_debug_depth"] == (0.0, 2.0, "int")


class TestTheta:
    def test_fixed_theta_decodes_to_approved_values(self):
        decoded = decode_theta(FIXED_THETA)
        assert decoded == {
            "crossover_prob": 0.5,
            "score_weight": 1.0,
            "delta_weight": 0.4,
            "novelty_weight": 0.25,
            "max_debug_depth": 1,
        }
        theta_gate(decoded)

    def test_length_mismatch(self):
        with pytest.raises(ComparisonSpecError, match="theta_length_mismatch"):
            decode_theta([0.5, 1.0])

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0.5"])
    def test_nonfinite_or_invalid_element(self, bad):
        with pytest.raises(
            ComparisonSpecError, match="nonfinite_or_invalid_theta"
        ):
            decode_theta([bad, 1.0, 0.4, 0.25, 1.0])

    def test_weight_sum_zero_rejected(self):
        decoded = decode_theta([0.5, 0.0, 0.0, 0.0, 1.0])
        with pytest.raises(
            ComparisonSpecError, match="empty_parent_selection_weights"
        ):
            theta_gate(decoded)

    def test_decode_clips_and_rounds_consistently(self):
        # Same encoding/rounding as the US-007 acceptance vector: floats are
        # clipped to bounds, the int dim rounds half-even after clipping.
        decoded = decode_theta([2.0, 3.0, -1.0, 2.0, 5.0])
        assert decoded == {
            "crossover_prob": 0.95,
            "score_weight": 2.0,
            "delta_weight": 0.0,
            "novelty_weight": 2.0,
            "max_debug_depth": 2,
        }
        theta_gate(decoded)

    def test_gate_rejects_post_decode_range_violation(self):
        with pytest.raises(
            ComparisonSpecError, match="theta_dimension_out_of_range"
        ):
            theta_gate(
                {
                    "crossover_prob": 0.5,
                    "score_weight": 2.5,
                    "delta_weight": 0.4,
                    "novelty_weight": 0.25,
                    "max_debug_depth": 1,
                }
            )
        with pytest.raises(
            ComparisonSpecError, match="max_debug_depth_out_of_range"
        ):
            theta_gate(
                {
                    "crossover_prob": 0.5,
                    "score_weight": 1.0,
                    "delta_weight": 0.4,
                    "novelty_weight": 0.25,
                    "max_debug_depth": 3,
                }
            )


class TestModelProfile:
    def test_defaults_are_the_approved_pin(self):
        profile = ModelProfile()
        assert profile.model_id == "k3"
        assert profile.base_url == "https://api.kimi.com/coding/v1"
        assert profile.reasoning_effort == "low"
        assert profile.max_output_tokens == 8192
        assert profile.temperature == 1.0
        assert profile.top_p == 0.95
        assert profile.stream is True
        assert profile.allowed_openai_params == ("reasoning_effort",)
        assert profile.transport_retries == 0

    @pytest.mark.parametrize(
        ("overrides", "constraint"),
        [
            ({"model_id": ""}, "model_id_invalid"),
            ({"model_id": 7}, "model_id_invalid"),
            ({"base_url": "http://example.com"}, "model_base_url_invalid"),
            ({"reasoning_effort": "high"}, "reasoning_effort_must_be_low"),
            ({"max_output_tokens": 0}, "max_output_tokens_impractical"),
            ({"max_output_tokens": 8193}, "max_output_tokens_impractical"),
            ({"max_output_tokens": 1.5}, "max_output_tokens_impractical"),
            ({"temperature": 0.0}, "temperature_out_of_range"),
            ({"temperature": float("nan")}, "temperature_out_of_range"),
            ({"temperature": True}, "temperature_out_of_range"),
            ({"top_p": 0.0}, "top_p_out_of_range"),
            ({"top_p": 1.5}, "top_p_out_of_range"),
            ({"stream": False}, "stream_must_be_enabled"),
            ({"stream": 1}, "stream_must_be_enabled"),
            (
                {"allowed_openai_params": ("temperature",)},
                "allowed_openai_params_invalid",
            ),
            (
                {"allowed_openai_params": ["reasoning_effort"]},
                "allowed_openai_params_invalid",
            ),
            ({"transport_retries": -1}, "transport_retries_invalid"),
            ({"transport_retries": 0.5}, "transport_retries_invalid"),
        ],
    )
    def test_invalid_profile_raises_named_constraint(self, overrides, constraint):
        with pytest.raises(ComparisonSpecError, match=constraint):
            ModelProfile(**overrides)

    def test_identity_changes_with_any_field(self):
        pin = ModelProfile().identity()
        assert ModelProfile(temperature=0.5).identity() != pin
        assert ModelProfile(transport_retries=1).identity() != pin

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            profile = ModelProfile()
            profile.temperature = 0.5


class TestFrozenProtocol:
    def test_canonical_protocol_constructs(self):
        proto = canonical_protocol()
        assert proto.schema == COMPARISON_SCHEMA
        assert proto.num_cells == 27
        assert proto.num_inner_runs == 216

    @pytest.mark.parametrize(
        ("overrides", "constraint"),
        [
            ({"schema": "other.v1"}, "protocol_frozen:schema"),
            ({"tasks": TASKS[:2]}, "protocol_frozen:tasks"),
            (
                {"tasks": (TaskPin("banknote", 267, 1372, 5),) + TASKS[1:]},
                "protocol_frozen:tasks",
            ),
            ({"methods": ("FIXED", "RANDOM")}, "protocol_frozen:methods"),
            ({"search_seeds": (11, 22, 34)}, "protocol_frozen:search_seeds"),
            ({"search_seeds": [11, 22, 33]}, "protocol_field_not_an_int:search_seeds"),
            ({"configs_per_cell": 7}, "protocol_frozen:configs_per_cell"),
            ({"inner_generations": 4}, "protocol_frozen:inner_generations"),
            (
                {"main_candidates_per_generation": 3},
                "protocol_frozen:main_candidates_per_generation",
            ),
            (
                {"fixed_theta": (0.5, 1.0, 0.4, 0.25, 2.0)},
                "protocol_frozen:fixed_theta",
            ),
            (
                {"fixed_theta": (0.5, 1.0, 0.4, 0.25, True)},
                "nonfinite_or_invalid_theta",
            ),
            (
                {"inner_budget": InnerBudget(max_main_requests=5)},
                "protocol_frozen:inner_budget",
            ),
            (
                {"cell_budget": CellBudget(max_requests=63)},
                "protocol_frozen:cell_budget",
            ),
            (
                {"global_budget": GlobalBudget(max_requests=1729)},
                "protocol_frozen:global_budget",
            ),
            (
                {"model_profile": ModelProfile(temperature=0.5)},
                "protocol_frozen:model_profile",
            ),
            (
                {"split_seed_base": 20261011},
                "protocol_frozen:split_seed_base",
            ),
            # Float stand-ins compare equal to pins but are rejected on type.
            (
                {"configs_per_cell": 8.0},
                "protocol_field_not_an_int:configs_per_cell",
            ),
            (
                {"search_seeds": (11.0, 22.0, 33.0)},
                "protocol_field_not_an_int:search_seeds",
            ),
            # A caller-owned list would stay mutable after validation.
            (
                {"fixed_theta": [0.5, 1.0, 0.4, 0.25, 1.0]},
                "fixed_theta_must_be_a_tuple",
            ),
        ],
    )
    def test_deviation_refused_with_named_rule(self, overrides, constraint):
        with pytest.raises(ComparisonSpecError, match=constraint):
            ComparisonProtocol(**overrides)

    def test_identity_is_stable_and_canonical(self):
        first = canonical_protocol().identity()
        second = canonical_protocol().identity()
        assert first == second
        assert len(first) == 64
        int(first, 16)  # lowercase hex

    def test_to_dict_is_json_serializable_canonical(self):
        payload = canonical_protocol().to_dict()
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        assert json.loads(rendered) == payload
        assert payload["global_budget"] == {
            "max_requests": 1728,
            "max_tokens": 5_400_000,
            "max_elapsed_seconds": 86_400.0,
        }
        # The dim bounds (incl. the tighter debug cap) enter the identity.
        dims = {d["name"]: d for d in payload["dims"]}
        assert dims["max_debug_depth"]["upper"] == 2.0
