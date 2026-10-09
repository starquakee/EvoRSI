"""Offline contract tests for research/search/search_vector.py (US-007).

Pins the retained/dropped dim split against the frozen legacy adapter
(research/adapters/safe_rsi_de_adapter.py — the record of the old 8-dim
experiment): retained dims are legacy indices 0, 1, 2, 3, 7 and dropped dims
are legacy indices 4, 5, 6, with identical bounds and hydra paths.
"""
from __future__ import annotations

import pytest

from research.adapters import safe_rsi_de_adapter as legacy
from research.search.search_vector import (
    DIMS,
    DROP_RATIONALE,
    DROPPED_DIMS,
    RETAINED_DIMS,
    SearchDim,
    by_name,
    decode_retained,
    hydra_overrides,
    mapping_explanation,
    safety_gate,
)

# Legacy 8-dim layout: [crossover, score, delta, novelty, cards, ancestors,
# siblings, debug_depth]
RETAINED_LEGACY_INDICES = (0, 1, 2, 3, 7)
DROPPED_LEGACY_INDICES = (4, 5, 6)


class TestDimSplitMatchesLegacy:
    def test_retained_contents_and_order(self):
        assert [d.name for d in RETAINED_DIMS] == [
            "crossover_prob",
            "score_weight",
            "delta_weight",
            "novelty_weight",
            "max_debug_depth",
        ]

    def test_dropped_contents_and_order(self):
        assert [d.name for d in DROPPED_DIMS] == [
            "max_related_cards",
            "ancestor_k",
            "sibling_k",
        ]

    def test_dims_is_retained_then_dropped(self):
        assert DIMS == RETAINED_DIMS + DROPPED_DIMS
        assert len(DIMS) == 8

    def test_bounds_match_legacy_indices(self):
        for dim, legacy_index in zip(RETAINED_DIMS, RETAINED_LEGACY_INDICES):
            assert dim.lower == pytest.approx(float(legacy.LOWER[legacy_index]))
            assert dim.upper == pytest.approx(float(legacy.UPPER[legacy_index]))
        for dim, legacy_index in zip(DROPPED_DIMS, DROPPED_LEGACY_INDICES):
            assert dim.lower == pytest.approx(float(legacy.LOWER[legacy_index]))
            assert dim.upper == pytest.approx(float(legacy.UPPER[legacy_index]))

    def test_hydra_paths_match_legacy_overrides(self):
        legacy_cfg = legacy.decode([0.5, 1.0, 0.4, 0.25, 3, 2, 2, 4])
        legacy_paths = {
            override.split("=")[0] for override in legacy.hydra_overrides(legacy_cfg)
        }
        for dim in DIMS:
            assert dim.hydra_path in legacy_paths

    def test_retained_dims_have_effects_dropped_do_not(self):
        for dim in RETAINED_DIMS:
            assert dim.effect, dim.name
        for dim in DROPPED_DIMS:
            assert dim.effect == ""

    def test_by_name(self):
        assert by_name("crossover_prob") is RETAINED_DIMS[0]
        assert by_name("sibling_k") is DROPPED_DIMS[2]
        with pytest.raises(KeyError):
            by_name("nope")
        assert isinstance(by_name("max_debug_depth"), SearchDim)


class TestDecodeRetained:
    def test_decode_matches_legacy_for_shared_dims(self):
        legacy_cfg = legacy.decode([0.5, 1.0, 0.4, 0.25, 3, 2, 2, 4])
        decoded = decode_retained([0.5, 1.0, 0.4, 0.25, 4])
        assert decoded == {
            "crossover_prob": pytest.approx(legacy_cfg.crossover_prob),
            "score_weight": pytest.approx(legacy_cfg.score_weight),
            "delta_weight": pytest.approx(legacy_cfg.delta_weight),
            "novelty_weight": pytest.approx(legacy_cfg.novelty_weight),
            "max_debug_depth": legacy_cfg.max_debug_depth,
        }

    def test_clip_to_bounds(self):
        decoded = decode_retained([1e9, -1e9, 1e9, -1e9, 1e9])
        assert decoded["crossover_prob"] == pytest.approx(0.95)
        assert decoded["score_weight"] == 0.0
        assert decoded["delta_weight"] == pytest.approx(2.0)
        assert decoded["novelty_weight"] == 0.0
        assert decoded["max_debug_depth"] == 6
        decoded = decode_retained([-1e9, 0.5, 0.5, 0.5, -1e9])
        assert decoded["crossover_prob"] == pytest.approx(0.05)
        assert decoded["max_debug_depth"] == 0

    def test_int_dim_rounds_half_even_like_legacy_rint(self):
        assert decode_retained([0.5, 1.0, 0.4, 0.25, 3.6])["max_debug_depth"] == 4
        assert decode_retained([0.5, 1.0, 0.4, 0.25, 1.4])["max_debug_depth"] == 1
        # rint(2.5) == 2 (round-half-even), matching the legacy decode
        assert decode_retained([0.5, 1.0, 0.4, 0.25, 2.5])["max_debug_depth"] == 2
        assert decode_retained([0.5, 1.0, 0.4, 0.25, 3.5])["max_debug_depth"] == 4

    @pytest.mark.parametrize("length", [0, 4, 6, 8])
    def test_vector_length_mismatch(self, length):
        with pytest.raises(ValueError, match="vector_length_mismatch"):
            decode_retained([0.5] * length)


def _decoded(**overrides):
    base = {
        "crossover_prob": 0.5,
        "score_weight": 1.0,
        "delta_weight": 0.4,
        "novelty_weight": 0.25,
        "max_debug_depth": 4,
    }
    base.update(overrides)
    return base


class TestSafetyGate:
    def test_zero_selection_weights_rejected(self):
        with pytest.raises(ValueError, match="empty_parent_selection_weights"):
            safety_gate(_decoded(score_weight=0.0, delta_weight=0.0, novelty_weight=0.0))

    def test_debug_depth_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="max_debug_depth_out_of_range"):
            safety_gate(_decoded(max_debug_depth=7))

    def test_valid_config_passes(self):
        assert safety_gate(_decoded()) is None

    def test_debug_depth_boundary(self):
        assert safety_gate(_decoded(max_debug_depth=6)) is None


class TestHydraOverrides:
    def test_canonical_format_and_order(self):
        overrides = hydra_overrides(_decoded())
        assert overrides == [
            "search.runner.solver.crossover_prob=0.500000",
            "search.runner.solver.experience.parent_selection.weights.score=1.000000",
            "search.runner.solver.experience.parent_selection.weights.delta=0.400000",
            "search.runner.solver.experience.parent_selection.weights.novelty=0.250000",
            "search.runner.solver.max_debug_depth=4",
        ]

    def test_shared_dim_overrides_match_legacy_strings(self):
        legacy_cfg = legacy.decode([0.5, 1.0, 0.4, 0.25, 3, 2, 2, 4])
        legacy_overrides = set(legacy.hydra_overrides(legacy_cfg))
        for override in hydra_overrides(_decoded()):
            assert override in legacy_overrides

    def test_gate_runs_first(self):
        with pytest.raises(ValueError, match="empty_parent_selection_weights"):
            hydra_overrides(
                _decoded(score_weight=0.0, delta_weight=0.0, novelty_weight=0.0)
            )


class TestMappingExplanation:
    def test_structure_and_rationale(self):
        explanation = mapping_explanation()
        assert set(explanation["retained"]) == {d.name for d in RETAINED_DIMS}
        assert set(explanation["dropped"]) == {d.name for d in DROPPED_DIMS}
        for name, entry in explanation["dropped"].items():
            assert entry["rationale"] == DROP_RATIONALE, name
        for name, entry in explanation["retained"].items():
            assert entry["effect"], name
            assert entry["hydra_path"].startswith("search.runner.solver")
        assert "INNER" in explanation["inner_vs_outer"]
        assert "OUTER" in explanation["inner_vs_outer"]

    def test_drop_rationale_mentions_prompt_memory_and_model_calls(self):
        assert "prompt_memory" in DROP_RATIONALE
        assert "rich_memory_summary" in DROP_RATIONALE
