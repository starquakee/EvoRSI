"""Offline contract tests for research/adapters (US-001).

No network or sandbox access: submission is monkeypatched/spied. These tests
pin the safety-gate and utility-gating behavior that later stories rely on.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from research.adapters import rsi_eval_adapter
from research.adapters.rsi_eval_adapter import EvalResult, evaluate_candidate
from research.adapters.safe_rsi_de_adapter import (
    LOWER,
    UPPER,
    StrategyConfig,
    decode,
    decode_batch,
    hydra_overrides,
    safety_gate,
)
from research.adapters.sandbox_eval_client import (
    FORBIDDEN_MARKERS,
    SandboxResult,
    safety_gate_source,
    submit_and_wait,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENIGN_SOURCE = (REPO_ROOT / "research/experiments/hello_synth_job.py").read_text()


class TestSourceGate:
    def test_benign_ml_code_passes(self):
        ok, reason = safety_gate_source(BENIGN_SOURCE)
        assert ok, reason
        assert reason == "ok"

    def test_gate_delegates_to_versioned_policy(self):
        from research.contracts.source_gate import load_policy

        policy = load_policy()
        assert tuple(FORBIDDEN_MARKERS) == tuple(policy.denied_source_markers)
        assert policy.policy_version == "source-gate.v2"

    @pytest.mark.parametrize("marker", FORBIDDEN_MARKERS)
    def test_every_marker_rejected(self, marker: str):
        ok, reason = safety_gate_source(f"payload = {marker!r}\n{marker}do_thing()\n")
        assert not ok
        assert f"forbidden_marker:{marker}" in reason

    def test_marker_matching_is_case_insensitive(self):
        ok, reason = safety_gate_source("import OS\nOS.SYSTEM('id')\n")
        assert not ok
        assert "os.system" in reason

    def test_missing_policy_fails_closed(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SOURCE_GATE_POLICY_PATH", str(tmp_path / "missing.json"))
        ok, reason = safety_gate_source(BENIGN_SOURCE)
        assert not ok
        assert "policy_unavailable" in reason

    def test_submit_refuses_before_network(self, monkeypatch):
        def _boom(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("network client constructed for rejected source")

        monkeypatch.setattr("httpx.Client", _boom)
        with pytest.raises(ValueError, match="forbidden_marker"):
            submit_and_wait("import subprocess\n")

    def test_submit_refuses_unsafe_data_dir_before_network(self, monkeypatch):
        def _boom(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("network client constructed for unsafe data_dir")

        monkeypatch.setattr("httpx.Client", _boom)
        with pytest.raises(ValueError, match="path_traversal|path_protected_component"):
            submit_and_wait(BENIGN_SOURCE, data_dir="/data/../answers")


class TestEvaluateCandidate:
    def test_rejected_source_never_submitted(self, monkeypatch):
        def _boom(*args, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("submit called for rejected source")

        monkeypatch.setattr(rsi_eval_adapter, "submit_and_wait", _boom)
        result = evaluate_candidate("import subprocess\n")
        assert result.status == "rejected"
        assert result.job_id is None
        assert not result.safety_ok
        assert result.utility == -math.inf
        assert result.elapsed_s == 0.0

    def test_submit_exception_maps_to_error(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(rsi_eval_adapter, "submit_and_wait", _raise)
        result = evaluate_candidate(BENIGN_SOURCE)
        assert result.status == "error"
        assert result.safety_ok
        assert "submit_error:RuntimeError" in result.safety_reason
        assert result.utility == -math.inf

    def test_unscored_job_is_negative_infinity(self, monkeypatch):
        monkeypatch.setattr(
            rsi_eval_adapter, "submit_and_wait",
            lambda *a, **k: SandboxResult("job-1", "failed", None, {"status": "failed"}),
        )
        result = evaluate_candidate(BENIGN_SOURCE)
        assert result.status == "failed"
        assert result.job_id == "job-1"
        assert result.utility == -math.inf
        assert result.safety_reason == "job_not_scored"

    def test_scored_job_utility(self, monkeypatch):
        monkeypatch.setattr(
            rsi_eval_adapter, "submit_and_wait",
            lambda *a, **k: SandboxResult("job-2", "completed", 0.9, {"status": "completed"}),
        )
        result = evaluate_candidate(BENIGN_SOURCE, cost_weight=0.0)
        assert result.status == "completed"
        assert result.capability_score == pytest.approx(0.9)
        assert result.utility == pytest.approx(0.9)
        assert result.safety_reason == "ok"


class TestStrategyAdapter:
    def test_decode_clamps_to_bounds(self):
        cfg = decode([1e9] * 8)
        assert cfg.crossover_prob == pytest.approx(float(UPPER[0]))
        assert cfg.max_related_cards == int(UPPER[4])
        cfg = decode([-1e9] * 8)
        assert cfg.crossover_prob == pytest.approx(float(LOWER[0]))
        assert cfg.max_related_cards == int(LOWER[4])

    def test_decode_rounds_integer_dims(self):
        cfg = decode([0.5, 1.0, 0.4, 0.25, 2.6, 1.4, 2.5, 3.6])
        assert cfg.max_related_cards == 3
        assert cfg.ancestor_k == 1
        assert cfg.max_debug_depth == 4

    def test_gate_rejects_zero_selection_weights(self):
        cfg = StrategyConfig(0.5, 0.0, 0.0, 0.0, 3, 2, 2, 4)
        ok, reason = safety_gate(cfg)
        assert not ok
        assert reason == "empty_parent_selection_weights"

    def test_gate_rejects_excess_debug_depth(self):
        cfg = StrategyConfig(0.5, 1.0, 0.0, 0.0, 3, 2, 2, 7)
        ok, reason = safety_gate(cfg)
        assert not ok
        assert reason == "debug_depth_limit"

    def test_gate_rejects_excess_memory_cards(self):
        cfg = StrategyConfig(0.5, 1.0, 0.0, 0.0, 9, 2, 2, 4)
        ok, reason = safety_gate(cfg)
        assert not ok
        assert reason == "memory_context_limit"

    def test_hydra_overrides_shape_and_gate(self):
        cfg = decode([0.5, 1.0, 0.4, 0.25, 3, 2, 2, 4])
        overrides = hydra_overrides(cfg)
        assert len(overrides) == 8
        assert overrides[0] == "search.runner.solver.crossover_prob=0.500000"
        assert "max_debug_depth=4" in overrides[1]
        unsafe = StrategyConfig(0.5, 0.0, 0.0, 0.0, 3, 2, 2, 4)
        with pytest.raises(ValueError, match="empty_parent_selection_weights"):
            hydra_overrides(unsafe)

    def test_decode_batch(self):
        cfgs = decode_batch([[0.5, 1.0, 0.4, 0.25, 3, 2, 2, 4]] * 3)
        assert len(cfgs) == 3
        assert all(isinstance(c, StrategyConfig) for c in cfgs)
