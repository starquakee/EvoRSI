"""Offline tests for research/comparison/phase.py (EXP-007).

The server-side phase switch is scripted through module seams — no docker,
no stack contact. Every failure must call the durable failure sink BEFORE
raising; the already-final path is accepted only with a matching registry
hash; the original deadline bounds activation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from research.comparison import phase
from research.comparison.phase import PhaseActivationError

REPO_ROOT = Path(__file__).resolve().parents[2]
SHA = "c" * 64


@pytest.fixture()
def scripted(monkeypatch: pytest.MonkeyPatch):
    state = {"phase": phase.PHASE_SEARCH, "sha": SHA, "compose": [], "idle": True}
    monkeypatch.setattr(
        phase,
        "verify_idle",
        lambda *a, **k: (_ for _ in ()).throw(PhaseActivationError("not_idle"))
        if not state["idle"]
        else None,
    )
    monkeypatch.setattr(phase, "current_phase", lambda *a, **k: state["phase"])
    monkeypatch.setattr(phase, "_wait_dispatcher_running", lambda *a, **k: None)

    def _compose(repo_root, registry_path, **kwargs):
        state["compose"].append(registry_path)
        # Simulate the recreate actually applying the env override.
        state["phase"] = (
            phase.PHASE_FINAL
            if registry_path == phase.FINAL_REGISTRY_CONTAINER_PATH
            else phase.PHASE_SEARCH
        )

    monkeypatch.setattr(phase, "_compose_recreate_dispatcher", _compose)
    monkeypatch.setattr(
        phase, "_container_registry_sha256", lambda *a, **k: state["sha"]
    )
    return state


def _activate(state, **overrides):
    failures: list[str] = []
    kwargs = {
        "expected_registry_sha256": SHA,
        "deadline_epoch": 2000.0,
        "now_fn": lambda: 1000.0,
        "on_failure": failures.append,
    }
    kwargs.update(overrides)
    result = phase.activate_final_phase(REPO_ROOT, **kwargs)
    return result, failures


class TestActivateFinalPhase:
    def test_switches_from_search(self, scripted):
        result, failures = _activate(scripted)
        assert result == "activated_final"
        assert failures == []
        assert scripted["compose"] == [phase.FINAL_REGISTRY_CONTAINER_PATH]

    def test_already_final_requires_matching_hash(self, scripted):
        scripted["phase"] = phase.PHASE_FINAL
        result, failures = _activate(scripted)
        assert result == "already_final"
        assert scripted["compose"] == []  # no recreate

    def test_already_final_hash_mismatch_stops(self, scripted):
        scripted["phase"] = phase.PHASE_FINAL
        scripted["sha"] = "d" * 64
        failures: list[str] = []
        with pytest.raises(PhaseActivationError, match="phase_registry_hash_mismatch"):
            phase.activate_final_phase(
                REPO_ROOT,
                expected_registry_sha256=SHA,
                deadline_epoch=2000.0,
                now_fn=lambda: 1000.0,
                on_failure=failures.append,
            )
        # the durable failure sink fired BEFORE the raise
        assert failures == ["phase_registry_hash_mismatch"]

    def test_deadline_expired_stops_without_compose(self, scripted):
        failures: list[str] = []
        with pytest.raises(
            PhaseActivationError, match="phase_activation_deadline_expired"
        ):
            phase.activate_final_phase(
                REPO_ROOT,
                expected_registry_sha256=SHA,
                deadline_epoch=999.0,
                now_fn=lambda: 1000.0,
                on_failure=failures.append,
            )
        assert failures == ["phase_activation_deadline_expired"]
        assert scripted["compose"] == []

    def test_not_idle_stops(self, scripted):
        scripted["idle"] = False
        failures: list[str] = []
        with pytest.raises(PhaseActivationError, match="phase_precondition_failed"):
            phase.activate_final_phase(
                REPO_ROOT,
                expected_registry_sha256=SHA,
                deadline_epoch=2000.0,
                now_fn=lambda: 1000.0,
                on_failure=failures.append,
            )
        assert failures and failures[0].startswith("phase_precondition_failed:not_idle")
        assert scripted["compose"] == []

    def test_switch_but_env_not_updated_stops(
        self, scripted, monkeypatch: pytest.MonkeyPatch
    ):
        # Compose "succeeds" but the dispatcher stays in search phase.
        monkeypatch.setattr(
            phase,
            "_compose_recreate_dispatcher",
            lambda repo_root, path, **kw: None,
        )
        failures: list[str] = []
        with pytest.raises(PhaseActivationError, match="phase_env_mismatch"):
            phase.activate_final_phase(
                REPO_ROOT,
                expected_registry_sha256=SHA,
                deadline_epoch=2000.0,
                now_fn=lambda: 1000.0,
                on_failure=failures.append,
            )
        assert failures == ["phase_env_mismatch_after_recreate"]

    def test_failure_sink_precedes_raise_on_switch_error(
        self, scripted, monkeypatch: pytest.MonkeyPatch
    ):
        def _bad_compose(repo_root, registry_path, **kwargs):
            raise PhaseActivationError("dispatcher_recreate_failed:boom")

        monkeypatch.setattr(phase, "_compose_recreate_dispatcher", _bad_compose)
        failures: list[str] = []
        with pytest.raises(PhaseActivationError, match="phase_switch_failed"):
            phase.activate_final_phase(
                REPO_ROOT,
                expected_registry_sha256=SHA,
                deadline_epoch=2000.0,
                now_fn=lambda: 1000.0,
                on_failure=failures.append,
            )
        assert failures == ["phase_switch_failed:dispatcher_recreate_failed:boom"]


class TestRestoreSearchPhase:
    def test_restore_from_final(self, scripted):
        scripted["phase"] = phase.PHASE_FINAL
        failures: list[str] = []
        result = phase.restore_search_phase(REPO_ROOT, on_failure=failures.append)
        assert result == "activated_search"
        assert failures == []
        assert scripted["compose"] == [phase.SEARCH_REGISTRY_CONTAINER_PATH]

    def test_restore_idempotent_in_search(self, scripted):
        result = phase.restore_search_phase(REPO_ROOT, on_failure=lambda r: None)
        assert result == "already_search"
        assert scripted["compose"] == []
