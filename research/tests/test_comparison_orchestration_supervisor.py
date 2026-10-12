"""Supervisor regressions for real ledger handoff, resume, and final release."""
import json
from pathlib import Path
from typing import Sequence

import pytest

from research.comparison import deploy
from research.comparison.budget import ComparisonLedger, ComparisonLedgerLocked, ProviderUsage
from research.comparison.orchestrate import Orchestrator, OrchestrationError
from research.comparison.specs import iter_cells
from research.tests.test_comparison_specs import COMMIT, make_bindings


class DeterministicNative:
    def setup_population(self, seed: int) -> tuple[tuple[float, ...], ...]:
        return self.first_batch(seed)

    def first_batch(self, seed: int) -> tuple[tuple[float, ...], ...]:
        return ((0.5, 1.0, 0.4, 0.25, 1.0),) * 4

    def second_batch(self, seed: int, fitness: Sequence[float]) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
        return self.first_batch(seed), self.first_batch(seed)

    def complete_run(self, seed: int, first_fitness: Sequence[float], second_fitness: Sequence[float]):
        # Fixture-only completion for the strengthened NativeProvider
        # protocol; these tests never reach an iStratDE freeze.
        import hashlib
        from research.comparison.native import NativeCompletion
        digest = hashlib.sha256(
            json.dumps([seed, [float(f) for f in first_fitness], [float(f) for f in second_fitness]]).encode()
        ).hexdigest()
        return NativeCompletion(self.first_batch(seed), self.first_batch(seed), digest)


def make_controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Orchestrator:
    monkeypatch.setattr(deploy, "git_head_commit", lambda root: COMMIT)
    return Orchestrator.initialize(runtime_dir=tmp_path / "state", repo_root=tmp_path,
                                   cells=iter_cells(make_bindings(), COMMIT),
                                   provider=DeterministicNative(), now_fn=lambda: 1000.0)


def state_path(orch: Orchestrator, key: str) -> Path:
    return orch.runtime_dir / "cells" / key.replace(":", "-") / "cell.json"


def test_all_27_terminal_flags_without_search_do_not_open_final_barrier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orch = make_controller(tmp_path, monkeypatch)
    for cell in orch.cells:
        path = state_path(orch, cell.cell_key())
        state = json.loads(path.read_text())
        state["terminal"] = {"status": "failed_no_valid_program", "reason": "all_configs_completed"}
        path.write_text(json.dumps(state))
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        with pytest.raises(OrchestrationError):
            orch.verify_final_barrier(ledger)


@pytest.mark.parametrize("method", ["FIXED", "RANDOM", "ISTRATDE"])
def test_resume_rejects_changed_valid_vector_before_any_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    orch = make_controller(tmp_path, monkeypatch)
    path = state_path(orch, f"banknote:11:{method}")
    state = json.loads(path.read_text())
    state["vectors"]["0"][0] = 0.6  # valid bounds, wrong frozen plan
    path.write_text(json.dumps(state))
    with pytest.raises(OrchestrationError):
        Orchestrator.load(runtime_dir=orch.runtime_dir, repo_root=tmp_path,
                          bindings=make_bindings(), provider=DeterministicNative(), now_fn=lambda: 1000.0)


def test_child_can_acquire_real_ledger_without_parent_lock_conflict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orch = make_controller(tmp_path, monkeypatch)
    ledger_path = tmp_path / "ledger.json"
    acquired = []

    def child(spec_path: Path) -> int:
        try:
            with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: 1000.0):
                acquired.append(True)
        except ComparisonLedgerLocked:
            acquired.append(False)
        return 2  # controlled early failure after testing real lock ownership

    with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        with pytest.raises(OrchestrationError):
            orch.run_schedule(ledger, child)
    assert acquired == [True], "production child must own the ledger while parent relinquishes it"


def test_post_child_missing_intent_stops_original_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orch = make_controller(tmp_path, monkeypatch)
    ledger_path = tmp_path / "ledger.json"

    def child(spec_path: Path) -> int:
        (orch.runtime_dir / "intents" / spec_path.name).unlink()
        return 0

    with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        with pytest.raises(OrchestrationError):
            orch.run_schedule(ledger, child)
    with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: 1000.0) as reopened:
        assert reopened.stopped, "evidence-read errors after child exit must durably stop the original ledger"


def test_resume_rejects_durable_intent_without_cell_launch_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orch = make_controller(tmp_path, monkeypatch)
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        orch.prepare_config(orch.positions[0], ledger)
    # Simulate power loss after intent landed but before the cell record landed.
    path = state_path(orch, orch.positions[0].cell_key)
    state = json.loads(path.read_text())
    state["configs"] = {}
    path.write_text(json.dumps(state))
    with pytest.raises(OrchestrationError):
        Orchestrator.load(runtime_dir=orch.runtime_dir, repo_root=tmp_path,
                          bindings=make_bindings(), provider=DeterministicNative(), now_fn=lambda: 1000.0)


def test_child_committed_usage_survives_parent_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orch = make_controller(tmp_path, monkeypatch)
    ledger_path = tmp_path / "ledger.json"
    cell = next(c for c in orch.cells if c.cell_key() == orch.positions[0].cell_key)

    def child(spec_path: Path) -> int:
        with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: 1000.0) as owned:
            reservation = owned.reserve(cell.inner_run(0), "main", estimated_input_tokens=100)
            owned.commit(reservation.request_id, ProviderUsage(17, 23), provider_request_id="synthetic-owned-request")
        return 2  # after-ledger-close result-write failure

    with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        with pytest.raises(OrchestrationError):
            orch.run_schedule(ledger, child)
    with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: 1000.0) as reopened:
        snapshot = reopened.snapshot()
        assert reopened.stopped
        assert snapshot["global"]["committed"] == {"requests": 1, "tokens": 40}
        assert snapshot["cells"][cell.cell_key()]["committed"] == {"requests": 1, "tokens": 40}
        assert len(snapshot["requests"]) == 1


def test_expired_original_deadline_never_launches_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orch = make_controller(tmp_path, monkeypatch)
    clock = [1000.0]
    orch._now = lambda: clock[0]
    launches = []
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: clock[0]) as ledger:
        ledger.start_clock()
        clock[0] += 86401.0
        def child(spec_path: Path) -> int:
            launches.append(spec_path)
            return 2
        with pytest.raises(OrchestrationError):
            orch.run_schedule(ledger, child)
    assert launches == [], "the original 24h deadline must be checked BEFORE any new config launch"
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: clock[0]) as fresh:
        assert fresh.stopped


def test_unverified_cleanup_result_cannot_complete_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from research.comparison.runner import RUN_RESULT_SCHEMA, ComparisonRunSpec, comparison_config_identity
    orch = make_controller(tmp_path, monkeypatch)
    position = orch.positions[0]
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        spec_path = orch.prepare_config(position, ledger)
        spec = ComparisonRunSpec.from_dict(json.loads(spec_path.read_text()))  # type: ignore[union-attr]
        result_path = orch._result_path(spec.run)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({
            "schema": RUN_RESULT_SCHEMA, "run_key": spec.run.run_key(),
            "run_identity": spec.run.identity(), "cell_identity": spec.run.cell.identity(),
            "config_identity": comparison_config_identity(spec),
            "status": "completed", "termination": "completed", "incumbent": None,
            "worker_cleanup_verified": False,
        }))
        with pytest.raises(OrchestrationError):
            orch.complete_config(position, ledger)
        assert ledger.stopped, "missing or false cleanup/evidence must durably stop the original ledger"


@pytest.mark.parametrize("boundary", ["prepare_config", "complete_config", "_cell_state"])
def test_any_boundary_integrity_error_stops_original_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str) -> None:
    orch = make_controller(tmp_path, monkeypatch)
    ledger_path = tmp_path / "ledger.json"
    def fail_boundary(*args: object, **kwargs: object) -> None:
        raise OrchestrationError("synthetic_boundary_integrity_failure")
    monkeypatch.setattr(orch, boundary, fail_boundary)
    with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        with pytest.raises(OrchestrationError):
            orch.run_schedule(ledger, lambda spec_path: 0)
    with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: 1000.0) as fresh:
        assert fresh.stopped, "every integrity failure in the controller must stop the actual original ledger"


def test_claimed_prediction_hash_without_snapshot_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from research.comparison.runner import ComparisonRunSpec
    from research.tests.test_comparison_orchestrate import write_fake_result
    orch = make_controller(tmp_path, monkeypatch)
    position = orch.positions[0]
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        spec = ComparisonRunSpec.from_dict(json.loads(orch.prepare_config(position, ledger).read_text()))  # type: ignore[union-attr]
        write_fake_result(orch, spec, 0.8)
        # Keep claimed prediction SHA in the result, remove any actual snapshots.
        for snapshot in (orch._run_dir(spec.run) / "predictions").glob("*.csv"):
            snapshot.unlink()
        with pytest.raises(OrchestrationError):
            orch.complete_config(position, ledger)
        assert ledger.stopped


def test_result_cannot_claim_cell_cap_without_original_ledger_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from research.comparison.runner import ComparisonRunSpec
    from research.tests.test_comparison_orchestrate import write_fake_result
    orch = make_controller(tmp_path, monkeypatch)
    position = orch.positions[0]
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        spec = ComparisonRunSpec.from_dict(json.loads(orch.prepare_config(position, ledger).read_text()))  # type: ignore[union-attr]
        write_fake_result(orch, spec, None, status="cell_terminated",
                          termination="CellBudgetExhausted:cell_terminated:cell_main_budget_insufficient")
        with pytest.raises(OrchestrationError):
            orch.complete_config(position, ledger)
        assert ledger.stopped


def test_bounded_zero_weight_native_vector_is_retained_without_resampling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class BoundedInvalidNative(DeterministicNative):
        def first_batch(self, seed: int) -> tuple[tuple[float, ...], ...]:
            valid = super().first_batch(seed)
            return ((0.5, 0.0, 0.0, 0.0, 1.0), *valid[1:])
    monkeypatch.setattr(deploy, "git_head_commit", lambda root: COMMIT)
    orch = Orchestrator.initialize(runtime_dir=tmp_path / "state", repo_root=tmp_path,
                                   cells=iter_cells(make_bindings(), COMMIT),
                                   provider=BoundedInvalidNative(), now_fn=lambda: 1000.0)
    for method in ("RANDOM", "ISTRATDE"):
        state = json.loads(state_path(orch, f"banknote:11:{method}").read_text())
        assert state["vectors"]["0"] == [0.5, 0.0, 0.0, 0.0, 1.0]
