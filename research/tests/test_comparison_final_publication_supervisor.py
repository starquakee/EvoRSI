"""Supervisor final-release regression; no real sealed data is read."""
from pathlib import Path
import json
import pytest
from research.comparison import data
from research.comparison.protocol import METHODS, SEARCH_SEEDS, TASKS, identity_of


@pytest.mark.parametrize("plant_terminal_files", [False, True])
def test_self_hashed_27_terminal_claims_cannot_release_final_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plant_terminal_files: bool) -> None:
    forged = {
        "schema": "istratde-comparison-barrier.v1",
        "schedule_identity": "a" * 64,
        "experiment_identity": "b" * 64,
        "source_commit": "c" * 40,
        "cells": {
            f"{task.task_id}:{seed}:{method}": {
                "terminal_status": "failed_no_valid_program", "winner": None,
            }
            for task in TASKS for seed in SEARCH_SEEDS for method in METHODS
        },
    }
    forged["barrier_identity"] = identity_of(forged)
    if plant_terminal_files:
        for key in forged["cells"]:
            directory = tmp_path / ".runtime" / "comparison-istratde" / "cells" / key.replace(":", "-")
            directory.mkdir(parents=True)
            (directory / "cell.json").write_text(json.dumps({
                "terminal": {"status": "failed_no_valid_program", "reason": "all_configs_completed"},
                "configs": {}, "winner": None,
            }))
    accessed = []
    def deny_sealed_access(*args: object, **kwargs: object) -> dict:
        accessed.append(True)
        raise AssertionError("publication reached sealed manifest with no search/ledger evidence")
    monkeypatch.setattr(data, "load_manifest", deny_sealed_access)
    with pytest.raises(data.ComparisonDataError):
        data.publish_final_refit(barrier_evidence=forged, repo_root=tmp_path,
                                 sealed_root=tmp_path / "sealed", public_root=tmp_path / "public",
                                 evaluator_root=tmp_path / "evaluator", registry_path=tmp_path / "registry.json")
    assert not accessed


def test_real_empty_ledger_cannot_justify_all_skipped_terminal_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from research.comparison.budget import ComparisonLedger
    from research.tests.test_comparison_orchestration_supervisor import make_controller, state_path
    orch = make_controller(tmp_path, monkeypatch)
    ledger_path = orch.runtime_dir / "comparison-ledger.json"
    with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        experiment_identity = ledger.experiment_identity
    barrier = {"schema": "istratde-comparison-barrier.v1", "source_commit": orch.source_commit,
               "schedule_identity": orch._schedule_identity, "experiment_identity": experiment_identity,
               "cells": {}}
    for cell in orch.cells:
        key = cell.cell_key()
        path = state_path(orch, key)
        state = json.loads(path.read_text())
        state["configs"] = {
            str(pos.config_index): {"status": "not_launched_cell_exhausted",
                                    "config_index": pos.config_index, "position": pos.position}
            for pos in orch.positions if pos.cell_key == key
        }
        state["terminal"] = {"status": "failed_no_valid_program", "reason": "cell_budget_exhausted"}
        state["winner"] = None
        path.write_text(json.dumps(state))
        barrier["cells"][key] = {"terminal_status": "failed_no_valid_program", "winner": None}  # type: ignore[index]
    barrier["barrier_identity"] = identity_of(barrier)
    accessed = []
    def deny_sealed_access(*args: object, **kwargs: object) -> dict:
        accessed.append(True)
        raise AssertionError("untouched real ledger cannot justify 27 exhausted terminal cells")
    monkeypatch.setattr(data, "load_manifest", deny_sealed_access)
    with pytest.raises(data.ComparisonDataError):
        data.publish_final_refit(barrier_evidence=barrier, repo_root=tmp_path,
                                 sealed_root=tmp_path / "sealed", public_root=tmp_path / "public",
                                 evaluator_root=tmp_path / "evaluator", registry_path=tmp_path / "registry.json",
                                 evidence_root=orch.runtime_dir)
    assert not accessed
