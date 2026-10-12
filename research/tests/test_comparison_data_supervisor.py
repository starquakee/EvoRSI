"""Supervisor regressions: inconsistent manifest data must not become cell bindings."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pytest
from research.comparison.data import DEFAULT_MANIFEST_PATH, ComparisonDataError, load_task_bindings
from research.comparison.specs import TaskBinding


@pytest.mark.parametrize("mutation", [
    "foreign_binding", "stale_identity", "binding_file_disagreement",
    "wrong_split_seed", "wrong_split_folds", "wrong_split_version",
])
def test_binding_loader_rejects_inconsistent_manifest(tmp_path: Path, mutation: str) -> None:
    manifest: dict[str, Any] = json.loads(DEFAULT_MANIFEST_PATH.read_text())
    entry = manifest["tasks"]["banknote"]
    if mutation == "foreign_binding":
        entry["task_binding"] = manifest["tasks"]["wdbc"]["task_binding"]
        entry["task_binding_identity"] = manifest["tasks"]["wdbc"]["task_binding_identity"]
    elif mutation == "stale_identity":
        entry["task_binding_identity"] = "0" * 64
    elif mutation == "binding_file_disagreement":
        entry["task_binding"]["train_csv_sha256"] = "0" * 64
        entry["task_binding_identity"] = TaskBinding(**entry["task_binding"]).identity()
    elif mutation == "wrong_split_seed":
        entry["split_seed"] += 1
    elif mutation == "wrong_split_folds":
        entry["split_folds"] = {"train": [0, 1, 3], "validation": 2, "final": 4}
    elif mutation == "wrong_split_version":
        manifest["sklearn_version"] = "0.0"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ComparisonDataError):
        load_task_bindings(path)
