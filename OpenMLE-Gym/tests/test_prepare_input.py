"""Check package preparation against an input used by the original consumers."""

import json
from pathlib import Path
import pandas as pd
import pytest
from openmle_gym.cli import main

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "OpenMLE-Gym/examples/consumer-inputs/titanic-extended-task.json"


def package(tmp_path):
    record = json.loads(ORIGINAL.read_text())
    task = tmp_path / record["metadata"]["task_name"]
    (task / "info").mkdir(parents=True)
    (task / "data/public").mkdir(parents=True)
    metadata = record["metadata"].copy()
    (task / "data/public/description.txt").write_text(metadata.pop("task_description"))
    (task / "info/data_description.txt").write_text(metadata.pop("data_description"))
    metadata.pop("data_dir")
    (task / "info/task_metadata.json").write_text(json.dumps(metadata))
    return task, record


def test_prepare_cli_reproduces_original_record(tmp_path):
    task, original = package(tmp_path)
    output = tmp_path / "eval.parquet"
    assert (
        main(
            [
                "prepare-input",
                "--task-dir",
                str(task),
                "--sandbox-task-dir",
                "/mounted/tasks/" + task.name,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    got = pd.read_parquet(output).iloc[0]
    original["metadata"]["data_dir"] = "/mounted/tasks/" + task.name
    assert list(got.prompt) == original["prompt"]
    assert got.metadata == original["metadata"]


def test_prepare_does_not_invent_missing_metadata(tmp_path):
    task, _ = package(tmp_path)
    (task / "info/task_metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="task_metadata.json"):
        main(
            [
                "prepare-input",
                "--task-dir",
                str(task),
                "--sandbox-task-dir",
                "/mounted/tasks/" + task.name,
                "--output",
                str(tmp_path / "eval.parquet"),
            ]
        )
