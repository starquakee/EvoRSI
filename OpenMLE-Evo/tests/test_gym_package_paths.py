"""Regression coverage for the documented existing Gym consumption path."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = (
    ROOT.parent / "OpenMLE-Gym/examples/consumer-inputs/titanic-extended-task.json"
)
TASK = "titanic-extended@1"


@pytest.fixture(autouse=True)
def logging_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGGING_DIR", str(tmp_path / "logs"))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def build(tmp_path, validation_dir, submit_root):
    original = json.loads(EXAMPLE.read_text())
    package = tmp_path / "download" / TASK
    (package / "info").mkdir(parents=True)
    (package / "data/public").mkdir(parents=True)
    metadata = original["metadata"].copy()
    (package / "data/public/description.txt").write_text(
        metadata.pop("task_description")
    )
    (package / "info/data_description.txt").write_text(metadata.pop("data_description"))
    metadata.pop("data_dir")
    (package / "info/task_metadata.json").write_text(json.dumps(metadata))
    parquet = tmp_path / "eval.parquet"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openmle_gym.cli",
            "prepare-input",
            "--task-dir",
            str(package),
            "--sandbox-task-dir",
            validation_dir,
            "--output",
            str(parquet),
        ],
        cwd=ROOT.parent / "OpenMLE-Gym",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    restored = pd.read_parquet(parquet).iloc[0]
    assert list(restored.prompt) == original["prompt"]
    expected = original["metadata"].copy()
    expected["data_dir"] = validation_dir
    assert restored.metadata == expected

    cfg = {
        "data": {
            "eval_data": str(parquet),
            "input_key": "prompt",
            "metadata_key": "metadata",
            "leaderboard_dir": str(tmp_path),
            "submit_dir": "unused",
            "submit_data_dir_root": submit_root,
            "evaluation_protocol": "self_valid",
        },
        "sandbox": {
            "base_url_map": [{"resource": "cpu", "base_url": "http://unused.example"}],
            "job_timeout": 30,
            "wait_timeout": 30,
            "poll_interval": 0,
            "use_score2reward": True,
            "use_clear_run_log_score": True,
        },
    }
    config = tmp_path / "build.yaml"
    OmegaConf.save(OmegaConf.create(cfg), config)
    builder = module(
        "gym_example_builder",
        ROOT / "third_party/aira-evo/examples/mle_bench/build_tasks.py",
    )
    builder.build_tasks(config, tmp_path / "built")
    task_cfg = OmegaConf.to_container(
        OmegaConf.load(tmp_path / "built" / TASK / "config.yaml")
    )
    adapter = module(
        "gym_example_task",
        ROOT / "third_party/aira-evo/examples/mle_bench/base_task.py",
    )
    return adapter, adapter.SandboxMLEBenchTask(task_cfg), config


@pytest.mark.parametrize("submit_root", ["/mounted/tasks", "/mounted/test"])
def test_existing_evo_phase_paths_and_scores(tmp_path, monkeypatch, submit_root):
    adapter, task, _ = build(tmp_path, f"/mounted/tasks/{TASK}", submit_root)
    requested = []

    async def response(**kwargs):
        requested.append(kwargs["data_dir"])
        # Deliberately disagree with the stdout score: preserve current default selection.
        return 200, {
            "result": {
                "result": "success",
                "score": 0.75,
                "run_log": "Final Validation Score: 0.99",
            }
        }

    monkeypatch.setattr(adapter, "get_sandbox_result", response)
    monkeypatch.setenv("SANDBOX_CPU_API_KEY", "test-only")
    validation = task.evaluate_code("print('unchanged code')", phase="validation")
    submission = task.evaluate_code("print('unchanged code')", phase="test")
    assert requested == [f"/mounted/tasks/{TASK}", f"{submit_root}/{TASK}"]
    assert validation["selection_score"] == 0.75
    assert submission["submit_score"] == 0.75
    assert (
        task.build_submit_code("print('unchanged code')") == "print('unchanged code')"
    )


def test_same_parquet_is_consumed_by_existing_sft_builder(tmp_path):
    _, _, config = build(tmp_path, f"/mounted/tasks/{TASK}", "/mounted/tasks")
    sft = ROOT.parent / "OpenMLE-ERL/SFT"
    output = tmp_path / "sft-built"
    output.mkdir()
    script = output / "build_tasks.py"
    shutil.copyfile(
        sft / "third_party/aira-evo/examples/mle_bench/build_tasks.py", script
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sft)
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    evo_config = OmegaConf.to_container(
        OmegaConf.load(tmp_path / "built" / TASK / "config.yaml")
    )
    sft_config = OmegaConf.to_container(OmegaConf.load(output / TASK / "config.yaml"))
    for key in (
        "data_dir",
        "uuid",
        "task_description",
        "data_description",
        "public_system_prompt",
        "public_user_prompt",
        "higher_is_better",
    ):
        assert sft_config[key] == evo_config[key]


@pytest.mark.skipif(
    not os.environ.get("OPENMLE_GYM_TEST_PACKAGE"),
    reason="Set a downloaded package for real scorer verification",
)
def test_downloaded_package_matches_record_and_existing_scorer(tmp_path):
    package = Path(os.environ["OPENMLE_GYM_TEST_PACKAGE"]).resolve()
    record = json.loads(EXAMPLE.read_text())
    assert (
        record["metadata"]["task_description"]
        == (package / "data/public/description.txt").read_text()
    )
    assert (
        record["metadata"]["data_description"]
        == (package / "info/data_description.txt").read_text()
    )
    prepared = tmp_path / "real-package.parquet"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openmle_gym.cli",
            "prepare-input",
            "--task-dir",
            str(package),
            "--sandbox-task-dir",
            "/mounted/tasks/" + TASK,
            "--task-uuid",
            record["metadata"]["uuid"],
            "--output",
            str(prepared),
        ],
        cwd=ROOT.parent / "OpenMLE-Gym",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    restored = pd.read_parquet(prepared).iloc[0]
    record["metadata"]["data_dir"] = "/mounted/tasks/" + TASK
    assert list(restored.prompt) == record["prompt"]
    assert restored.metadata == record["metadata"]
    scorer = ROOT.parent / "OpenMLE-Gym/openmle-sandbox/node_workers/read_and_metric.py"
    result = subprocess.run(
        [
            sys.executable,
            str(scorer),
            "-d",
            str(package),
            "-a",
            str(package / "data/private/test_answer.csv"),
            "-p",
            str(package / "data/public/sample_submission.csv"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r"##SCORE##([0-9.eE+-]+)", result.stdout)
    assert match, result.stdout
    score = float(match.group(1))
    assert math.isfinite(score) and score >= 0
