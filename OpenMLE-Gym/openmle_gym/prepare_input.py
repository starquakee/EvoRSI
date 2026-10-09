"""Prepare the existing prompt/metadata format from a complete Gym package."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

PROMPT_PATH = Path(__file__).parent / "templates/selfvalid0327-sft4.json"
METADATA_KEYS = (
    "task_name",
    "task",
    "cpu_gpu",
    "source",
    "modality",
    "higher_is_better",
    "theoretical_min",
    "theoretical_max",
    "leaderboard_min",
    "leaderboard_max",
)


def prepare_input(
    task_dir: str,
    sandbox_task_dir: str,
    output: str,
    task_uuid: str | None = None,
) -> dict:
    # Copy task-specific fields from the package into the consumer input schema.
    package = Path(task_dir).resolve()
    sandbox_path = Path(sandbox_task_dir)
    if not sandbox_path.is_absolute():
        raise ValueError(
            "--sandbox-task-dir must be an absolute container-visible task path"
        )
    source_metadata = json.loads(
        (package / "info/task_metadata.json").read_text(encoding="utf-8")
    )
    metadata = {key: source_metadata[key] for key in METADATA_KEYS}
    metadata.update(
        uuid=task_uuid or source_metadata.get("uuid") or str(uuid4()),
        data_dir=str(sandbox_path),
        task_description=(package / "data/public/description.txt").read_text(
            encoding="utf-8"
        ),
        data_description=(package / "info/data_description.txt").read_text(
            encoding="utf-8"
        ),
    )
    prompt = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"prompt": prompt, "metadata": metadata}]).to_parquet(
        destination, index=False, engine="pyarrow"
    )
    return {
        "output": str(destination),
        "rows": 1,
        "task_dir": str(package),
        "data_dir": str(sandbox_path),
        "submit_data_dir_root": str(sandbox_path.parent),
        "uuid": metadata["uuid"],
    }
