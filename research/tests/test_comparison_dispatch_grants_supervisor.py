"""A filesystem basename must never create worker authorization."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMP = Path(tempfile.mkdtemp(prefix="comparison-supervisor-dispatch-"))
CONFIG = TEMP / "sandbox_config.json"
CONFIG.write_text(json.dumps({"cpu": {"endpoints": ["http://worker.invalid:8080"]}, "gpu": {"ranges": []}}))
os.environ.setdefault("STORAGE_PATH", str(TEMP))
os.environ.setdefault("SANDBOX_CONFIG_FILE", str(CONFIG))
os.environ.setdefault("EXTERNAL_EVALUATOR_ENABLED", "1")
os.environ.setdefault("EVALUATOR_REGISTRY_PATH", str(ROOT / "research/evaluator/registry.v1.json"))
sys.path.insert(0, str(ROOT / "sandbox-controller/task_dispatcher"))
import task_dispatcher as td


def test_data_dir_basename_alone_cannot_authorize_task():
    registry = td.EXTERNAL_EVALUATOR_REGISTRY
    assert registry is not None
    grant, reason = td.resolve_worker_data_grant(
        registry, {"data_dir": "/mnt/rsi_data/hello_synth"}
    )
    assert grant is None
    assert reason


def test_explicit_registered_task_can_authorize_its_own_data():
    registry = td.EXTERNAL_EVALUATOR_REGISTRY
    assert registry is not None
    grant, reason = td.resolve_worker_data_grant(
        registry, {"task_id": "hello_synth", "data_dir": "/mnt/rsi_data/hello_synth"}
    )
    assert grant == "/mnt/rsi_data/hello_synth/data/public"
    assert reason is None
