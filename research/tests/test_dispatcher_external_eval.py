"""Dispatcher external-evaluation integration tests (US-005).

Runs the WSL task_dispatcher.execute_job end-to-end offline with faked
Redis/DB/worker transport: the trusted evaluator scores the submission
controller-side only AFTER the (fake) candidate process has stopped, the
worker command stream never touches the evaluator registry/answers, and
tampered or unallowlisted submissions fail closed with machine-readable
reasons. No containers, no network, no candidate code execution.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_STORAGE = Path(tempfile.mkdtemp(prefix="us005-dispatch-storage-"))
_TEST_CONFIG = _TEST_STORAGE / "sandbox_config.json"
_TEST_CONFIG.write_text(
    json.dumps({"cpu": {"endpoints": ["http://worker.invalid:8080"]}, "gpu": {"ranges": []}}),
    encoding="utf-8",
)
os.environ["STORAGE_PATH"] = str(_TEST_STORAGE)
os.environ["SANDBOX_CONFIG_FILE"] = str(_TEST_CONFIG)
os.environ["EXTERNAL_EVALUATOR_ENABLED"] = "1"
os.environ["EVALUATOR_REGISTRY_PATH"] = str(
    REPO_ROOT / "research" / "evaluator" / "registry.v1.json"
)

sys.path.insert(0, str(REPO_ROOT / "sandbox-controller" / "task_dispatcher"))

import task_dispatcher as td  # noqa: E402

REGISTRY_DIR = (REPO_ROOT / "research" / "evaluator").resolve()
ANSWER_BYTES = (REGISTRY_DIR / "tasks" / "hello_synth" / "test_answer.csv").read_bytes()


class FakeCursor:
    def __init__(self, conn: "FakeConn") -> None:
        self.conn = conn

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args) -> None:
        return None

    def execute(self, sql, params=None) -> None:
        normalized = " ".join(str(sql).split()).upper()
        if normalized.startswith("UPDATE JOBS"):
            self.conn.updates.append(params)
        self._fetchone = None

    def fetchone(self):
        return None


class FakeConn:
    def __init__(self) -> None:
        self.updates: list = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key):
        return self.store.get(key)

    def delete(self, key) -> None:
        self.store.pop(key, None)


def make_job(tmp_path: Path, *, task_id: str, submission: bytes) -> tuple[dict, Path]:
    job_id = f"job-{task_id}"
    job_dir = tmp_path / job_id
    code_dir = job_dir / "code"
    code_dir.mkdir(parents=True)
    code_file = code_dir / "main.py"
    code_file.write_text("print('candidate')\n", encoding="utf-8")
    (code_dir / "submission.csv").write_bytes(submission)
    job_data = {
        "job_id": job_id,
        "task_id": task_id,
        "code_file_path": str(code_file),
        "job_storage_dir": str(job_dir),
        "data_dir": "/data/hello_synth",
        "environment": {"EXECUTION_MODE": "shell"},
        "timeout": 600,
    }
    return job_data, code_dir / "submission.csv"


@pytest.fixture()
def harness(monkeypatch, tmp_path):
    events: list[str] = []
    commands: list[str] = []

    def fake_exec(*, worker_endpoint, command, exec_dir, job_id, redis_client, db_conn,
                  job_deadline, on_started=None, exec_class=None, job_root=None):
        commands.append(command)
        events.append("exec_start")
        if on_started is not None:
            on_started("fake-session")
        events.append("exec_end")  # process stopped before returning
        return td.SandboxResultWrapper({"exit_code": 0, "status": "completed", "output": ""})

    real_score = td.score_job_externally

    def spy_score(**kwargs):
        events.append("score")
        return real_score(**kwargs)

    monkeypatch.setattr(td, "execute_shell_command_with_deadline", fake_exec)
    monkeypatch.setattr(td, "wait_for_file_exists", lambda *a, **k: True)
    monkeypatch.setattr(td, "score_job_externally", spy_score)
    conn = FakeConn()
    return {"events": events, "commands": commands, "conn": conn, "redis": FakeRedis()}


def run_job(harness, job_data):
    td.execute_job(harness["redis"], harness["conn"], "http://worker.invalid:8080", job_data)
    final = harness["conn"].updates[-1]
    status = final[0]
    result = json.loads(final[1])
    return status, result


def perfect_submission() -> bytes:
    return ANSWER_BYTES


def test_scores_after_process_stop_with_evidence(harness, tmp_path) -> None:
    job_data, submission_path = make_job(
        tmp_path, task_id="hello_synth", submission=perfect_submission()
    )
    status, result = run_job(harness, job_data)

    assert status == "completed"
    assert result["result"] == "success"
    assert result["score"] == 1.0
    # scoring happened strictly after the candidate process stopped
    assert harness["events"] == ["exec_start", "exec_end", "score"]
    # worker commands never reference the evaluator registry or answers
    for command in harness["commands"]:
        assert str(REGISTRY_DIR) not in command
        assert "test_answer" not in command
        assert "read_and_metric" not in command
    evidence = result["evaluation"]
    assert evidence["evaluator"] == "external_trusted"
    assert evidence["task_id"] == "hello_synth"
    assert evidence["split"] == "test"
    assert evidence["metric"] == "accuracy"
    assert evidence["direction"] == "maximize"
    assert evidence["answer_sha256"] == hashlib.sha256(ANSWER_BYTES).hexdigest()
    assert evidence["prediction_sha256"] == hashlib.sha256(perfect_submission()).hexdigest()
    # immutable snapshot persisted with the exact scored bytes
    snapshot = Path(evidence["prediction_snapshot"])
    assert snapshot.read_bytes() == submission_path.read_bytes()


def test_wrong_predictions_score_lower(harness, tmp_path) -> None:
    lines = ANSWER_BYTES.decode("utf-8").splitlines()
    flipped = [lines[0]] + [
        f"{row.split(',')[0]},{1 - int(row.split(',')[1])}" for row in lines[1:]
    ]
    job_data, _ = make_job(
        tmp_path, task_id="hello_synth", submission=("\n".join(flipped) + "\n").encode()
    )
    status, result = run_job(harness, job_data)
    assert status == "completed"
    assert result["score"] == 0.0


def test_tampered_submission_fails_closed(harness, tmp_path) -> None:
    lines = ANSWER_BYTES.decode("utf-8").splitlines()
    tampered = [lines[0]] + [lines[1].rsplit(",", 1)[0] + ",2"] + lines[2:]
    job_data, _ = make_job(
        tmp_path, task_id="hello_synth", submission=("\n".join(tampered) + "\n").encode()
    )
    status, result = run_job(harness, job_data)
    assert status == "failed"
    assert result["result"] == "scoring_failed"
    assert result["score"] is None
    assert result["evaluation"]["reason"] == "submission_label_invalid"


def test_unallowlisted_task_fails_closed(harness, tmp_path) -> None:
    job_data, _ = make_job(tmp_path, task_id="evil_task", submission=perfect_submission())
    job_data["data_dir"] = "/data/evil_task"
    status, result = run_job(harness, job_data)
    assert status == "failed"
    assert result["result"] == "scoring_failed"
    assert result["evaluation"]["reason"] == "task_not_allowlisted"


def test_oversized_submission_fails_closed(harness, tmp_path) -> None:
    big = b"id,label\n" + b"1,1\n" * 400_000
    job_data, _ = make_job(tmp_path, task_id="hello_synth", submission=big)
    status, result = run_job(harness, job_data)
    assert status == "failed"
    assert result["evaluation"]["reason"] == "prediction_oversized"


def test_task_id_resolved_from_data_dir_basename(harness, tmp_path) -> None:
    job_data, _ = make_job(tmp_path, task_id="hello_synth_e2e", submission=perfect_submission())
    status, result = run_job(harness, job_data)
    assert status == "completed"
    assert result["evaluation"]["task_id"] == "hello_synth"


def test_candidate_metric_module_never_imported(harness, tmp_path) -> None:
    # A candidate-plantable utils/metric.py inside data_dir must never be
    # imported or executed by the trusted evaluator path.
    assert "metric" not in sys.modules or "hello_synth" not in str(
        getattr(sys.modules.get("metric"), "__file__", "")
    )
    job_data, _ = make_job(tmp_path, task_id="hello_synth", submission=perfect_submission())
    status, _ = run_job(harness, job_data)
    assert status == "completed"
    for module_name, module in list(sys.modules.items()):
        module_file = str(getattr(module, "__file__", "") or "")
        assert "utils/metric.py" not in module_file
        assert "utils\\metric.py" not in module_file


def test_registry_loaded_at_startup() -> None:
    assert td.EXTERNAL_EVALUATOR_ENABLED is True
    assert td.EXTERNAL_EVALUATOR_REGISTRY is not None
    assert "hello_synth" in td.EXTERNAL_EVALUATOR_REGISTRY.tasks
