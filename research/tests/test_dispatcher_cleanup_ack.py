"""US-008: cancellation/timeout cleanup acknowledgements must be explicit.

A kill or session delete is proof of terminated processes ONLY when the
worker explicitly acknowledges verified cleanup. These tests exercise the
dispatcher contract offline (faked HTTP/Redis/DB; no containers, no
candidate code on the host):
  * SandboxHTTPController.kill returns the ack; HTTP failure propagates.
  * execute_shell_command_with_deadline attaches the ack's
    cleanup_verified to timeout/cancel errors (False when unproven).
  * execute_job records worker_cleanup_verified and emits completed_at
    ONLY with proven cleanup; queued-never-started gets its own proof.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_STORAGE = Path(tempfile.mkdtemp(prefix="us008-cleanup-ack-"))
_TEST_CONFIG = _TEST_STORAGE / "sandbox_config.json"
_TEST_CONFIG.write_text(
    json.dumps({"cpu": {"endpoints": ["http://worker.invalid:8080"]}, "gpu": {"ranges": []}}),
    encoding="utf-8",
)
os.environ.setdefault("STORAGE_PATH", str(_TEST_STORAGE))
os.environ.setdefault("SANDBOX_CONFIG_FILE", str(_TEST_CONFIG))
os.environ.setdefault("EXTERNAL_EVALUATOR_ENABLED", "1")
os.environ.setdefault(
    "EVALUATOR_REGISTRY_PATH", str(REPO_ROOT / "research" / "evaluator" / "registry.v1.json")
)

sys.path.insert(0, str(REPO_ROOT / "sandbox-controller" / "task_dispatcher"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import task_dispatcher as td  # noqa: E402
from test_dispatcher_worker_control import (  # noqa: E402
    CANDIDATE_CODE,
    _evidence_for,
    _make_job,
    _stage_bytes,
)


class _Resp:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise td.httpx.HTTPStatusError(str(self.status_code), request=None, response=self)

    def json(self) -> dict:
        return self._payload


class _RecordingCursor:
    def __init__(self, conn: "_RecordingConn") -> None:
        self.conn = conn

    def __enter__(self) -> "_RecordingCursor":
        return self

    def __exit__(self, *args) -> None:
        return None

    def execute(self, sql, params=None) -> None:
        normalized = " ".join(str(sql).split()).upper()
        if normalized.startswith("UPDATE JOBS"):
            self.conn.updates.append((normalized, list(params or [])))

    def fetchone(self):
        return None


class _RecordingConn:
    def __init__(self) -> None:
        self.updates: list = []

    def cursor(self) -> _RecordingCursor:
        return _RecordingCursor(self)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _FlagRedis:
    def __init__(self, cancelled: bool = False, job_id: str = "") -> None:
        self.store = {f"job:{job_id}:cancelled": "1"} if cancelled else {}
        self.deleted: list[str] = []

    def get(self, key):
        return self.store.get(key)

    def delete(self, key) -> None:
        self.deleted.append(key)
        self.store.pop(key, None)


# --- kill/cleanup ack plumbing -----------------------------------------------


def _wire_worker(monkeypatch, *, kill_behavior: str, cleanup_verified: bool = True) -> list:
    posts: list = []

    def fake_post(url, json=None, timeout=None, headers=None):
        posts.append((url, json))
        if url.endswith("/v1/shell/exec"):
            return _Resp({"data": {"session_id": "s-1", "status": "running"}})
        if url.endswith("/v1/shell/kill"):
            if kill_behavior == "error":
                return _Resp({"error": "worker_cleanup_unproven"}, status_code=503)
            return _Resp({"data": {"session_id": "s-1", "status": "terminated",
                                   "cleanup_verified": kill_behavior == "verified"}})
        # wait/view keep the session running so the loop hits the deadline.
        return _Resp({"data": {"session_id": "s-1", "status": "running"}})

    def fake_delete(url, timeout=None, headers=None):
        return _Resp({"data": {"session_id": "s-1", "removed": True, "cleanup_verified": cleanup_verified}})

    monkeypatch.setattr(td.httpx, "post", fake_post)
    monkeypatch.setattr(td.httpx, "delete", fake_delete)
    monkeypatch.setattr(td, "WORKER_CONTROL_KEY", "k" * 32)
    return posts


def test_timeout_records_verified_cleanup_from_kill_ack(monkeypatch):
    _wire_worker(monkeypatch, kill_behavior="verified")
    redis_fake = _FlagRedis()
    with pytest.raises(td.SandboxCommandTimeoutError) as info:
        td.execute_shell_command_with_deadline(
            worker_endpoint="http://worker.invalid:8080",
            command="sleep 99",
            exec_dir=None,
            job_id="job-t",
            redis_client=redis_fake,
            db_conn=_RecordingConn(),
            job_deadline=0.0,  # already past
        )
    assert info.value.cleanup_verified is True


def test_timeout_with_failed_kill_is_unproven(monkeypatch):
    _wire_worker(monkeypatch, kill_behavior="error")
    redis_fake = _FlagRedis()
    with pytest.raises(td.SandboxCommandTimeoutError) as info:
        td.execute_shell_command_with_deadline(
            worker_endpoint="http://worker.invalid:8080",
            command="sleep 99",
            exec_dir=None,
            job_id="job-t",
            redis_client=redis_fake,
            db_conn=_RecordingConn(),
            job_deadline=0.0,
        )
    assert info.value.cleanup_verified is False


def test_cancel_records_kill_ack(monkeypatch):
    _wire_worker(monkeypatch, kill_behavior="verified")
    redis_fake = _FlagRedis(cancelled=True, job_id="job-c")
    with pytest.raises(td.SandboxCommandCancelledError) as info:
        td.execute_shell_command_with_deadline(
            worker_endpoint="http://worker.invalid:8080",
            command="sleep 99",
            exec_dir=None,
            job_id="job-c",
            redis_client=redis_fake,
            db_conn=_RecordingConn(),
            job_deadline=None,
        )
    assert info.value.cleanup_verified is True


# --- execute_job recording ----------------------------------------------------


def _run_cancelled_job(monkeypatch, tmp_path, *, cleanup_verified):
    monkeypatch.setattr(td, "ENABLE_LOCAL_JOB_SCRATCH", True)
    monkeypatch.setattr(td, "LOCAL_SCRATCH_ROOT", tmp_path / "scratch")
    monkeypatch.setattr(td, "LOCAL_DISPATCHER_LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(td, "wait_for_file_exists", lambda *a, **k: True)

    def fake_exec(*, worker_endpoint, command, exec_dir, job_id, redis_client, db_conn,
                  job_deadline, on_started=None, exec_class=None, job_root=None):
        if exec_class == "candidate":
            raise td.SandboxCommandCancelledError(
                "sess-1", "cancelled", view=None, cleanup_verified=cleanup_verified
            )
        if on_started is not None:
            on_started("stage-session")
        return td.SandboxResultWrapper({"exit_code": 0, "status": "completed", "output": ""})

    monkeypatch.setattr(td, "execute_shell_command_with_deadline", fake_exec)
    job = _make_job(tmp_path, _evidence_for(CANDIDATE_CODE))
    harness_tmp = {"tmp": tmp_path}
    _stage_bytes(harness_tmp, CANDIDATE_CODE)
    conn = _RecordingConn()
    # No pre-set cancel flag: the cancellation surfaces through the faked
    # execution call raising SandboxCommandCancelledError, not the
    # pre-dispatch queue check.
    redis_fake = _FlagRedis()
    td.execute_job(redis_fake, conn, "http://worker.invalid:8080", job)
    return conn, redis_fake


def _result_json(params) -> dict:
    for value in params:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except ValueError:
                continue
            if isinstance(parsed, dict) and ("worker_cleanup_verified" in parsed or "execution_never_started" in parsed):
                return parsed
    raise AssertionError(f"no result JSON found in params: {params}")


def test_execute_job_cancel_with_proven_cleanup_emits_completed_at(monkeypatch, tmp_path):
    conn, redis_fake = _run_cancelled_job(monkeypatch, tmp_path, cleanup_verified=True)
    sql, params = conn.updates[-1]
    assert params[0] == "cancelled"
    assert "COMPLETED_AT" in sql
    result = _result_json(params)
    assert result["worker_cleanup_verified"] is True
    assert f"job:job-identity:cancelled" in redis_fake.deleted


def test_execute_job_cancel_with_unproven_cleanup_withholds_completed_at(monkeypatch, tmp_path):
    conn, redis_fake = _run_cancelled_job(monkeypatch, tmp_path, cleanup_verified=False)
    sql, params = conn.updates[-1]
    assert params[0] == "cancelled"
    # No terminal completed_at without an explicit verified cleanup ack.
    assert "COMPLETED_AT" not in sql
    result = _result_json(params)
    assert result["worker_cleanup_verified"] is False
    # The cancel flag is retained so the state stays open for review.
    assert redis_fake.deleted == []


def test_pre_dispatch_cancel_records_execution_never_started(monkeypatch, tmp_path):
    monkeypatch.setattr(td, "ENABLE_LOCAL_JOB_SCRATCH", True)
    monkeypatch.setattr(td, "LOCAL_SCRATCH_ROOT", tmp_path / "scratch")
    monkeypatch.setattr(td, "LOCAL_DISPATCHER_LOG_ROOT", tmp_path / "logs")
    executed: list = []
    monkeypatch.setattr(
        td,
        "execute_shell_command_with_deadline",
        lambda **kwargs: executed.append(kwargs) or td.SandboxResultWrapper({}),
    )
    job = _make_job(tmp_path, _evidence_for(CANDIDATE_CODE))
    conn = _RecordingConn()
    redis_fake = _FlagRedis(cancelled=True, job_id=job["job_id"])
    td.execute_job(redis_fake, conn, "http://worker.invalid:8080", job)
    sql, params = conn.updates[-1]
    assert params[0] == "cancelled"
    assert "COMPLETED_AT" in sql
    result = _result_json(params)
    assert result["execution_never_started"] is True
    assert executed == []  # nothing ever reached the worker
