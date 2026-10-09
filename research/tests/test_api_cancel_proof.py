"""US-008: API queued cancellation needs a real execution_never_started proof.

The proof is the actual Redis LREM result on a job that was still queued
(not a stale LRANGE match, never an in-flight/running job). Only with
that proof does the API write terminal evidence (result +
execution_never_started + completed_at). Offline: faked DB/Redis, no
containers, no candidate code.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_STORAGE = Path(tempfile.mkdtemp(prefix="us008-api-cancel-"))
os.environ.setdefault("STORAGE_PATH", str(_TEST_STORAGE))
os.environ.setdefault("UPLOAD_ROOT", str(_TEST_STORAGE / "uploads"))
os.environ.setdefault("SANDBOX_API_KEYS", "us008-test-key")
os.environ.setdefault("ENABLE_OBS", "0")

sys.path.insert(0, str(REPO_ROOT / "sandbox-controller" / "api_server"))

import api_server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# api_server may have been imported first by another test module with its
# own key; always use the effective one.
API_HEADERS = {"X-API-Key": os.environ["SANDBOX_API_KEYS"].split(",")[0]}
JOB_ID = "job-cancel-proof"


class FakeCursor:
    def __init__(self, conn: "FakeConn") -> None:
        self.conn = conn
        self.rowcount = 0
        self._row: dict | None = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args) -> None:
        return None

    def execute(self, sql, params=None) -> None:
        normalized = " ".join(str(sql).split()).upper()
        if normalized.startswith("UPDATE JOBS"):
            self.rowcount = self.conn.update_rowcount
            self.conn.updates.append((normalized, list(params or [])))
        self._row = self.conn.next_row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return [] if self._row is None else [self._row]


class FakeConn:
    def __init__(self, prior_status: str | None) -> None:
        self.next_row: dict | None = {"status": prior_status} if prior_status else None
        self.update_rowcount = 1
        self.updates: list = []

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class FakeRedis:
    def __init__(self, queued_payload: str | None = None) -> None:
        self.queues: dict[str, list[str]] = {}
        if queued_payload is not None:
            self.queues[api_server.JOB_QUEUE_CPU_P1] = [queued_payload]
        self.flags: dict[str, str] = {}

    def lrange(self, key, start, end):
        return list(self.queues.get(key, []))

    def lrem(self, key, count, value):
        try:
            self.queues.get(key, []).remove(value)
            return 1
        except ValueError:
            return 0

    def setex(self, key, ttl, value):
        self.flags[key] = value


@pytest.fixture()
def client():
    return TestClient(api_server.app)


def _wire(monkeypatch, conn, redis_fake):
    @contextmanager
    def _fake_get_db_connection():
        yield conn

    monkeypatch.setattr(api_server, "get_db_connection", _fake_get_db_connection)
    monkeypatch.setattr(api_server, "redis_client", redis_fake)


def _queued_payload() -> str:
    return json.dumps({"job_id": JOB_ID, "code_file_path": "/x/main.py"})


def test_queued_cancel_with_real_lrem_proves_never_started(client, monkeypatch):
    conn = FakeConn("queued")
    redis_fake = FakeRedis(_queued_payload())
    _wire(monkeypatch, conn, redis_fake)
    response = client.delete(f"/api/v1/jobs/{JOB_ID}", headers=API_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["removed_from_queue"] is True
    assert body["execution_never_started"] is True
    # Terminal evidence written: result JSON + completed_at.
    proof_updates = [u for u in conn.updates if "COMPLETED_AT" in u[0]]
    assert len(proof_updates) == 1
    result = json.loads(proof_updates[0][1][0])
    assert result["execution_never_started"] is True
    # The queue entry is actually gone.
    assert redis_fake.queues[api_server.JOB_QUEUE_CPU_P1] == []


def test_queued_cancel_without_lrem_match_is_not_a_proof(client, monkeypatch):
    # Job row says queued, but the queue holds no such entry (the
    # dispatcher may have just dequeued it): no never-started claim.
    conn = FakeConn("queued")
    redis_fake = FakeRedis(None)
    _wire(monkeypatch, conn, redis_fake)
    response = client.delete(f"/api/v1/jobs/{JOB_ID}", headers=API_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["removed_from_queue"] is False
    assert body["execution_never_started"] is False
    assert all("COMPLETED_AT" not in u[0] for u in conn.updates)


def test_running_cancel_is_never_labelled_never_started(client, monkeypatch):
    conn = FakeConn("running")
    redis_fake = FakeRedis(_queued_payload())  # stale queue copy must not count
    _wire(monkeypatch, conn, redis_fake)
    response = client.delete(f"/api/v1/jobs/{JOB_ID}", headers=API_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["execution_never_started"] is False
    assert body["cleanup_acknowledgement"] == "awaiting_dispatcher_cleanup_proof"
    # The stale queue entry is left alone (dispatcher owns in-flight jobs),
    # and the cancel flag is set for the dispatcher poll.
    assert redis_fake.queues[api_server.JOB_QUEUE_CPU_P1] != []
    assert redis_fake.flags.get(f"job:{JOB_ID}:cancelled") == "1"
    assert all("COMPLETED_AT" not in u[0] for u in conn.updates)


def test_unknown_job_cancel_is_404(client, monkeypatch):
    conn = FakeConn(None)
    conn.update_rowcount = 0
    _wire(monkeypatch, conn, FakeRedis(None))
    response = client.delete(f"/api/v1/jobs/{JOB_ID}", headers=API_HEADERS)
    assert response.status_code == 404
