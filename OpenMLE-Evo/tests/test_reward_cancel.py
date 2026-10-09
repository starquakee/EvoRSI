"""US-007 sandbox-job cancellation tests (offline; fake async client).

Covers the live-job registry + best-effort DELETE cancellation in
``tts_search.reward_func_utils``: wait_timeout cancels the orphaned remote
job, normal completion never cancels, and ``cancel_all_live_jobs`` drains the
registry. The HTTP layer is a recording fake; sleeps are patched out.
"""
from __future__ import annotations

import asyncio

import pytest

from tts_search import reward_func_utils

BENIGN_CODE = "import pandas as pd\nprint('rows:', 1)\n"


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.headers = {}

    def json(self):
        return self._payload


class CancelRecordingClient:
    """Stands in for httpx.AsyncClient; records every request."""

    base_url = "https://sandbox.example"

    def __init__(self, *, job_status: str = "completed"):
        self.requests: list[tuple[str, str]] = []
        self.job_status = job_status

    async def post(self, url, *, json=None, headers=None):
        self.requests.append(("POST", url))
        return FakeResponse(200, {"job_id": "job-1"})

    async def get(self, url, *, headers=None):
        self.requests.append(("GET", url))
        return FakeResponse(200, {"status": self.job_status,
                                  "completed_at": "2026-10-08T00:00:00Z",
                                  "result": {"worker_cleanup_verified": True}})

    async def delete(self, url, *, headers=None):
        self.requests.append(("DELETE", url))
        self.job_status = "cancelled"
        return FakeResponse(200, {"status": "cancelled"})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def _sleep(delay):
        return None

    monkeypatch.setattr(reward_func_utils.asyncio, "sleep", _sleep)


@pytest.fixture(autouse=True)
def clean_registry():
    for job_id in reward_func_utils.live_job_ids():
        reward_func_utils.unregister_live_job(job_id)
    yield
    for job_id in reward_func_utils.live_job_ids():
        reward_func_utils.unregister_live_job(job_id)


def _run(client, **kwargs):
    return asyncio.run(
        reward_func_utils.get_sandbox_result(
            client=client,
            code_str=BENIGN_CODE,
            data_dir="/tmp/data",
            **kwargs,
        )
    )


def test_wait_timeout_cancels_remote_job():
    # wait_timeout=0: the poll deadline expires immediately, exercising the
    # 504 path where the orphaned remote job must be cancelled.
    client = CancelRecordingClient(job_status="running")
    status_code, payload = _run(client, wait_timeout=0, poll_interval=0)

    assert status_code == 504
    assert payload["error"] == "wait_timeout exceeded"
    assert payload["job_id"] == "job-1"
    deletes = [r for r in client.requests if r[0] == "DELETE"]
    assert deletes == [("DELETE", "/api/v1/jobs/job-1")]
    cancel_status = payload["cancel_status"]
    assert cancel_status["job_id"] == "job-1"
    assert cancel_status["cancel_http_status"] == 200
    assert cancel_status["cancel_error"] is None
    # Registry is drained on the terminal path.
    assert reward_func_utils.live_job_ids() == frozenset()


def test_normal_completion_never_cancels():
    client = CancelRecordingClient(job_status="completed")
    status_code, payload = _run(client, wait_timeout=10, poll_interval=0)

    assert status_code == 200
    assert payload["status"] == "completed"
    assert [r for r in client.requests if r[0] == "DELETE"] == []
    assert reward_func_utils.live_job_ids() == frozenset()


def test_cancel_error_is_reported_not_raised():
    class FailingDeleteClient(CancelRecordingClient):
        async def delete(self, url, *, headers=None):
            self.requests.append(("DELETE", url))
            raise ConnectionError("sandbox unreachable")

    client = FailingDeleteClient(job_status="running")
    status_code, payload = _run(client, wait_timeout=0, poll_interval=0)

    assert status_code == 504
    cancel_status = payload["cancel_status"]
    assert cancel_status["cancel_http_status"] is None
    assert "ConnectionError" in cancel_status["cancel_error"]
    assert cancel_status["cleanup_verified"] is False
    assert reward_func_utils.live_job_ids() == frozenset({"job-1"})


def test_cancel_all_live_jobs_cancels_everything():
    client = CancelRecordingClient()
    reward_func_utils.register_live_job("job-a")
    reward_func_utils.register_live_job("job-b")
    assert reward_func_utils.live_job_ids() == frozenset({"job-a", "job-b"})

    results = asyncio.run(reward_func_utils.cancel_all_live_jobs(client))

    assert [r["job_id"] for r in results] == ["job-a", "job-b"]
    assert all(r["cancel_http_status"] == 200 for r in results)
    deletes = [r for r in client.requests if r[0] == "DELETE"]
    assert deletes == [
        ("DELETE", "/api/v1/jobs/job-a"),
        ("DELETE", "/api/v1/jobs/job-b"),
    ]
    assert reward_func_utils.live_job_ids() == frozenset()


def test_deprecated_path_wait_timeout_cancels():
    client = CancelRecordingClient(job_status="running")
    with pytest.warns(DeprecationWarning):
        status_code, payload = asyncio.run(
            reward_func_utils.get_sandbox_result_deprecated(
                client,
                BENIGN_CODE,
                "/tmp/data",
                resource_type="cpu",
                wait_timeout=0,
                poll_interval=0,
            )
        )
    assert status_code == 504
    assert payload["cancel_status"]["cancel_http_status"] == 200
    assert [r for r in client.requests if r[0] == "DELETE"] == [
        ("DELETE", "/api/v1/jobs/job-1")
    ]
    assert reward_func_utils.live_job_ids() == frozenset()
