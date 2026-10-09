"""Offline tests for sandbox_eval_client remote-job cancellation (US-007).

The legacy client raised TimeoutError on wait timeout WITHOUT cancelling the
remote job. These tests pin the repair: live-job registry, cancel-before-
raise with evidence, and fail-safe cancel_job/cancel_all_live. All HTTP is
faked via monkeypatched httpx.Client (same pattern as test_adapters.py); a
fake clock keeps the waits instant. No network, no sandbox.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import time as stdlib_time

from research.adapters.sandbox_eval_client import SandboxEvalClient

REPO_ROOT = Path(__file__).resolve().parents[2]
BENIGN_SOURCE = (REPO_ROOT / "research/experiments/hello_synth_job.py").read_text()

ENDPOINT = "http://sandbox.test"
API_KEY = "test-key"


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPError(f"status {self.status_code}")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeHttpxClient:
    """Queue-scripted stand-in for httpx.Client (sync)."""

    calls: list = []
    get_payloads: list = []
    delete_status = 202
    delete_raises: Exception | None = None
    get_raises: Exception | None = None

    @classmethod
    def reset(cls):
        cls.calls = []
        cls.get_payloads = []
        cls.delete_status = 202
        cls.delete_raises = None
        cls.get_raises = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def post(self, url, json=None, headers=None):
        type(self).calls.append(("POST", url))
        return FakeResponse({"job_id": "job-1"})

    def get(self, url, headers=None):
        type(self).calls.append(("GET", url))
        if type(self).get_raises is not None:
            raise type(self).get_raises
        if type(self).get_payloads:
            return FakeResponse(type(self).get_payloads.pop(0))
        return FakeResponse({"status": "cancelled", "completed_at": "2026-10-08T00:00:00Z", "result": {"worker_cleanup_verified": True}})

    def delete(self, url, headers=None):
        type(self).calls.append(("DELETE", url))
        if type(self).delete_raises is not None:
            raise type(self).delete_raises
        return FakeResponse({"status": "cancelled"}, status_code=type(self).delete_status)


@pytest.fixture()
def fake_http(monkeypatch):
    FakeHttpxClient.reset()
    clock = FakeClock()
    monkeypatch.setattr("httpx.Client", FakeHttpxClient)
    # sandbox_eval_client calls time.monotonic/time.sleep via the stdlib
    # module; patch it directly (monkeypatch restores it afterwards).
    monkeypatch.setattr(stdlib_time, "monotonic", clock.monotonic)
    monkeypatch.setattr(stdlib_time, "sleep", clock.sleep)
    return FakeHttpxClient


def _client():
    return SandboxEvalClient(endpoint=ENDPOINT, api_key=API_KEY)


class TestWaitTimeoutCancels:
    def test_timeout_cancels_before_raising(self, fake_http):
        fake_http.get_payloads = [{"status": "queued"}]
        client = _client()
        with pytest.raises(TimeoutError) as excinfo:
            client.submit_and_wait(
                BENIGN_SOURCE, timeout=5, poll_interval=100.0
            )
        # DELETE hit the right job URL BEFORE the TimeoutError was raised:
        # the raised error carries the cancel evidence gathered by the DELETE.
        assert ("DELETE", "/api/v1/jobs/job-1") in fake_http.calls
        message = str(excinfo.value)
        assert "job-1" in message
        assert "cancel_evidence=" in message
        assert "'cancel_http_status': 202" in message
        assert "'terminal_status': 'cancelled'" in message
        # registry drained despite the failure
        assert client.live_job_ids() == frozenset()

    def test_poll_abort_cancels_live_job(self, fake_http):
        fake_http.get_raises = httpx.HTTPError("connection lost")
        client = _client()
        with pytest.raises(httpx.HTTPError, match="connection lost"):
            client.submit_and_wait(BENIGN_SOURCE, timeout=5, poll_interval=1.0)
        assert ("DELETE", "/api/v1/jobs/job-1") in fake_http.calls
        assert client.live_job_ids() == frozenset({"job-1"})


class TestTerminalStatusNoCancel:
    def test_completed_job_not_cancelled(self, fake_http):
        fake_http.get_payloads = [
            {"status": "completed", "result": {"score": 0.9}}
        ]
        client = _client()
        result = client.submit_and_wait(BENIGN_SOURCE, timeout=5, poll_interval=1.0)
        assert result.status == "completed"
        assert result.score == pytest.approx(0.9)
        assert not [c for c in fake_http.calls if c[0] == "DELETE"]
        assert client.live_job_ids() == frozenset()

    def test_failed_job_not_cancelled(self, fake_http):
        fake_http.get_payloads = [{"status": "failed", "result": {}}]
        client = _client()
        result = client.submit_and_wait(BENIGN_SOURCE, timeout=5, poll_interval=1.0)
        assert result.status == "failed"
        assert not [c for c in fake_http.calls if c[0] == "DELETE"]
        assert client.live_job_ids() == frozenset()


class TestCancelJob:
    def test_delete_then_terminal_poll(self, fake_http):
        client = _client()
        evidence = client.cancel_job("job-x")
        assert evidence == {
            "job_id": "job-x",
            "cancel_http_status": 202,
            "cancel_error": None,
            "terminal_status": "cancelled",
            "cleanup_verified": True,
        }
        assert ("DELETE", "/api/v1/jobs/job-x") in fake_http.calls

    def test_transport_failure_recorded_not_raised(self, fake_http):
        fake_http.delete_raises = httpx.HTTPError("delete boom")
        client = _client()
        evidence = client.cancel_job("job-y")
        assert evidence["cancel_http_status"] is None
        assert "HTTPError" in evidence["cancel_error"]
        assert evidence["terminal_status"] is None

    def test_missing_credentials_recorded_not_raised(self, fake_http, monkeypatch):
        monkeypatch.delenv("SANDBOX_ENDPOINT", raising=False)
        monkeypatch.delenv("SANDBOX_API_KEY", raising=False)
        client = SandboxEvalClient()
        evidence = client.cancel_job("job-z")
        assert evidence["cancel_error"] == "sandbox_endpoint_or_api_key_unavailable"
        assert not [c for c in fake_http.calls if c[0] == "DELETE"]


class TestCancelAllLive:
    def test_cancels_every_registered_id(self, fake_http):
        client = _client()
        client._register("job-a")
        client._register("job-b")
        results = client.cancel_all_live()
        assert [r["job_id"] for r in results] == ["job-a", "job-b"]
        deletes = [url for method, url in fake_http.calls if method == "DELETE"]
        assert deletes == ["/api/v1/jobs/job-a", "/api/v1/jobs/job-b"]
        assert all(r["cancel_http_status"] == 202 for r in results)
        assert client.live_job_ids() == frozenset()

    def test_empty_registry_is_noop(self, fake_http):
        client = _client()
        assert client.cancel_all_live() == []
        assert not fake_http.calls
