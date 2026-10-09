"""Evo submission source-gate wiring tests (US-004).

Both sandbox submission paths in tts_search.reward_func_utils (the actual
``get_sandbox_result`` and the deprecated pre-v0.9.32 one) must run the
unified versioned source policy gate on the exact submitted bytes BEFORE
any HTTP request. All Evo operators (draft/debug/improve/crossover) reach
the sandbox through these two functions, so spying here proves every
operator shares the same gate. No network and no job execution: the HTTP
layer is a recording fake.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tts_search import reward_func_utils

BENIGN_CODE = "import pandas as pd\nprint('rows:', 1)\n"
MALICIOUS_CODE = "import subprocess\nsubprocess.call(['id'])\n"


class RecordingResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.headers = {}

    def json(self):
        return self._payload


class RecordingClient:
    """Stands in for httpx.AsyncClient; records every request."""

    base_url = "https://sandbox.example"

    def __init__(self):
        self.requests: list[tuple[str, str, dict]] = []

    async def post(self, url, *, json=None, headers=None):
        self.requests.append(("POST", url, json or {}))
        return RecordingResponse(200, {"job_id": "job-1"})

    async def get(self, url, *, headers=None):
        self.requests.append(("GET", url, {}))
        return RecordingResponse(200, {"status": "completed", "result": {"score": 0.5}})


@pytest.fixture()
def client():
    return RecordingClient()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def _sleep(delay):
        return None

    monkeypatch.setattr(reward_func_utils.asyncio, "sleep", _sleep)


def _run_actual(client, code):
    return asyncio.run(
        reward_func_utils.get_sandbox_result(
            client=client,
            code_str=code,
            data_dir="/tmp/data",
            wait_timeout=10,
            poll_interval=0,
        )
    )


def _run_deprecated(client, code):
    with pytest.warns(DeprecationWarning):
        return asyncio.run(
            reward_func_utils.get_sandbox_result_deprecated(
                client,
                code,
                "/tmp/data",
                resource_type="cpu",
                wait_timeout=10,
                poll_interval=0,
            )
        )


class TestActualSubmissionPath:
    def test_malicious_code_denied_before_any_http(self, client):
        status_code, payload = _run_actual(client, MALICIOUS_CODE)
        assert status_code == 422
        assert payload["error"] == "source_gate_denied"
        assert payload["retryable"] is False
        gate = payload["gate"]
        assert gate["allowed"] is False
        assert gate["security_boundary"] is False
        assert gate["policy_version"] == "source-gate.v2"
        assert any(d["rule"] == "denied_marker" for d in gate["denials"])
        assert client.requests == []

    def test_benign_ml_code_passes_gate_and_submits(self, client):
        status_code, payload = _run_actual(client, BENIGN_CODE)
        assert status_code == 200
        assert payload["status"] == "completed"
        posts = [r for r in client.requests if r[0] == "POST"]
        assert len(posts) == 1
        assert posts[0][2]["code"] == BENIGN_CODE

    def test_missing_policy_fails_closed_without_http(self, client, monkeypatch, tmp_path):
        monkeypatch.setenv("SOURCE_GATE_POLICY_PATH", str(tmp_path / "missing.json"))
        status_code, payload = _run_actual(client, BENIGN_CODE)
        assert status_code == 422
        assert payload["error"] == "source_gate_denied"
        assert any(
            d["rule"] == "policy_unavailable" for d in payload["gate"]["denials"]
        )
        assert client.requests == []


class TestDeprecatedSubmissionPath:
    def test_malicious_code_denied_before_any_http(self, client):
        status_code, payload = _run_deprecated(client, MALICIOUS_CODE)
        assert status_code == 422
        assert payload["error"] == "source_gate_denied"
        assert any(
            d["rule"] == "denied_marker" for d in payload["gate"]["denials"]
        )
        assert client.requests == []

    def test_benign_ml_code_passes_gate_and_submits(self, client):
        status_code, payload = _run_deprecated(client, BENIGN_CODE)
        assert status_code == 200
        posts = [r for r in client.requests if r[0] == "POST"]
        assert len(posts) == 1


class TestSingleSharedGate:
    def test_both_paths_invoke_the_same_gate(self, client, monkeypatch):
        from research.contracts import source_gate

        calls: list[bytes] = []
        original = source_gate.check_source_bytes

        def spy(source, policy):
            calls.append(source)
            return original(source, policy)

        monkeypatch.setattr(reward_func_utils._source_gate, "check_source_bytes", spy)
        _run_actual(client, BENIGN_CODE)
        _run_deprecated(client, BENIGN_CODE)
        assert calls == [BENIGN_CODE.encode(), BENIGN_CODE.encode()]

    def test_no_ungated_job_submission_in_evo_tree(self):
        """The only /api/v1/jobs POSTs in the Evo tree live behind the gate."""
        evo_root = Path(reward_func_utils.__file__).resolve().parents[1]
        gated = Path(reward_func_utils.__file__).resolve()
        offenders = []
        for path in list(evo_root.glob("tts_search/*.py")) + list(
            (evo_root / "third_party/aira-evo/examples/mle_bench").glob("*.py")
        ):
            if path.resolve() == gated:
                continue
            text = path.read_text(encoding="utf-8")
            if "/api/v1/jobs" in text and "post" in text.lower():
                offenders.append(str(path))
        assert offenders == []
