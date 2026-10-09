"""Dispatcher batch-2 repair tests (US-006): worker control authentication
and staged source-identity binding.

Covers:
  * SandboxHTTPController sends the X-RSI-Control-Key header (from the
    dispatcher-only key file) on every worker control call, and the key
    loader fails closed when configured-but-unreadable/too short.
  * verify_staged_source_identity binds the admission gate evidence to the
    exact staged bytes: absent evidence, policy drift and byte divergence
    all fail closed with machine-readable reasons.
  * execute_job in local-scratch mode never issues the candidate execution
    command when the identity check fails, and runs end-to-end when the
    staged bytes match the gated evidence.

Offline only: faked worker transport/Redis/DB, no containers, no network,
no candidate code execution.
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
_TEST_STORAGE = Path(tempfile.mkdtemp(prefix="us006-dispatch-ctrl-"))
_TEST_CONFIG = _TEST_STORAGE / "sandbox_config.json"
_TEST_CONFIG.write_text(
    json.dumps({"cpu": {"endpoints": ["http://worker.invalid:8080"]}, "gpu": {"ranges": []}}),
    encoding="utf-8",
)
os.environ.setdefault("STORAGE_PATH", str(_TEST_STORAGE))
os.environ.setdefault("SANDBOX_CONFIG_FILE", str(_TEST_CONFIG))
# Match test_dispatcher_external_eval: whichever module imports
# task_dispatcher first must initialize it with the evaluator enabled.
os.environ.setdefault("EXTERNAL_EVALUATOR_ENABLED", "1")
os.environ.setdefault(
    "EVALUATOR_REGISTRY_PATH", str(REPO_ROOT / "research" / "evaluator" / "registry.v1.json")
)

sys.path.insert(0, str(REPO_ROOT / "sandbox-controller" / "task_dispatcher"))

import task_dispatcher as td  # noqa: E402


def _real_policy():
    assert td.SOURCE_GATE is not None, "source gate must be importable in tests"
    return td.SOURCE_GATE.load_policy()


def _evidence_for(code_bytes: bytes) -> dict:
    policy = _real_policy()
    return {
        "allowed": True,
        "policy_version": policy.policy_version,
        "policy_sha256": policy.policy_sha256,
        "source_sha256": hashlib.sha256(code_bytes).hexdigest(),
        "security_boundary": False,
        "check_kind": "static_admission_filter",
    }


# --- verify_staged_source_identity -----------------------------------------


def test_identity_absent_evidence_fails_closed(tmp_path):
    staged = tmp_path / "main.py"
    staged.write_text("print('x')\n", encoding="utf-8")
    for evidence in (None, {}, {"allowed": False}, {"allowed": "true"}):
        reason, _ = td.verify_staged_source_identity({"source_gate": evidence}, staged)
        assert reason == "source_identity_absent", evidence
    reason, _ = td.verify_staged_source_identity({}, staged)
    assert reason == "source_identity_absent"


def test_identity_policy_mismatch_fails_closed(tmp_path):
    code = b"print('candidate')\n"
    staged = tmp_path / "main.py"
    staged.write_bytes(code)
    evidence = _evidence_for(code)
    evidence["policy_sha256"] = "0" * 64
    reason, detail = td.verify_staged_source_identity({"source_gate": evidence}, staged)
    assert reason == "source_identity_policy_mismatch"
    assert "policy_sha256" in (detail or "")
    evidence = _evidence_for(code)
    evidence["policy_version"] = "source-gate.v999"
    reason, _ = td.verify_staged_source_identity({"source_gate": evidence}, staged)
    assert reason == "source_identity_policy_mismatch"


def test_identity_byte_divergence_fails_closed(tmp_path):
    evidence = _evidence_for(b"print('gated bytes')\n")
    staged = tmp_path / "main.py"
    staged.write_bytes(b"print('swapped after admission')\n")
    reason, _ = td.verify_staged_source_identity({"source_gate": evidence}, staged)
    assert reason == "source_identity_mismatch"


def test_identity_entrypoint_hash_used_when_present(tmp_path):
    code = b"print('uploaded entrypoint')\n"
    evidence = _evidence_for(b"tree-hash-placeholder-irrelevant")
    evidence["entrypoint"] = "pkg/main.py"
    evidence["entrypoint_sha256"] = hashlib.sha256(code).hexdigest()
    staged = tmp_path / "main.py"
    staged.write_bytes(code)
    assert td.verify_staged_source_identity({"source_gate": evidence}, staged) == (None, None)


def test_identity_match_passes(tmp_path):
    code = b"print('candidate')\n"
    staged = tmp_path / "main.py"
    staged.write_bytes(code)
    assert td.verify_staged_source_identity({"source_gate": _evidence_for(code)}, staged) == (None, None)


def test_identity_gate_unavailable_fails_closed(monkeypatch, tmp_path):
    code = b"print('candidate')\n"
    staged = tmp_path / "main.py"
    staged.write_bytes(code)
    evidence = _evidence_for(code)
    monkeypatch.setattr(td, "SOURCE_GATE", None)
    reason, _ = td.verify_staged_source_identity({"source_gate": evidence}, staged)
    assert reason == "source_identity_gate_unavailable"


# --- worker control key loading + header wiring -----------------------------


def test_control_key_loader_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(td, "WORKER_CONTROL_KEY_FILE", "")
    assert td._load_worker_control_key() is None
    monkeypatch.setattr(td, "WORKER_CONTROL_KEY_FILE", str(tmp_path / "absent"))
    with pytest.raises(RuntimeError):
        td._load_worker_control_key()
    key_file = tmp_path / "key"
    key_file.write_text("short", encoding="utf-8")
    monkeypatch.setattr(td, "WORKER_CONTROL_KEY_FILE", str(key_file))
    with pytest.raises(RuntimeError):
        td._load_worker_control_key()
    key_file.write_text("k" * 64 + "\n", encoding="utf-8")
    assert td._load_worker_control_key() == "k" * 64


def test_control_headers_only_with_key(monkeypatch):
    monkeypatch.setattr(td, "WORKER_CONTROL_KEY", None)
    assert td._worker_control_headers() is None
    monkeypatch.setattr(td, "WORKER_CONTROL_KEY", "secret-control-key")
    assert td._worker_control_headers() == {"X-RSI-Control-Key": "secret-control-key"}


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_controller_sends_control_key_on_post_and_delete(monkeypatch):
    captured: list[dict] = []

    def fake_post(url, json=None, timeout=None, headers=None):
        captured.append({"method": "POST", "url": url, "headers": headers})
        return _FakeResponse({"data": {"session_id": "s-1", "status": "running"}})

    def fake_delete(url, timeout=None, headers=None):
        captured.append({"method": "DELETE", "url": url, "headers": headers})
        return _FakeResponse({"data": {"session_id": "s-1", "removed": True, "cleanup_verified": True}})

    monkeypatch.setattr(td.httpx, "post", fake_post)
    monkeypatch.setattr(td.httpx, "delete", fake_delete)
    monkeypatch.setattr(td, "WORKER_CONTROL_KEY", "secret-control-key")

    controller = td.SandboxHTTPController("http://worker.invalid:8080")
    payload = controller.exec_command(command="true", exec_dir=None)
    assert payload["session_id"] == "s-1"
    assert controller.cleanup("s-1") is True

    assert [c["method"] for c in captured] == ["POST", "DELETE"]
    for call in captured:
        assert call["headers"] == {"X-RSI-Control-Key": "secret-control-key"}
        assert "secret-control-key" not in call["url"]


def test_cleanup_acknowledgement_must_be_explicit(monkeypatch):
    """cleanup() is True ONLY on a 200 verified acknowledgement."""
    monkeypatch.setattr(td, "WORKER_CONTROL_KEY", "k" * 32)
    controller = td.SandboxHTTPController("http://worker.invalid:8080")

    cases = [
        (_FakeResponse({"data": {"removed": True}}), False),  # no cleanup proof
        (_FakeResponse({"data": {"removed": True, "cleanup_verified": False}}), False),
        (_FakeResponse({"data": {"removed": False, "cleanup_verified": True}}), False),
        (_FakeResponse({"data": {"removed": True, "cleanup_verified": True}}), True),
        (_FakeResponse({"error": "worker_cleanup_unproven"}, status_code=503), False),
    ]
    for response, expected in cases:
        monkeypatch.setattr(td.httpx, "delete", lambda *a, _r=response, **k: _r)
        assert controller.cleanup("s-1") is expected

    def unreachable(*args, **kwargs):
        raise td.httpx.ConnectError("offline")

    monkeypatch.setattr(td.httpx, "delete", unreachable)
    assert controller.cleanup("s-1") is False


def test_kill_returns_ack_or_raises(monkeypatch):
    """kill() returns the worker ack snapshot; HTTP failures raise."""
    monkeypatch.setattr(td, "WORKER_CONTROL_KEY", "k" * 32)
    controller = td.SandboxHTTPController("http://worker.invalid:8080")

    monkeypatch.setattr(
        td.httpx,
        "post",
        lambda *a, **k: _FakeResponse({"data": {"session_id": "s-1", "status": "terminated", "cleanup_verified": True}}),
    )
    ack = controller.kill("s-1")
    assert ack is not None and ack.get("cleanup_verified") is True

    def http_error(*args, **kwargs):
        response = _FakeResponse({}, status_code=503)

        def raise_for_status():
            raise td.httpx.HTTPStatusError("503", request=None, response=response)

        response.raise_for_status = raise_for_status
        return response

    monkeypatch.setattr(td.httpx, "post", http_error)
    with pytest.raises(td.SandboxHTTPError):
        controller.kill("s-1")


# --- execute_job wiring: no execution without proven identity ---------------


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


CANDIDATE_CODE = b"print('candidate writes submission')\n"


@pytest.fixture()
def scratch_harness(monkeypatch, tmp_path):
    commands: list[str] = []
    exec_calls: list[dict] = []

    def fake_exec(*, worker_endpoint, command, exec_dir, job_id, redis_client, db_conn,
                  job_deadline, on_started=None, exec_class=None, job_root=None):
        commands.append(command)
        exec_calls.append({"command": command, "exec_class": exec_class, "job_root": job_root})
        if on_started is not None:
            on_started("fake-session")
        return td.SandboxResultWrapper(
            {"exit_code": 0, "status": "completed", "output": "", "cleanup_verified": True}
        )

    monkeypatch.setattr(td, "ENABLE_LOCAL_JOB_SCRATCH", True)
    monkeypatch.setattr(td, "LOCAL_SCRATCH_ROOT", tmp_path / "scratch")
    monkeypatch.setattr(td, "LOCAL_DISPATCHER_LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(td, "execute_shell_command_with_deadline", fake_exec)
    monkeypatch.setattr(td, "wait_for_file_exists", lambda *a, **k: True)
    monkeypatch.setattr(td, "EXTERNAL_EVALUATOR_ENABLED", True)
    monkeypatch.setattr(
        td,
        "score_job_externally",
        lambda **kwargs: (0.5, td.RESULT_SUCCESS, None, {"evaluator": "external_trusted"}),
    )
    return {
        "commands": commands,
        "exec_calls": exec_calls,
        "conn": FakeConn(),
        "redis": FakeRedis(),
        "tmp": tmp_path,
    }


def _make_job(tmp_path: Path, evidence: object) -> dict:
    job_id = "job-identity"
    job_dir = tmp_path / job_id
    code_dir = job_dir / "code"
    code_dir.mkdir(parents=True)
    code_file = code_dir / "main.py"
    code_file.write_bytes(CANDIDATE_CODE)
    job = {
        "job_id": job_id,
        "task_id": "hello_synth",
        "code_file_path": str(code_file),
        "job_storage_dir": str(job_dir),
        "data_dir": "/data/hello_synth",
        "environment": {"EXECUTION_MODE": "shell"},
        "timeout": 600,
    }
    if evidence is not None:
        job["source_gate"] = evidence
    return job


def _stage_bytes(harness, code: bytes) -> None:
    """Simulate the worker-side staging copy with given bytes."""
    staged_dir = td.local_code_dir("job-identity", harness["tmp"] / "job-identity")
    staged_dir.mkdir(parents=True, exist_ok=True)
    (staged_dir / "main.py").write_bytes(code)


def test_execute_job_without_identity_evidence_never_executes(scratch_harness):
    job = _make_job(scratch_harness["tmp"], None)
    _stage_bytes(scratch_harness, CANDIDATE_CODE)
    td.execute_job(scratch_harness["redis"], scratch_harness["conn"], "http://worker.invalid:8080", job)
    status, result_raw = scratch_harness["conn"].updates[-1][0], scratch_harness["conn"].updates[-1][1]
    result = json.loads(result_raw)
    assert status == "failed"
    assert result["result"] == td.RESULT_SOURCE_IDENTITY
    assert "source_identity_absent" in (result.get("run_log") or "")
    # Only the trusted stage command ran; the candidate exec command never did.
    assert len(scratch_harness["commands"]) == 1
    assert "python" not in scratch_harness["commands"][0]


def test_execute_job_with_tampered_staged_bytes_never_executes(scratch_harness):
    job = _make_job(scratch_harness["tmp"], _evidence_for(CANDIDATE_CODE))
    _stage_bytes(scratch_harness, b"print('tampered after admission')\n")
    td.execute_job(scratch_harness["redis"], scratch_harness["conn"], "http://worker.invalid:8080", job)
    status, result_raw = scratch_harness["conn"].updates[-1][0], scratch_harness["conn"].updates[-1][1]
    result = json.loads(result_raw)
    assert status == "failed"
    assert result["result"] == td.RESULT_SOURCE_IDENTITY
    assert "source_identity_mismatch" in (result.get("run_log") or "")
    assert len(scratch_harness["commands"]) == 1


def test_execute_job_with_matching_identity_runs_and_scores(scratch_harness):
    job = _make_job(scratch_harness["tmp"], _evidence_for(CANDIDATE_CODE))
    _stage_bytes(scratch_harness, CANDIDATE_CODE)
    td.execute_job(scratch_harness["redis"], scratch_harness["conn"], "http://worker.invalid:8080", job)
    status, result_raw = scratch_harness["conn"].updates[-1][0], scratch_harness["conn"].updates[-1][1]
    result = json.loads(result_raw)
    assert status == "completed"
    assert result["score"] == 0.5
    # stage + candidate exec both ran under the bound identity.
    assert len(scratch_harness["commands"]) == 2
    # Execution classes are routed explicitly: the dispatcher-generated
    # stage command runs as trusted staging, the candidate command runs
    # under the per-job sandbox root (trampoline confinement, US-008).
    stage_call, candidate_call = scratch_harness["exec_calls"]
    assert stage_call["exec_class"] == "staging"
    assert stage_call["job_root"] is None
    assert candidate_call["exec_class"] == "candidate"
    expected_root = str(td.local_job_root("job-identity", scratch_harness["tmp"] / "job-identity"))
    assert candidate_call["job_root"] == expected_root
