"""API admission gate tests (US-004).

Exercises the WSL sandbox-controller api_server admission path with spies
on the DB insert and the Redis enqueue: a denied submission must neither
be written to the job store, inserted into the DB, nor enqueued, and every
rejection must be machine-readable. Existing job query/cancel semantics
are preserved. No containers, no network, no candidate code execution.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_STORAGE = Path(tempfile.mkdtemp(prefix="us004-api-storage-"))
os.environ.setdefault("STORAGE_PATH", str(_TEST_STORAGE))
os.environ.setdefault("UPLOAD_ROOT", str(_TEST_STORAGE / "uploads"))
os.environ.setdefault("SANDBOX_API_KEYS", "us004-test-key")
os.environ.setdefault("ENABLE_OBS", "0")

sys.path.insert(0, str(REPO_ROOT / "sandbox-controller" / "api_server"))

import api_server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

API_HEADERS = {"X-API-Key": "us004-test-key"}
BENIGN_CODE = (
    "import os\n"
    "import pandas as pd\n"
    "train = pd.read_csv(os.path.join(os.environ['DATA_DIR'], 'train.csv'))\n"
    "print('rows:', len(train))\n"
)


class FakeCursor:
    def __init__(self, conn: "FakeConn") -> None:
        self.conn = conn
        self.rowcount = 0
        self._row = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args) -> None:
        return None

    def execute(self, sql, params=None) -> None:
        normalized = " ".join(str(sql).split()).upper()
        if normalized.startswith("INSERT INTO JOBS"):
            self.conn.inserted.append(params)
        elif normalized.startswith("UPDATE JOBS"):
            self.rowcount = self.conn.update_rowcount
        self._row = self.conn.next_row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return [] if self._row is None else [self._row]


class FakeConn:
    def __init__(self) -> None:
        self.inserted: list = []
        self.next_row = None
        self.update_rowcount = 0
        self.closed = False
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        pass


class FakeRedis:
    def __init__(self) -> None:
        self.pushes: list[tuple[str, str]] = []
        self.queues: dict[str, list[str]] = {}
        self.flags: dict[str, str] = {}

    def rpush(self, queue: str, payload: str) -> None:
        self.pushes.append((queue, payload))
        self.queues.setdefault(queue, []).append(payload)

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        return list(self.queues.get(key, []))

    def lrem(self, key: str, count: int, value: str) -> int:
        try:
            self.queues.get(key, []).remove(value)
            return 1
        except ValueError:
            return 0

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.flags[key] = value

    def scan_iter(self, pattern: str):
        return iter(())

    def get(self, key: str):
        return None

    def ping(self) -> bool:
        return True


@pytest.fixture()
def spies(monkeypatch):
    conn = FakeConn()
    redis_fake = FakeRedis()

    @contextmanager
    def _fake_get_db_connection():
        yield conn

    monkeypatch.setattr(api_server, "get_db_connection", _fake_get_db_connection)
    monkeypatch.setattr(api_server, "redis_client", redis_fake)
    return conn, redis_fake


@pytest.fixture()
def client():
    return TestClient(api_server.app)


def _submit_payload(**overrides) -> dict:
    payload = {
        "name": "us004-gate-test",
        "code": BENIGN_CODE,
        "data_dir": "/data/tasks/hello_synth",
        "timeout": 60,
        "resource_type": "cpu",
        "priority": 1,
        "environment": {"EXECUTION_MODE": "shell"},
    }
    payload.update(overrides)
    return payload


def _job_dirs() -> list[Path]:
    jobs_root = api_server.STORAGE_PATH / "jobs"
    if not jobs_root.exists():
        return []
    return [p for p in jobs_root.rglob("job_*") if p.is_dir()]


def _make_upload(files: dict, upload_id: str) -> Path:
    root = api_server.UPLOAD_ROOT / upload_id
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


class TestDeniedSubmissions:
    def test_malicious_inline_code_denied_before_writes(self, client, spies):
        conn, redis_fake = spies
        before = set(_job_dirs())
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(code="import subprocess\nsubprocess.call(['id'])\n"),
            headers=API_HEADERS,
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["error"] == "source_gate_denied"
        gate = detail["gate"]
        assert gate["allowed"] is False
        assert gate["security_boundary"] is False
        assert any(d["rule"] == "denied_marker" for d in gate["denials"])
        assert gate["policy_version"] == "source-gate.v2"
        assert redis_fake.pushes == []
        assert conn.inserted == []
        assert set(_job_dirs()) == before

    def test_execution_command_alternate_path_denied(self, client, spies):
        conn, redis_fake = spies
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(execution_command="bash run.sh"),
            headers=API_HEADERS,
        )
        assert resp.status_code == 422
        denials = resp.json()["detail"]["gate"]["denials"]
        assert any(d["rule"] == "execution_command_denied" for d in denials)
        assert redis_fake.pushes == []
        assert conn.inserted == []

    @pytest.mark.parametrize("key", ["LD_PRELOAD", "JOB_OUTPUT_DIR", "SOURCE_GATE_POLICY_PATH"])
    def test_dangerous_environment_keys_denied(self, client, spies, key):
        conn, redis_fake = spies
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(environment={"EXECUTION_MODE": "shell", key: "x"}),
            headers=API_HEADERS,
        )
        assert resp.status_code == 422
        denials = resp.json()["detail"]["gate"]["denials"]
        assert any(d["rule"] == "environment_key_denied" for d in denials)
        assert redis_fake.pushes == []
        assert conn.inserted == []

    def test_unsafe_requirement_denied(self, client, spies):
        conn, redis_fake = spies
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(requirements=["git+https://example.invalid/r.git"]),
            headers=API_HEADERS,
        )
        assert resp.status_code == 422
        denials = resp.json()["detail"]["gate"]["denials"]
        assert any(d["rule"] == "requirement_not_allowlisted" for d in denials)
        assert redis_fake.pushes == []
        assert conn.inserted == []

    def test_working_dir_traversal_denied(self, client, spies):
        conn, redis_fake = spies
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(working_dir=str(api_server.STORAGE_PATH / ".." / "evaluation")),
            headers=API_HEADERS,
        )
        assert resp.status_code == 422
        rules = [d["rule"] for d in resp.json()["detail"]["gate"]["denials"]]
        assert "path_traversal" in rules or "path_protected_component" in rules
        assert redis_fake.pushes == []
        assert conn.inserted == []

    def test_data_dir_traversal_denied(self, client, spies):
        conn, redis_fake = spies
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(data_dir="/data/../evaluation"),
            headers=API_HEADERS,
        )
        assert resp.status_code == 422
        rules = [d["rule"] for d in resp.json()["detail"]["gate"]["denials"]]
        assert "path_traversal" in rules or "path_protected_component" in rules
        assert redis_fake.pushes == []
        assert conn.inserted == []

    @pytest.mark.parametrize("key", ["PATH", "HOME", "PYTHONHOME", "PYTHONUSERBASE", "WORKER_CONTROL_TOKEN"])
    def test_platform_and_worker_control_env_keys_denied(self, client, spies, key):
        conn, redis_fake = spies
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(environment={"EXECUTION_MODE": "shell", key: "x"}),
            headers=API_HEADERS,
        )
        assert resp.status_code == 422
        denials = resp.json()["detail"]["gate"]["denials"]
        assert any(d["rule"] == "environment_key_denied" for d in denials)
        assert redis_fake.pushes == []
        assert conn.inserted == []

    def test_working_dir_outside_captured_artifact_denied(self, client, spies):
        conn, redis_fake = spies
        upload = _make_upload({"main.py": BENIGN_CODE}, "upload_wd_outside")
        other = api_server.STORAGE_PATH / "shared-workdir"
        other.mkdir(parents=True, exist_ok=True)
        before = set(_job_dirs())
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(
                code=None,
                code_file_path=str(upload / "main.py"),
                working_dir=str(other),
            ),
            headers=API_HEADERS,
        )
        assert resp.status_code == 422
        denials = resp.json()["detail"]["gate"]["denials"]
        assert any(d["rule"] == "working_dir_outside_artifact" for d in denials)
        assert redis_fake.pushes == []
        assert conn.inserted == []
        assert set(_job_dirs()) == before

    def test_inline_code_with_working_dir_denied(self, client, spies):
        conn, redis_fake = spies
        before = set(_job_dirs())
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(working_dir=str(api_server.UPLOAD_ROOT)),
            headers=API_HEADERS,
        )
        assert resp.status_code == 422
        denials = resp.json()["detail"]["gate"]["denials"]
        assert any(d["rule"] == "working_dir_outside_artifact" for d in denials)
        assert redis_fake.pushes == []
        assert conn.inserted == []
        assert set(_job_dirs()) == before

    def test_missing_policy_fails_closed(self, client, spies, monkeypatch, tmp_path):
        conn, redis_fake = spies
        monkeypatch.setenv("SOURCE_GATE_POLICY_PATH", str(tmp_path / "missing.json"))
        resp = client.post("/api/v1/jobs", json=_submit_payload(), headers=API_HEADERS)
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "source_gate_policy_unavailable"
        assert redis_fake.pushes == []
        assert conn.inserted == []


class TestAllowedSubmissions:
    def test_benign_inline_code_enqueued_with_gate_evidence(self, client, spies):
        conn, redis_fake = spies
        resp = client.post("/api/v1/jobs", json=_submit_payload(), headers=API_HEADERS)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["job_id"].startswith("job_")
        assert body["status"] == "queued"
        assert len(redis_fake.pushes) == 1
        assert len(conn.inserted) == 1
        job_data = json.loads(redis_fake.pushes[0][1])
        evidence = job_data["source_gate"]
        assert evidence["allowed"] is True
        assert evidence["policy_version"] == "source-gate.v2"
        assert evidence["policy_sha256"]
        # Bound source hash matches the exact submitted bytes.
        tree_file = Path(job_data["code_file_path"])
        assert tree_file.read_bytes() == BENIGN_CODE.encode()
        file_hash = hashlib.sha256(BENIGN_CODE.encode()).hexdigest()
        expected = hashlib.sha256(b"main.py\0" + file_hash.encode("ascii") + b"\n")
        assert evidence["source_sha256"] == expected.hexdigest()

    def test_benign_code_file_path_enqueued(self, client, spies, tmp_path):
        conn, redis_fake = spies
        src_dir = api_server.STORAGE_PATH / "incoming"
        src_dir.mkdir(parents=True, exist_ok=True)
        src = src_dir / "train.py"
        src.write_text(BENIGN_CODE, encoding="utf-8")
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(code=None, code_file_path=str(src)),
            headers=API_HEADERS,
        )
        assert resp.status_code == 200, resp.text
        assert len(redis_fake.pushes) == 1
        job_data = json.loads(redis_fake.pushes[0][1])
        final_path = Path(job_data["code_file_path"])
        assert final_path.read_text(encoding="utf-8") == BENIGN_CODE
        assert str(api_server.STORAGE_PATH / "jobs") in str(final_path)

    def test_malicious_code_file_path_denied_and_cleaned(self, client, spies):
        conn, redis_fake = spies
        src_dir = api_server.STORAGE_PATH / "incoming2"
        src_dir.mkdir(parents=True, exist_ok=True)
        src = src_dir / "evil.py"
        src.write_text("import os\nos.system('id')\n", encoding="utf-8")
        before = set(_job_dirs())
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(code=None, code_file_path=str(src)),
            headers=API_HEADERS,
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "source_gate_denied"
        assert redis_fake.pushes == []
        assert conn.inserted == []
        assert set(_job_dirs()) == before

    def test_upload_admission_retains_original_and_materializes_capture(self, client, spies):
        conn, redis_fake = spies
        upload = _make_upload(
            {"main.py": BENIGN_CODE, "data.txt": "payload-bytes\n"}, "upload_keep"
        )
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(code=None, code_file_path=str(upload / "main.py")),
            headers=API_HEADERS,
        )
        assert resp.status_code == 200, resp.text
        assert len(redis_fake.pushes) == 1
        # Original upload retained byte-for-byte (never relocated/mutated).
        assert (upload / "main.py").read_text(encoding="utf-8") == BENIGN_CODE
        assert (upload / "data.txt").read_text(encoding="utf-8") == "payload-bytes\n"
        job_data = json.loads(redis_fake.pushes[0][1])
        final_path = Path(job_data["code_file_path"])
        assert final_path.read_text(encoding="utf-8") == BENIGN_CODE
        assert str(api_server.STORAGE_PATH / "jobs") in str(final_path)
        # Whole upload tree materialized once under the job code root.
        assert final_path.parent.name == "upload_keep"
        assert (final_path.parent / "data.txt").read_text(encoding="utf-8") == "payload-bytes\n"
        # Entrypoint identity is bound into the gate evidence.
        evidence = job_data["source_gate"]
        assert evidence["entrypoint"] == "main.py"
        assert evidence["entrypoint_sha256"] == hashlib.sha256(BENIGN_CODE.encode()).hexdigest()

    def test_upload_changed_after_capture_cannot_change_executed_bytes(
        self, client, spies, monkeypatch
    ):
        """Immutable capture: mutating the upload after capture is inert."""
        conn, redis_fake = spies
        upload = _make_upload({"main.py": BENIGN_CODE}, "upload_mutated")
        original_capture = api_server.source_snapshot.capture_source

        def capturing_then_mutating(path, policy, *, allowed_root=None):
            snapshot = original_capture(path, policy, allowed_root=allowed_root)
            (upload / "main.py").write_text("import subprocess\n", encoding="utf-8")
            return snapshot

        monkeypatch.setattr(
            api_server.source_snapshot, "capture_source", capturing_then_mutating
        )
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(code=None, code_file_path=str(upload / "main.py")),
            headers=API_HEADERS,
        )
        assert resp.status_code == 200, resp.text
        assert len(redis_fake.pushes) == 1
        job_data = json.loads(redis_fake.pushes[0][1])
        final_path = Path(job_data["code_file_path"])
        # Executed bytes are the captured (benign) bytes...
        assert final_path.read_text(encoding="utf-8") == BENIGN_CODE
        # ...even though the upload itself now holds denied content.
        assert "subprocess" in (upload / "main.py").read_text(encoding="utf-8")

    def test_materialized_tamper_fails_closed_before_enqueue(
        self, client, spies, monkeypatch
    ):
        """Any divergence between captured verdict and materialized bytes is fatal."""
        conn, redis_fake = spies
        upload = _make_upload({"main.py": BENIGN_CODE}, "upload_tamper")
        original_materialize = api_server.source_snapshot.SourceSnapshot.materialize

        def tampering_materialize(self, code_root):
            target = original_materialize(self, code_root)
            (target / "main.py").write_text("import subprocess\n", encoding="utf-8")
            return target

        monkeypatch.setattr(
            api_server.source_snapshot.SourceSnapshot,
            "materialize",
            tampering_materialize,
        )
        before = set(_job_dirs())
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(code=None, code_file_path=str(upload / "main.py")),
            headers=API_HEADERS,
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error"] == "source_gate_verify_failed"
        assert redis_fake.pushes == []
        assert conn.inserted == []
        assert set(_job_dirs()) == before
        # Original upload untouched by the failed admission.
        assert (upload / "main.py").read_text(encoding="utf-8") == BENIGN_CODE

    def test_upload_directory_symlink_denied(self, client, spies):
        conn, redis_fake = spies
        upload = _make_upload({"main.py": BENIGN_CODE}, "upload_symlink")
        outside = api_server.STORAGE_PATH / "elsewhere"
        outside.mkdir(parents=True, exist_ok=True)
        (outside / "payload.py").write_text("import subprocess\n", encoding="utf-8")
        (upload / "extra").symlink_to(outside, target_is_directory=True)
        before = set(_job_dirs())
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(code=None, code_file_path=str(upload / "main.py")),
            headers=API_HEADERS,
        )
        assert resp.status_code == 422
        denials = resp.json()["detail"]["gate"]["denials"]
        assert any(d["rule"] == "symlink_denied" for d in denials)
        assert redis_fake.pushes == []
        assert conn.inserted == []
        assert set(_job_dirs()) == before
        assert (upload / "main.py").read_text(encoding="utf-8") == BENIGN_CODE

    def test_upload_package_entrypoint_and_working_dir_mapping(self, client, spies):
        conn, redis_fake = spies
        upload = _make_upload(
            {
                "pkg/__init__.py": '"""pkg"""\n',
                "pkg/train.py": BENIGN_CODE,
                "assets/labels.txt": "a\n",
            },
            "upload_pkg",
        )
        resp = client.post(
            "/api/v1/jobs",
            json=_submit_payload(
                code=None,
                code_file_path=str(upload / "pkg" / "train.py"),
                working_dir=str(upload / "pkg"),
            ),
            headers=API_HEADERS,
        )
        assert resp.status_code == 200, resp.text
        assert len(redis_fake.pushes) == 1
        job_data = json.loads(redis_fake.pushes[0][1])
        final_workdir = Path(job_data["environment"]["EXECUTION_WORKDIR"])
        # working_dir mapped inside the materialized captured tree.
        assert final_workdir.name == "pkg"
        assert str(api_server.STORAGE_PATH / "jobs") in str(final_workdir)
        assert (final_workdir / "train.py").read_text(encoding="utf-8") == BENIGN_CODE
        assert Path(job_data["code_file_path"]) == final_workdir / "train.py"
        # Sibling assets of the upload tree came along with the capture.
        assert (final_workdir.parent / "assets" / "labels.txt").exists()
        evidence = job_data["source_gate"]
        assert evidence["entrypoint"] == "pkg/train.py"


class TestExistingSemanticsPreserved:
    def test_query_unknown_job_is_404(self, client, spies):
        conn, _ = spies
        conn.next_row = None
        resp = client.get("/api/v1/jobs/job_unknown", headers=API_HEADERS)
        assert resp.status_code == 404

    def test_cancel_unknown_job_is_404(self, client, spies):
        conn, _ = spies
        conn.update_rowcount = 0
        resp = client.delete("/api/v1/jobs/job_unknown", headers=API_HEADERS)
        assert resp.status_code == 404

    def test_cancel_queued_job_succeeds(self, client, spies):
        conn, redis_fake = spies
        conn.update_rowcount = 1
        resp = client.delete("/api/v1/jobs/job_abc", headers=API_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["message"] == "Job cancelled successfully"
        assert redis_fake.flags.get("job:job_abc:cancelled") == "1"

    def test_auth_key_comes_from_environment(self, client, spies):
        resp = client.post(
            "/api/v1/jobs", json=_submit_payload(), headers={"X-API-Key": "wrong"}
        )
        assert resp.status_code == 401
