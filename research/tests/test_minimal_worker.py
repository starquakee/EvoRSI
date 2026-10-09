"""Minimal worker service protocol + batch-2 security tests (US-006).

Exercises deploy/rsi-trustworthy/worker/minimal_worker.py directly on
127.0.0.1 with an ephemeral port (trusted first-party code, no candidate
code, no containers): control-key authentication, single execution slot,
exec/wait/view/kill/cleanup semantics, exit codes, bounded output capture,
process-group kill, and the post-exit cleanup gate.

Privilege/container-boundary behaviour (candidate uid drop, real /proc uid
sweep) is verified here with MOCKS only: no host-wide process enumeration
or privilege changes ever run on the development host. The REAL
detached-child cleanup and candidate-isolation proofs run inside the new
worker container via deploy/rsi-trustworthy/validate_stack.py.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = REPO_ROOT / "deploy" / "rsi-trustworthy" / "worker" / "minimal_worker.py"

spec = importlib.util.spec_from_file_location("minimal_worker", WORKER_PATH)
assert spec is not None and spec.loader is not None
worker = importlib.util.module_from_spec(spec)
sys.modules["minimal_worker"] = worker
spec.loader.exec_module(worker)

TEST_KEY = b"test-control-key-0123456789abcdef"
CANDIDATE_UID = 65432
CANDIDATE_GID = 65432


def make_config(**overrides):
    params = {
        "control_key": TEST_KEY,
        "candidate_uid": CANDIDATE_UID,
        "candidate_gid": CANDIDATE_GID,
        "guard_active": False,
    }
    params.update(overrides)
    return worker.WorkerConfig(**params)


@pytest.fixture()
def server_url():
    server = worker.create_server("127.0.0.1", 0, config=make_config())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _opener() -> urllib.request.OpenerDirector:
    # Local probes must never route through process proxy settings.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _post(url: str, path: str, payload: dict, *, key: bytes | None = TEST_KEY) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["X-RSI-Control-Key"] = key.decode("utf-8")
    request = urllib.request.Request(url + path, data=body, headers=headers, method="POST")
    try:
        with _opener().open(request, timeout=15) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _delete(url: str, path: str, *, key: bytes | None = TEST_KEY) -> tuple[int, dict]:
    headers = {}
    if key is not None:
        headers["X-RSI-Control-Key"] = key.decode("utf-8")
    request = urllib.request.Request(url + path, headers=headers, method="DELETE")
    try:
        with _opener().open(request, timeout=15) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _exec(url: str, command: str, exec_dir=None) -> str:
    status, payload = _post(
        url, "/v1/shell/exec", {"id": "", "command": command, "exec_dir": exec_dir, "async_mode": True, "timeout": 30}
    )
    assert status == 200, payload
    session_id = payload["data"]["session_id"]
    assert session_id
    return session_id


def _wait_done(url: str, session_id: str, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        status, payload = _post(url, "/v1/shell/wait", {"id": session_id, "seconds": 1})
        assert status == 200, payload
        data = payload["data"]
        if data["status"] != "running":
            return data
        assert time.monotonic() < deadline, f"session {session_id} did not finish"


# --- control-key authentication -------------------------------------------


def test_healthz_open_without_key(server_url):
    with _opener().open(server_url + "/healthz", timeout=5) as response:
        assert response.status == 200
        assert json.loads(response.read().decode("utf-8"))["data"]["status"] == "ok"


def test_unauthenticated_control_calls_denied(server_url):
    for path, payload in (
        ("/v1/shell/exec", {"command": "echo hi"}),
        ("/v1/shell/wait", {"id": "x"}),
        ("/v1/shell/view", {"id": "x"}),
        ("/v1/shell/kill", {"id": "x"}),
    ):
        status, body = _post(server_url, path, payload, key=None)
        assert status == 401, (path, body)
        assert body["error"] == "worker_control_unauthorized"
        status, body = _post(server_url, path, payload, key=b"wrong-key-0000000000000000")
        assert status == 401, (path, body)
    status, body = _delete(server_url, "/v1/shell/sessions/whatever", key=None)
    assert status == 401 and body["error"] == "worker_control_unauthorized"
    # And nothing was executed by the unauthenticated exec attempt.
    status, body = _post(server_url, "/v1/shell/view", {"id": "whatever"})
    assert status == 404


# --- dispatcher shell protocol (authenticated) -----------------------------


def test_exec_echo_completes_with_output(server_url, tmp_path):
    session_id = _exec(server_url, "echo hello-trustworthy", exec_dir=str(tmp_path))
    data = _wait_done(server_url, session_id)
    assert data["status"] == "completed"
    assert data["exit_code"] == 0
    assert "hello-trustworthy" in data["output"]
    assert data["cleanup_verified"] is True


def test_exec_dir_is_working_directory(server_url, tmp_path):
    session_id = _exec(server_url, "pwd", exec_dir=str(tmp_path))
    data = _wait_done(server_url, session_id)
    assert data["output"].strip() == str(tmp_path)


def test_nonzero_exit_code_reported(server_url):
    session_id = _exec(server_url, "echo oops >&2; exit 3")
    data = _wait_done(server_url, session_id)
    assert data["status"] == "completed"
    assert data["exit_code"] == 3
    assert "oops" in data["stderr"]


def test_exec_requires_command_and_existing_dir(server_url):
    status, payload = _post(server_url, "/v1/shell/exec", {"command": "  "})
    assert status == 400 and payload["error"] == "command_required"
    status, payload = _post(
        server_url, "/v1/shell/exec", {"command": "true", "exec_dir": "/nonexistent-dir-xyz"}
    )
    assert status == 400 and payload["error"] == "exec_dir_not_found"


def test_unknown_session_is_404(server_url):
    for path in ("/v1/shell/wait", "/v1/shell/view", "/v1/shell/kill"):
        status, payload = _post(server_url, path, {"id": "no-such-session"})
        assert status == 404, (path, payload)
    status, _ = _delete(server_url, "/v1/shell/sessions/no-such-session")
    assert status == 404


def test_wait_reports_running_then_blocks(server_url):
    session_id = _exec(server_url, "sleep 2; echo done")
    start = time.monotonic()
    status, payload = _post(server_url, "/v1/shell/wait", {"id": session_id, "seconds": 1})
    elapsed = time.monotonic() - start
    assert status == 200
    assert payload["data"]["status"] == "running"
    assert 0.8 <= elapsed <= 5.0  # wait actually blocked ~1s
    data = _wait_done(server_url, session_id)
    assert data["exit_code"] == 0 and "done" in data["output"]


def test_kill_terminates_process_group_including_descendants(server_url):
    marker = "rsi_kill_marker_29731"
    session_id = _exec(
        server_url,
        f"bash -c 'echo $$ > /tmp/{marker}.child; sleep 29731' & echo started; wait",
    )
    status, payload = _post(server_url, "/v1/shell/view", {"id": session_id})
    assert payload["data"]["status"] == "running"
    time.sleep(0.5)

    status, payload = _post(server_url, "/v1/shell/kill", {"id": session_id})
    assert status == 200
    data = payload["data"]
    assert data["status"] == "terminated"

    # The backgrounded child (same process group) must be gone too.
    time.sleep(0.5)
    leftovers = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode()
        except OSError:
            continue
        if "sleep 29731" in cmdline:
            leftovers.append(cmdline)
    assert not leftovers, f"descendant survived kill: {leftovers}"


def test_output_capture_is_bounded(server_url):
    session_id = _exec(server_url, "python3 -c \"print('x' * (3 * 1024 * 1024))\"")
    data = _wait_done(server_url, session_id, timeout=30)
    assert data["status"] == "completed"
    assert len(data["output"].encode("utf-8")) <= worker.MAX_CAPTURE_BYTES + 4096


def test_cleanup_removes_session(server_url):
    session_id = _exec(server_url, "echo cleanup-me")
    _wait_done(server_url, session_id)
    status, _ = _delete(server_url, f"/v1/shell/sessions/{session_id}")
    assert status == 200
    status, payload = _post(server_url, "/v1/shell/view", {"id": session_id})
    assert status == 404


# --- single execution slot -------------------------------------------------


def test_single_execution_slot_rejects_concurrent_exec(server_url):
    first = _exec(server_url, "sleep 3")
    status, payload = _post(server_url, "/v1/shell/exec", {"command": "echo second"})
    assert status == 429
    assert payload["error"] == "execution_slot_busy"
    status, _ = _delete(server_url, f"/v1/shell/sessions/{first}")
    assert status == 200
    # Slot is free again once cleanup of the first session completed.
    session_id = _exec(server_url, "echo slot-free")
    data = _wait_done(server_url, session_id)
    assert data["status"] == "completed"


# --- cleanup gate (mocked sweeper; no host-wide enumeration) ---------------


def _serve_with_config(config):
    server = worker.create_server("127.0.0.1", 0, config=config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    return server, thread, url


def test_session_not_done_until_cleanup_completes():
    gate = threading.Event()
    calls: list[str] = []

    def blocking_sweeper() -> bool:
        calls.append("sweep")
        gate.wait(10)
        return True

    server, thread, url = _serve_with_config(make_config(sweeper=blocking_sweeper))
    try:
        session_id = _exec(url, "echo fast-exit")
        # Process exits immediately, but the session must stay "running"
        # while the cleanup sweep is still in progress.
        deadline = time.monotonic() + 5
        while not calls and time.monotonic() < deadline:
            time.sleep(0.05)
        assert calls, "sweeper never ran"
        status, payload = _post(url, "/v1/shell/view", {"id": session_id})
        assert payload["data"]["status"] == "running"
        gate.set()
        data = _wait_done(url, session_id)
        assert data["status"] == "completed"
        assert data["cleanup_verified"] is True
    finally:
        gate.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_unproven_cleanup_poisons_worker_fail_closed():
    server, thread, url = _serve_with_config(make_config(sweeper=lambda: False))
    try:
        session_id = _exec(url, "echo doomed")
        data = _wait_done(url, session_id)
        assert data["cleanup_verified"] is False
        # healthz turns 503 so the container healthcheck fails closed.
        request = urllib.request.Request(url + "/healthz", method="GET")
        try:
            with _opener().open(request, timeout=5) as response:
                raise AssertionError(f"healthz should be 503, got {response.status}")
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
            assert json.loads(exc.read().decode("utf-8"))["error"] == "worker_cleanup_unproven"
        # New exec is refused; the slot is never released.
        status, payload = _post(url, "/v1/shell/exec", {"command": "echo nope"})
        assert status == 503
        assert payload["error"] == "worker_cleanup_unproven"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# --- container guard (mocked /proc, environ, euid) -------------------------


def _fake_proc_root(tmp_path: Path, *, cmdline: bytes = b"python3 /opt/rsi_worker/minimal_worker.py\x00",
                    cap_eff: str = "00000000000000e0") -> Path:
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "cmdline").write_bytes(cmdline)
    (tmp_path / "self").mkdir()
    (tmp_path / "self" / "status").write_text(f"Name:\tpython3\nCapEff:\t{cap_eff}\n", encoding="utf-8")
    return tmp_path


def _guard_environ() -> dict:
    return {worker.CONTAINER_MARKER_ENV: worker.EXPECTED_CONTAINER_MARKER}


def test_guard_accepts_verified_container(tmp_path):
    proc_root = _fake_proc_root(tmp_path)
    worker.verify_container_guard(
        CANDIDATE_UID, CANDIDATE_GID, proc_root=proc_root, environ=_guard_environ(), euid=0
    )


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"euid": 1000}, "supervisor_not_root"),
        ({"candidate_uid": 0}, "candidate_id_invalid"),
        ({"candidate_uid": 0 + 1, "environ": {}}, "container_marker_absent"),
        ({"cmdline": b"/sbin/init\x00"}, "pid_namespace_not_container"),
        ({"cap_eff": "0000000000000020"}, "capabilities_missing"),
    ],
)
def test_guard_fails_closed(tmp_path, kwargs, reason):
    candidate_uid = kwargs.pop("candidate_uid", CANDIDATE_UID)
    cmdline = kwargs.pop("cmdline", b"python3 /opt/rsi_worker/minimal_worker.py\x00")
    cap_eff = kwargs.pop("cap_eff", "00000000000000e0")
    proc_root = _fake_proc_root(tmp_path, cmdline=cmdline, cap_eff=cap_eff)
    params = {
        "proc_root": proc_root,
        "environ": _guard_environ(),
        "euid": 0,
    }
    params.update(kwargs)
    with pytest.raises(worker.GuardError) as excinfo:
        worker.verify_container_guard(candidate_uid, CANDIDATE_GID, **params)
    assert excinfo.value.reason == reason


def test_main_style_env_parsing_fails_closed(monkeypatch):
    monkeypatch.delenv("RSI_WORKER_CANDIDATE_UID", raising=False)
    with pytest.raises(worker.GuardError):
        worker._required_int_env("RSI_WORKER_CANDIDATE_UID")


# --- uid sweep (mocked proc tree and kill; runs nowhere near real /proc) ---


def _write_proc_entry(root: Path, pid: int, uid: int) -> None:
    entry = root / str(pid)
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "status").write_text(f"Name:\tproc{pid}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n", encoding="utf-8")


def test_sweep_kills_only_candidate_uid_and_verifies(tmp_path):
    _write_proc_entry(tmp_path, 100, CANDIDATE_UID)
    _write_proc_entry(tmp_path, 1, 0)          # supervisor/init: never touched
    _write_proc_entry(tmp_path, 200, 0)        # other uid: never touched
    _write_proc_entry(tmp_path, 300, CANDIDATE_UID)
    killed: list[int] = []

    def fake_kill(pid: int, sig: int) -> None:
        killed.append(pid)
        shutil.rmtree(tmp_path / str(pid))  # simulated death

    ok = worker.sweep_candidate_processes(
        CANDIDATE_UID, proc_root=tmp_path, kill_fn=fake_kill, sleep_fn=lambda s: None
    )
    assert ok is True
    assert sorted(killed) == [100, 300]


def test_sweep_never_targets_protected_pids_even_with_candidate_uid(tmp_path):
    own = 424242
    _write_proc_entry(tmp_path, 1, CANDIDATE_UID)   # adversarial: pid 1
    _write_proc_entry(tmp_path, own, CANDIDATE_UID)  # adversarial: supervisor
    _write_proc_entry(tmp_path, 55, CANDIDATE_UID)
    killed: list[int] = []

    def fake_kill(pid: int, sig: int) -> None:
        killed.append(pid)
        shutil.rmtree(tmp_path / str(pid))

    ok = worker.sweep_candidate_processes(
        CANDIDATE_UID,
        proc_root=tmp_path,
        kill_fn=fake_kill,
        sleep_fn=lambda s: None,
        protected_pids=[own],
    )
    assert ok is True
    assert killed == [55]
    assert (tmp_path / "1").exists() and (tmp_path / str(own)).exists()


def test_sweep_reports_failure_when_processes_persist(tmp_path):
    _write_proc_entry(tmp_path, 100, CANDIDATE_UID)
    killed: list[int] = []
    ok = worker.sweep_candidate_processes(
        CANDIDATE_UID,
        proc_root=tmp_path,
        kill_fn=lambda pid, sig: killed.append(pid),  # refuses to die
        sleep_fn=lambda s: None,
        max_rounds=3,
    )
    assert ok is False
    assert killed == [100, 100, 100]


# --- privilege drop + environment scrub (mocked os calls) ------------------


def test_privilege_dropper_order(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(worker.os, "setgroups", lambda groups: calls.append(("setgroups", groups)))
    monkeypatch.setattr(worker.os, "setgid", lambda gid: calls.append(("setgid", gid)))
    monkeypatch.setattr(worker.os, "setuid", lambda uid: calls.append(("setuid", uid)))
    monkeypatch.setattr(worker.os, "umask", lambda mask: calls.append(("umask", mask)))
    worker.make_privilege_dropper(CANDIDATE_UID, CANDIDATE_GID)()
    assert calls == [
        ("setgroups", []),
        ("setgid", CANDIDATE_GID),
        ("setuid", CANDIDATE_UID),
        ("umask", 0o077),
    ]


def test_candidate_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("RSI_WORKER_CONTROL_KEY_FILE", "/run/rsi_control/key")
    monkeypatch.setenv("SANDBOX_API_KEYS", "should-not-propagate")
    env = worker.candidate_environment()
    assert env["PATH"] == "/usr/local/bin:/usr/bin:/bin"
    for key in env:
        assert "CONTROL" not in key and "API" not in key and "KEY" not in key
    assert "SANDBOX_API_KEYS" not in env


# --- control key file loading: fail-closed paths (no setuid on host) -------


def test_control_key_missing_file_fails_closed(tmp_path):
    with pytest.raises(worker.GuardError) as excinfo:
        worker.load_control_key(tmp_path / "absent", candidate_uid=CANDIDATE_UID)
    assert excinfo.value.reason == "control_key_unavailable"


def test_control_key_mode_too_open_fails_closed(tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text("k" * 64, encoding="utf-8")
    key_file.chmod(0o644)
    with pytest.raises(worker.GuardError) as excinfo:
        worker.load_control_key(key_file, candidate_uid=CANDIDATE_UID)
    assert excinfo.value.reason == "control_key_mode_too_open"


# --- US-008: truthful DELETE acknowledgement + exec-class sandbox routing ---


def test_delete_acknowledges_verified_cleanup_explicitly(server_url):
    session_id = _exec(server_url, "echo ack-me")
    _wait_done(server_url, session_id)
    status, payload = _delete(server_url, f"/v1/shell/sessions/{session_id}")
    assert status == 200
    data = payload["data"]
    assert data["removed"] is True
    assert data["cleanup_verified"] is True
    assert data["verified"] is True


def test_delete_unproven_cleanup_returns_503_not_false_removed():
    # Sweeper can never prove cleanup -> session preserved + poisoned.
    config = make_config(sweeper=lambda: False)
    server, thread, url = _serve_with_config(config)
    try:
        session_id = _exec(url, "echo doomed")
        deadline = time.monotonic() + 15.0
        while True:
            _, payload = _post(url, "/v1/shell/view", {"id": session_id})
            if payload["data"]["status"] != "running":
                break
            assert time.monotonic() < deadline
        status, payload = _delete(url, f"/v1/shell/sessions/{session_id}")
        assert status == 503
        assert payload["error"] == "worker_cleanup_unproven"
        assert payload["data"]["removed"] is False
        assert payload["data"]["cleanup_verified"] is False
        # The unclean session is preserved (evidence + occupied slot), and
        # the worker fails closed.
        status, _ = _delete(url, f"/v1/shell/sessions/{session_id}")
        assert status == 503
        import urllib.request as _rq
        try:
            _opener().open(url + "/healthz", timeout=5)
            raise AssertionError("poisoned worker must not be healthy")
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _guard_config(tmp_path):
    scratch = tmp_path / "scratch"
    job_root = scratch / "jobs-candidate-v1" / "job-x"
    code_dir = job_root / "code"
    code_dir.mkdir(parents=True)
    config = make_config(guard_active=True, scratch_root=str(scratch))
    # Host tests cannot drop privileges; validation is what is under test.
    config.spawn_with_privilege_drop = False
    return config, scratch, job_root, code_dir


def test_candidate_exec_requires_contained_job_root(tmp_path):
    config, scratch, job_root, code_dir = _guard_config(tmp_path)
    server, thread, url = _serve_with_config(config)
    try:
        base = {"id": "", "command": "echo hi", "async_mode": True, "timeout": 30}
        # Missing job_root -> fail closed.
        status, payload = _post(url, "/v1/shell/exec", {**base, "exec_dir": str(code_dir)})
        assert (status, payload["error"]) == (400, "job_root_required")
        # job_root outside the scratch root -> denied (incl. '..' tricks).
        status, payload = _post(
            url, "/v1/shell/exec", {**base, "exec_dir": str(code_dir), "job_root": str(tmp_path)}
        )
        assert (status, payload["error"]) == (400, "job_root_outside_scratch")
        status, payload = _post(
            url,
            "/v1/shell/exec",
            {**base, "exec_dir": str(code_dir), "job_root": str(scratch / "jobs-candidate-v1" / ".." / "..")},
        )
        assert (status, payload["error"]) == (400, "job_root_outside_scratch")
        # The scratch root itself is not a job root.
        status, payload = _post(
            url, "/v1/shell/exec", {**base, "exec_dir": str(code_dir), "job_root": str(scratch)}
        )
        assert (status, payload["error"]) == (400, "job_root_outside_scratch")
        # Missing exec_dir for candidate class.
        status, payload = _post(url, "/v1/shell/exec", {**base, "job_root": str(job_root)})
        assert (status, payload["error"]) == (400, "exec_dir_required")
        # exec_dir outside the job root.
        status, payload = _post(
            url, "/v1/shell/exec", {**base, "exec_dir": str(scratch), "job_root": str(job_root)}
        )
        assert (status, payload["error"]) == (400, "exec_dir_outside_job_root")
        # Unknown exec_class.
        status, payload = _post(
            url,
            "/v1/shell/exec",
            {**base, "exec_dir": str(code_dir), "job_root": str(job_root), "exec_class": "root"},
        )
        assert (status, payload["error"]) == (400, "invalid_exec_class")
        # Fully contained candidate exec is accepted.
        status, payload = _post(
            url, "/v1/shell/exec", {**base, "exec_dir": str(code_dir), "job_root": str(job_root)}
        )
        assert status == 200, payload
        _wait_done(url, payload["data"]["session_id"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_staging_exec_class_runs_without_job_root(tmp_path):
    config, scratch, job_root, code_dir = _guard_config(tmp_path)
    server, thread, url = _serve_with_config(config)
    try:
        status, payload = _post(
            url,
            "/v1/shell/exec",
            {"id": "", "command": "echo staged", "exec_dir": None, "exec_class": "staging",
             "async_mode": True, "timeout": 30},
        )
        assert status == 200, payload
        data = _wait_done(url, payload["data"]["session_id"])
        assert data["status"] == "completed"
        assert "staged" in data["output"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _NullStream:
    def read(self, size=-1):
        return b""


class _FakePopen:
    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.pid = 999999
        self.stdout = _NullStream()
        self.stderr = _NullStream()

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0

    def kill(self):
        return None


def test_candidate_session_wraps_command_in_trampoline(monkeypatch):
    spawned: list = []

    def fake_popen(argv, **kwargs):
        proc = _FakePopen(list(argv), **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(worker.subprocess, "Popen", fake_popen)
    config = make_config(guard_active=True, sweeper=lambda: True, scratch_root="/scratch")
    config.spawn_with_privilege_drop = True
    registry = worker.SessionRegistry(config)
    session = worker.Session("s-cand", "echo hi", "/scratch/job/code", config, registry,
                             exec_class=worker.EXEC_CLASS_CANDIDATE, job_root="/scratch/job")
    session.start()
    assert session.poll_done(10.0)
    proc = spawned[0]
    assert proc.argv == [
        "python3",
        "-I",
        "-S",
        worker.TRAMPOLINE_PATH,
        "--job-root",
        "/scratch/job",
        "--",
        "/bin/bash",
        "-c",
        "echo hi",
    ]
    assert proc.kwargs["user"] == CANDIDATE_UID
    assert proc.kwargs["group"] == CANDIDATE_GID
    assert proc.kwargs["env"]["RSI_WORKER_SCRATCH_ROOT"] == "/scratch"
    assert session.cleanup_verified is True


def test_staging_session_is_not_wrapped(monkeypatch):
    spawned: list = []

    def fake_popen(argv, **kwargs):
        proc = _FakePopen(list(argv), **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(worker.subprocess, "Popen", fake_popen)
    config = make_config(guard_active=True, sweeper=lambda: True, scratch_root="/scratch")
    config.spawn_with_privilege_drop = True
    registry = worker.SessionRegistry(config)
    session = worker.Session("s-stage", "echo hi", None, config, registry,
                             exec_class=worker.EXEC_CLASS_STAGING)
    session.start()
    assert session.poll_done(10.0)
    assert spawned[0].argv == ["/bin/bash", "-c", "echo hi"]


def test_execution_confinement_guard_fails_closed(tmp_path):
    trampoline = REPO_ROOT / "deploy" / "rsi-trustworthy" / "worker" / "exec_trampoline.py"
    # Real trampoline present + ABI >= 3 available -> passes.
    worker.verify_execution_confinement_available(
        trampoline_path=trampoline, abi_fn=lambda: worker.MIN_LANDLOCK_ABI
    )
    with pytest.raises(worker.GuardError, match="exec_trampoline_absent"):
        worker.verify_execution_confinement_available(
            trampoline_path=tmp_path / "absent.py", abi_fn=lambda: 99
        )
    with pytest.raises(worker.GuardError, match="landlock_abi_too_old"):
        worker.verify_execution_confinement_available(
            trampoline_path=trampoline, abi_fn=lambda: worker.MIN_LANDLOCK_ABI - 1
        )
