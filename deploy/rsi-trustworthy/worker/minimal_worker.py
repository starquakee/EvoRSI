"""Minimal sandbox worker service for the rsi-trustworthy isolated stack.

Runs INSIDE the locked-down worker container (no published ports, internal
isolated network only, cap_drop ALL + minimal SETUID/SETGID/KILL for the
supervisor, no-new-privileges, default seccomp, read-only rootfs, read-only
public input, per-job scratch only, no docker socket). It implements the
subset of the AIO worker shell protocol that the task dispatcher uses:

  POST   /v1/shell/exec          {id, command, exec_dir, exec_class, job_root, async_mode, timeout}
  POST   /v1/shell/wait          {id, seconds}
  POST   /v1/shell/view          {id}
  POST   /v1/shell/kill          {id}
  DELETE /v1/shell/sessions/{id}
  GET    /healthz

Responses wrap payloads in {"data": ...} to match the dispatcher's
SandboxHTTPController. Status values follow the dispatcher contract:
"running" while executing (including while post-exit cleanup is in
progress), "completed" on exit, "terminated" when killed. DELETE returns
an explicit verified acknowledgement (200 cleanup_verified=true) only
when cleanup is proven, otherwise 503 cleanup_verified=false.

Security model (batch-2 repair):
  * CONTROL AUTHENTICATION: every /v1/* endpoint requires the
    X-RSI-Control-Key header. The key file is mounted root-unreadable
    (0600, host uid) into the container; the root supervisor reads it once
    at startup through a privilege-dropped helper child keyed to the file
    owner uid and keeps it only in memory. Candidate processes (dedicated
    non-root uid, no capabilities, scrubbed environment) can neither read
    the key file nor the supervisor's /proc/<pid>/environ, so candidate
    code can never drive the worker control API.
  * CANDIDATE ISOLATION: sessions run as a dedicated non-root candidate
    uid/gid with a scrubbed environment; no-new-privileges is enforced
    container-wide and inherited. Candidate sessions (the default
    exec_class) are routed through the trusted pre-exec trampoline
    (exec_trampoline.py): after the uid drop it applies Landlock
    (read/exec system roots, read-only public input, read-write ONLY
    the current job scratch root) plus a narrow seccomp filter denying
    the chmod/chown metadata-mutation family, then execs the command.
    Only dispatcher-generated, control-key-authenticated staging
    sessions (exec_class=staging) run without the trampoline; candidate
    code can never reach this API to forge one.
  * ONE EXECUTION SLOT: at most one session is active at a time; a second
    exec is rejected (429 execution_slot_busy).
  * PROVEN CLEANUP: after a session exits or is killed, ALL processes
    owned by the candidate uid (including detached/reparented descendants
    that escaped the process group) are SIGKILLed via a bounded /proc uid
    sweep and output pipes are drained. The slot is released and the
    session reported done ONLY after zero candidate processes remain.
    If cleanup cannot be proven, the worker poisons itself: /healthz turns
    503 and further exec calls are refused (fail closed).
  * FAIL-CLOSED GUARD: at startup the supervisor verifies it is really
    inside its dedicated container/PID namespace (PID 1 is this script,
    container marker env, root euid, required capabilities, sane candidate
    ids). Any violation exits nonzero. The uid sweep therefore can never
    run on a host or foreign namespace.

This service replaces the AIO worker's /opt/gem/run.sh (which required
seccomp=unconfined and started browser/VNC services).
"""

from __future__ import annotations

import ctypes
import hmac
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

LISTEN_HOST = os.getenv("RSI_WORKER_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("RSI_WORKER_PORT", "8080"))
MAX_CAPTURE_BYTES = int(os.getenv("RSI_WORKER_MAX_CAPTURE_BYTES", str(1024 * 1024)))
MAX_BODY_BYTES = int(os.getenv("RSI_WORKER_MAX_BODY_BYTES", str(4 * 1024 * 1024)))
MAX_RETAINED_SESSIONS = int(os.getenv("RSI_WORKER_MAX_RETAINED_SESSIONS", "32"))

EXPECTED_CONTAINER_MARKER = "rsi-trustworthy-worker"
CONTAINER_MARKER_ENV = "RSI_WORKER_CONTAINER"
CONTROL_KEY_ENV = "RSI_WORKER_CONTROL_KEY_FILE"
DEFAULT_CONTROL_KEY_PATH = "/run/rsi_control/key"
MIN_CONTROL_KEY_BYTES = 16

SCRATCH_ROOT_ENV = "RSI_WORKER_SCRATCH_ROOT"
DEFAULT_SCRATCH_ROOT = "/mnt/local_sandbox_workdir"
TRAMPOLINE_PATH = "/opt/rsi_worker/exec_trampoline.py"
MIN_LANDLOCK_ABI = 3
EXEC_CLASS_CANDIDATE = "candidate"
EXEC_CLASS_STAGING = "staging"

CAP_KILL = 5
CAP_SETGID = 6
CAP_SETUID = 7
REQUIRED_CAP_MASK = (1 << CAP_KILL) | (1 << CAP_SETGID) | (1 << CAP_SETUID)

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_TERMINATED = "terminated"

SWEEP_MAX_ROUNDS = 20
SWEEP_INTERVAL_SECONDS = 0.1
DRAIN_JOIN_TIMEOUT_SECONDS = 5.0


class GuardError(Exception):
    """Container/PID-namespace/uid assumption failed; refuse to start."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class PoisonedError(Exception):
    """Worker cleanup could not be proven; the worker is fail-closed."""


def _read_cap_eff(proc_root: Path) -> int:
    status_text = (proc_root / "self" / "status").read_text(encoding="utf-8", errors="replace")
    for line in status_text.splitlines():
        if line.startswith("CapEff:"):
            return int(line.split()[1], 16)
    raise GuardError("capabilities_missing")


def verify_container_guard(
    candidate_uid: int,
    candidate_gid: int,
    *,
    proc_root: Path = Path("/proc"),
    environ: Optional[Dict[str, str]] = None,
    euid: Optional[int] = None,
) -> None:
    """Positively verify the dedicated container/PID-namespace assumptions.

    Raises GuardError (fail closed) unless ALL of the following hold:
    the supervisor runs as root with CAP_SETUID/SETGID/KILL, PID 1 of this
    namespace is this worker script, the container marker env matches, and
    the candidate ids are non-zero and distinct from the supervisor.
    """
    env = os.environ if environ is None else environ
    if os.name != "posix":
        raise GuardError("non_posix_host")
    effective_uid = os.geteuid() if euid is None else euid
    if effective_uid != 0:
        raise GuardError("supervisor_not_root")
    if not (0 < candidate_uid < 65536) or not (0 < candidate_gid < 65536):
        raise GuardError("candidate_id_invalid")
    if candidate_uid == effective_uid:
        raise GuardError("candidate_id_invalid")
    if env.get(CONTAINER_MARKER_ENV) != EXPECTED_CONTAINER_MARKER:
        raise GuardError("container_marker_absent")
    try:
        init_cmdline = (proc_root / "1" / "cmdline").read_bytes()
    except OSError as exc:
        raise GuardError("pid_namespace_not_container") from exc
    if b"minimal_worker.py" not in init_cmdline.replace(b"\x00", b" "):
        raise GuardError("pid_namespace_not_container")
    if _read_cap_eff(proc_root) & REQUIRED_CAP_MASK != REQUIRED_CAP_MASK:
        raise GuardError("capabilities_missing")


def query_landlock_abi() -> int:
    """Read-only Landlock ABI version query (no policy created or changed)."""
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(ctypes.c_long(444), None, ctypes.c_size_t(0), ctypes.c_uint(1 << 0))
    return int(result) if result and result > 0 else 0


def verify_execution_confinement_available(
    *,
    trampoline_path: Path = Path(TRAMPOLINE_PATH),
    abi_fn: Optional[Callable[[], int]] = None,
) -> None:
    """Fail closed unless the candidate pre-exec sandbox is usable.

    The cross-job scratch boundary (Landlock ABI >= 3 + seccomp metadata
    filter, applied by the trusted trampoline after the uid drop) is part
    of the worker security contract; without it the worker refuses to
    start rather than executing candidates unconfined.
    """
    if not trampoline_path.is_file():
        raise GuardError("exec_trampoline_absent")
    abi = (abi_fn or query_landlock_abi)()
    if abi < MIN_LANDLOCK_ABI:
        raise GuardError("landlock_abi_too_old")


def load_control_key(path: Path, *, candidate_uid: int) -> bytes:
    """Read the worker control key through a privilege-dropped helper child.

    The key file is owned by a non-root host uid with mode 0600, which the
    cap-dropped container root cannot open directly (no CAP_DAC_OVERRIDE).
    The supervisor (CAP_SETUID) forks a helper that drops to the file
    owner's uid/gid, reads the bounded file and pipes the bytes back. The
    key then exists only in supervisor memory; candidate processes (a
    different uid, no capabilities) can open neither the file nor the
    supervisor's /proc environ. Fails closed on any anomaly.
    """
    try:
        info = os.stat(path)
    except OSError as exc:
        raise GuardError("control_key_unavailable") from exc
    if not stat.S_ISREG(info.st_mode):
        raise GuardError("control_key_not_regular")
    if info.st_uid == 0 or info.st_uid == candidate_uid:
        raise GuardError("control_key_owner_invalid")
    if info.st_mode & 0o077:
        raise GuardError("control_key_mode_too_open")
    if info.st_size <= 0 or info.st_size > 4096:
        raise GuardError("control_key_size_invalid")

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # helper child: drop to the key file owner and read it
        try:
            os.close(read_fd)
            os.setgroups([])
            os.setgid(info.st_gid)
            os.setuid(info.st_uid)
            with open(path, "rb") as handle:
                data = handle.read(4096)
            os.write(write_fd, data)
        except BaseException:
            os._exit(1)
        os._exit(0)
    os.close(write_fd)
    chunks: List[bytes] = []
    while True:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)
    if not (os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0):
        raise GuardError("control_key_unavailable")
    key = b"".join(chunks).strip()
    if len(key) < MIN_CONTROL_KEY_BYTES:
        raise GuardError("control_key_too_short")
    return key


def candidate_environment() -> Dict[str, str]:
    """Scrubbed environment for candidate processes (no control key, ever)."""
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def make_privilege_dropper(candidate_uid: int, candidate_gid: int) -> Callable[[], None]:
    """preexec that drops the child to the unprivileged candidate ids."""

    def drop() -> None:
        os.setgroups([])
        os.setgid(candidate_gid)
        os.setuid(candidate_uid)
        os.umask(0o077)

    return drop


def iter_uid_processes(uid: int, *, proc_root: Path = Path("/proc")) -> List[int]:
    """PIDs whose REAL uid matches ``uid`` inside this PID namespace."""
    pids: List[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status_text = (entry / "status").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in status_text.splitlines():
            if line.startswith("Uid:"):
                fields = line.split()
                if len(fields) >= 2 and int(fields[1]) == uid:
                    pids.append(int(entry.name))
                break
    return pids


def reap_candidate_children(
    candidate_uid: int,
    *,
    proc_root: Path = Path("/proc"),
    waitpid_fn: Callable[[int, int], Tuple[int, int]] = os.waitpid,
) -> None:
    """Reap only adopted candidate-uid children after the session leader wait.

    Never use waitpid(-1): it could consume a trusted helper's exit status.
    The sole session leader has already been reaped by its Popen object.
    """
    for pid in iter_uid_processes(candidate_uid, proc_root=proc_root):
        if pid in (1, os.getpid()):
            continue
        try:
            waitpid_fn(pid, os.WNOHANG)
        except (ChildProcessError, ProcessLookupError):
            pass  # still owned by a live candidate parent, or already gone


def sweep_candidate_processes(
    candidate_uid: int,
    *,
    proc_root: Path = Path("/proc"),
    kill_fn: Callable[[int, int], None] = os.kill,
    sleep_fn: Callable[[float], None] = time.sleep,
    reap_fn: Optional[Callable[[], None]] = None,
    max_rounds: int = SWEEP_MAX_ROUNDS,
    interval: float = SWEEP_INTERVAL_SECONDS,
    protected_pids: Optional[List[int]] = None,
) -> bool:
    """SIGKILL every candidate-uid process; True only when none remain.

    Only ever signals processes whose real uid equals the candidate uid;
    pid 1, the supervisor itself and every non-candidate process are
    excluded twice (uid filter + explicit protected list). Intended to run
    solely inside the guard-verified worker container. ``reap_fn`` (wired
    only in production) collects detached descendants that died as
    zombies under PID 1 between rounds; without it a just-killed detached
    child would linger in /proc and falsely defeat verification.
    """
    protected = {1, os.getpid()}
    if protected_pids:
        protected.update(protected_pids)

    def remaining() -> List[int]:
        if reap_fn is not None:
            reap_fn()
        return [p for p in iter_uid_processes(candidate_uid, proc_root=proc_root) if p not in protected]

    for _ in range(max_rounds):
        pids = remaining()
        if not pids:
            return True
        for pid in pids:
            try:
                kill_fn(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        sleep_fn(interval)
    return not remaining()


@dataclass
class WorkerConfig:
    control_key: bytes
    candidate_uid: int = 0
    candidate_gid: int = 0
    guard_active: bool = False
    max_capture_bytes: int = MAX_CAPTURE_BYTES
    max_body_bytes: int = MAX_BODY_BYTES
    max_retained_sessions: int = MAX_RETAINED_SESSIONS
    # None -> real uid sweep when guard_active else a proven no-op.
    sweeper: Optional[Callable[[], bool]] = None
    spawn_with_privilege_drop: bool = field(default=False)
    scratch_root: str = DEFAULT_SCRATCH_ROOT
    trampoline_path: str = TRAMPOLINE_PATH

    def __post_init__(self) -> None:
        if not self.control_key:
            raise ValueError("control_key_required")
        self.spawn_with_privilege_drop = self.guard_active


class BoundedTail:
    """Keeps only the last ``limit`` bytes written to it."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1024, limit)
        self._buf = bytearray()
        self._lock = threading.Lock()

    def append(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            self._buf.extend(data)
            overflow = len(self._buf) - self.limit
            if overflow > 0:
                del self._buf[:overflow]

    def text(self) -> str:
        with self._lock:
            return bytes(self._buf).decode("utf-8", errors="replace")


class Session:
    def __init__(
        self,
        session_id: str,
        command: str,
        exec_dir: Optional[str],
        config: WorkerConfig,
        registry: "SessionRegistry",
        exec_class: str = EXEC_CLASS_CANDIDATE,
        job_root: Optional[str] = None,
    ) -> None:
        self.session_id = session_id
        self.command = command
        self.exec_dir = exec_dir
        self.exec_class = exec_class
        self.job_root = job_root
        self.config = config
        self.registry = registry
        self.created_at = time.time()
        self.stdout_tail = BoundedTail(config.max_capture_bytes)
        self.stderr_tail = BoundedTail(config.max_capture_bytes)
        self.killed = False
        self.cleaned = False
        self.cleanup_verified: Optional[bool] = None
        self._done = threading.Event()
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._start_lock = threading.Lock()
        self._drain_threads: List[threading.Thread] = []

    def start(self) -> None:
        with self._start_lock:
            if self._process is not None:
                return
            argv = ["/bin/bash", "-c", self.command]
            popen_kwargs: Dict[str, Any] = {}
            if self.config.spawn_with_privilege_drop:
                env = candidate_environment()
                env["RSI_WORKER_SCRATCH_ROOT"] = self.config.scratch_root
                popen_kwargs["env"] = env
                # Use subprocess's C-level identity setup; Python preexec_fn
                # is unsafe in this multi-threaded supervisor.
                popen_kwargs.update(
                    user=self.config.candidate_uid,
                    group=self.config.candidate_gid,
                    extra_groups=[],
                    umask=0o077,
                )
                if self.exec_class == EXEC_CLASS_CANDIDATE:
                    # Route candidate commands through the trusted pre-exec
                    # trampoline (Landlock + seccomp metadata filter,
                    # applied AFTER the C-level uid drop). The job_root was
                    # already proven contained under the scratch root by the
                    # exec handler; the trampoline re-proves it fail-closed.
                    # -I -S: isolated interpreter, no site/user packages.
                    argv = [
                        "python3",
                        "-I",
                        "-S",
                        self.config.trampoline_path,
                        "--job-root",
                        self.job_root or "",
                        "--",
                        *argv,
                    ]
            self._process = subprocess.Popen(
                argv,
                cwd=self.exec_dir or None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                **popen_kwargs,
            )
        for stream, tail in (
            (self._process.stdout, self.stdout_tail),
            (self._process.stderr, self.stderr_tail),
        ):
            thread = threading.Thread(
                target=self._drain,
                args=(stream, tail),
                daemon=True,
                name=f"drain-{self.session_id}",
            )
            thread.start()
            self._drain_threads.append(thread)
        threading.Thread(
            target=self._reap,
            daemon=True,
            name=f"reap-{self.session_id}",
        ).start()

    @staticmethod
    def _drain(stream: Any, tail: BoundedTail) -> None:
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                tail.append(chunk)
        except (OSError, ValueError):
            pass

    def _sweeper(self) -> Callable[[], bool]:
        if self.config.sweeper is not None:
            return self.config.sweeper
        if self.config.guard_active:
            return lambda: sweep_candidate_processes(
                self.config.candidate_uid,
                reap_fn=lambda: reap_candidate_children(self.config.candidate_uid),
            )
        return lambda: True

    def _reap(self) -> None:
        """Reap, then prove cleanup before declaring the session done.

        The uid sweep runs BEFORE joining the drain threads: detached
        descendants still holding the stdout/stderr pipes are SIGKILLed,
        which is what lets the pipes reach EOF and the drains finish.
        """
        assert self._process is not None
        self._process.wait()
        sweep_ok = False
        try:
            sweep_ok = bool(self._sweeper()())
        except Exception:
            sweep_ok = False
        drains_ok = True
        for thread in self._drain_threads:
            thread.join(timeout=DRAIN_JOIN_TIMEOUT_SECONDS)
            if thread.is_alive():
                drains_ok = False
        self.cleanup_verified = sweep_ok and drains_ok
        self.cleaned = True
        if not self.cleanup_verified:
            self.registry.poison("cleanup_unproven")
        self._done.set()

    def poll_done(self, timeout: float = 0.0) -> bool:
        return self._done.wait(timeout)

    def kill(self) -> None:
        self.killed = True
        process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            with _suppress_all():
                process.kill()

    def snapshot(self) -> Dict[str, Any]:
        process = self._process
        returncode = process.poll() if process is not None else None
        # A session stays "running" until post-exit cleanup is proven, so
        # the dispatcher never treats an unproven-cleanup session as done.
        if returncode is None or not self.cleaned:
            status = STATUS_RUNNING
        elif not self.cleanup_verified:
            status = "error"
        elif self.killed:
            status = STATUS_TERMINATED
        else:
            status = STATUS_COMPLETED
        payload: Dict[str, Any] = {
            "session_id": self.session_id,
            "status": status,
            "exit_code": returncode,
            "output": self.stdout_tail.text(),
            "stderr": self.stderr_tail.text(),
            "created_at": self.created_at,
        }
        if self.cleaned:
            payload["cleanup_verified"] = bool(self.cleanup_verified)
            if not self.cleanup_verified:
                payload["exit_code"] = 125
                payload["error"] = "cleanup_unproven"
        return payload


class _suppress_all:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: Any) -> bool:
        return True


class SessionRegistry:
    """Sessions plus the single execution slot and poison state."""

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self.poisoned = False
        self.poison_reason: Optional[str] = None

    def poison(self, reason: str) -> None:
        with self._lock:
            self.poisoned = True
            self.poison_reason = reason

    def create(
        self,
        command: str,
        exec_dir: Optional[str],
        exec_class: str = EXEC_CLASS_CANDIDATE,
        job_root: Optional[str] = None,
    ) -> Session:
        with self._lock:
            if self.poisoned:
                raise PoisonedError(self.poison_reason or "cleanup_unproven")
            active = [s for s in self._sessions.values() if not s.poll_done(0.0)]
            if active:
                raise RuntimeError("execution_slot_busy")
            self._evict_retained_locked()
            session = Session(uuid.uuid4().hex, command, exec_dir, self.config, self, exec_class, job_root)
            self._sessions[session.session_id] = session
        try:
            session.start()
        except Exception:
            with self._lock:
                self._sessions.pop(session.session_id, None)
            raise
        return session

    def _evict_retained_locked(self) -> None:
        done = [s for s in self._sessions.values() if s.poll_done(0.0)]
        done.sort(key=lambda s: s.created_at)
        while len(self._sessions) >= self.config.max_retained_sessions and done:
            self._sessions.pop(done.pop(0).session_id, None)

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str) -> Optional[Session]:
        session = self.get(session_id)
        if session is None:
            return None
        session.kill()
        # Block until cleanup is proven so the slot is free when the
        # dispatcher's DELETE returns.
        if not session.poll_done(SWEEP_MAX_ROUNDS * SWEEP_INTERVAL_SECONDS + DRAIN_JOIN_TIMEOUT_SECONDS + 5.0):
            self.poison("cleanup_timeout")
            return session  # preserve the occupied slot and its evidence
        if session.cleanup_verified is not True:
            self.poison("cleanup_unproven")
            return session
        with self._lock:
            self._sessions.pop(session_id, None)
        return session


class WorkerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: Tuple[str, int], config: WorkerConfig) -> None:
        self.worker_config = config
        self.registry = SessionRegistry(config)
        super().__init__(address, WorkerHandler)


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler, max_body: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    length_header = handler.headers.get("Content-Length")
    try:
        length = int(length_header) if length_header else 0
    except ValueError:
        return None, "invalid_content_length"
    if length <= 0:
        return {}, None
    if length > max_body:
        return None, "body_too_large"
    raw = handler.rfile.read(length)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "invalid_json"
    if not isinstance(data, dict):
        return None, "invalid_json"
    return data, None


class WorkerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "rsi-minimal-worker/2.0"
    server: WorkerHTTPServer  # narrow the type for mypy

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[worker-http] %s\n" % (fmt % args))

    def _authorized(self) -> bool:
        presented = self.headers.get("X-RSI-Control-Key")
        expected = self.server.worker_config.control_key
        if not presented or not expected:
            return False
        return hmac.compare_digest(presented.encode("utf-8"), expected)

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        _json_response(self, 401, {"error": "worker_control_unauthorized"})
        return False

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path.rstrip("/") == "/healthz":
            registry = self.server.registry
            if registry.poisoned:
                _json_response(self, 503, {"error": "worker_cleanup_unproven"})
                return
            _json_response(self, 200, {"data": {"status": "ok"}})
            return
        _json_response(self, 404, {"error": "not_found"})

    def do_DELETE(self) -> None:  # noqa: N802
        prefix = "/v1/shell/sessions/"
        if not self.path.startswith(prefix):
            _json_response(self, 404, {"error": "not_found"})
            return
        if not self._require_auth():
            return
        session_id = self.path[len(prefix):].strip("/")
        session = self.server.registry.remove(session_id)
        if session is None:
            _json_response(self, 404, {"error": "session_not_found"})
            return
        # Truthful acknowledgement: registry.remove preserves/poisons an
        # unclean session, so success is reported ONLY with an explicit
        # verified cleanup proof; anything else is a 503 fail-closed state.
        if session.cleanup_verified is True:
            _json_response(
                self,
                200,
                {"data": {"session_id": session_id, "removed": True, "cleanup_verified": True, "verified": True}},
            )
            return
        _json_response(
            self,
            503,
            {
                "error": "worker_cleanup_unproven",
                "data": {"session_id": session_id, "removed": False, "cleanup_verified": False},
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path not in {"/v1/shell/exec", "/v1/shell/wait", "/v1/shell/view", "/v1/shell/kill"}:
            _json_response(self, 404, {"error": "not_found"})
            return
        if not self._require_auth():
            return
        if path == "/v1/shell/exec":
            self._handle_exec()
        elif path == "/v1/shell/wait":
            self._handle_wait()
        elif path == "/v1/shell/view":
            self._handle_view()
        else:
            self._handle_kill()

    def _handle_exec(self) -> None:
        registry = self.server.registry
        if registry.poisoned:
            _json_response(self, 503, {"error": "worker_cleanup_unproven"})
            return
        body, error = _read_json_body(self, self.server.worker_config.max_body_bytes)
        if error:
            _json_response(self, 400, {"error": error})
            return
        assert body is not None
        command = body.get("command")
        if not isinstance(command, str) or not command.strip():
            _json_response(self, 400, {"error": "command_required"})
            return
        exec_dir = body.get("exec_dir")
        if exec_dir is not None and not isinstance(exec_dir, str):
            _json_response(self, 400, {"error": "invalid_exec_dir"})
            return
        if exec_dir and not os.path.isdir(exec_dir):
            _json_response(self, 400, {"error": "exec_dir_not_found"})
            return
        exec_class = body.get("exec_class", EXEC_CLASS_CANDIDATE)
        if exec_class not in (EXEC_CLASS_CANDIDATE, EXEC_CLASS_STAGING):
            _json_response(self, 400, {"error": "invalid_exec_class"})
            return
        job_root = body.get("job_root")
        config = self.server.worker_config
        if config.guard_active and exec_class == EXEC_CLASS_CANDIDATE:
            # Candidate sessions ALWAYS execute through the trusted
            # trampoline, which needs a per-job write root positively
            # contained under the scratch root. Trusted staging sessions
            # (dispatcher-generated, control-key authenticated) are the
            # only unrestricted class; candidate code can never reach this
            # API to forge one.
            if not isinstance(job_root, str) or not job_root:
                _json_response(self, 400, {"error": "job_root_required"})
                return
            scratch_real = Path(os.path.realpath(config.scratch_root))
            job_real = Path(os.path.realpath(job_root))
            if job_real == scratch_real or scratch_real not in job_real.parents:
                _json_response(self, 400, {"error": "job_root_outside_scratch"})
                return
            if not job_real.is_dir():
                _json_response(self, 400, {"error": "job_root_not_found"})
                return
            if not exec_dir:
                _json_response(self, 400, {"error": "exec_dir_required"})
                return
            exec_real = Path(os.path.realpath(exec_dir))
            if exec_real != job_real and job_real not in exec_real.parents:
                _json_response(self, 400, {"error": "exec_dir_outside_job_root"})
                return
            job_root = str(job_real)
        try:
            session = registry.create(command, exec_dir, exec_class=exec_class, job_root=job_root)
        except PoisonedError:
            _json_response(self, 503, {"error": "worker_cleanup_unproven"})
            return
        except RuntimeError as exc:
            _json_response(self, 429, {"error": str(exc)})
            return
        except Exception as exc:  # spawn failure
            _json_response(self, 500, {"error": f"spawn_failed: {exc}"})
            return
        payload = session.snapshot()
        _json_response(self, 200, {"data": payload})

    def _session_or_404(self, body: Optional[Dict[str, Any]]) -> Optional[Session]:
        if body is None:
            return None
        session_id = body.get("id")
        if not isinstance(session_id, str) or not session_id:
            _json_response(self, 400, {"error": "id_required"})
            return None
        session = self.server.registry.get(session_id)
        if session is None:
            _json_response(self, 404, {"error": "session_not_found"})
            return None
        return session

    def _handle_wait(self) -> None:
        body, error = _read_json_body(self, self.server.worker_config.max_body_bytes)
        if error:
            _json_response(self, 400, {"error": error})
            return
        session = self._session_or_404(body)
        if session is None:
            return
        assert body is not None
        try:
            seconds = float(body.get("seconds", 0))
        except (TypeError, ValueError):
            seconds = 0.0
        seconds = max(0.0, min(seconds, 60.0))
        session.poll_done(seconds)
        _json_response(self, 200, {"data": session.snapshot()})

    def _handle_view(self) -> None:
        body, error = _read_json_body(self, self.server.worker_config.max_body_bytes)
        if error:
            _json_response(self, 400, {"error": error})
            return
        session = self._session_or_404(body)
        if session is None:
            return
        _json_response(self, 200, {"data": session.snapshot()})

    def _handle_kill(self) -> None:
        body, error = _read_json_body(self, self.server.worker_config.max_body_bytes)
        if error:
            _json_response(self, 400, {"error": error})
            return
        session = self._session_or_404(body)
        if session is None:
            return
        session.kill()
        session.poll_done(SWEEP_MAX_ROUNDS * SWEEP_INTERVAL_SECONDS + DRAIN_JOIN_TIMEOUT_SECONDS + 5.0)
        _json_response(self, 200, {"data": session.snapshot()})


def create_server(host: str = LISTEN_HOST, port: int = LISTEN_PORT, *, config: WorkerConfig) -> WorkerHTTPServer:
    return WorkerHTTPServer((host, port), config)


def _required_int_env(name: str) -> int:
    raw = os.getenv(name)
    if raw is None:
        raise GuardError(f"{name.lower()}_absent")
    try:
        return int(raw)
    except ValueError as exc:
        raise GuardError(f"{name.lower()}_invalid") from exc


def main() -> None:
    try:
        candidate_uid = _required_int_env("RSI_WORKER_CANDIDATE_UID")
        candidate_gid = _required_int_env("RSI_WORKER_CANDIDATE_GID")
        verify_container_guard(candidate_uid, candidate_gid)
        verify_execution_confinement_available()
        key_path = Path(os.getenv(CONTROL_KEY_ENV, DEFAULT_CONTROL_KEY_PATH))
        control_key = load_control_key(key_path, candidate_uid=candidate_uid)
    except GuardError as exc:
        print(f"[minimal-worker] FATAL guard refused startup: {exc.reason}", flush=True)
        sys.exit(1)
    config = WorkerConfig(
        control_key=control_key,
        candidate_uid=candidate_uid,
        candidate_gid=candidate_gid,
        guard_active=True,
        scratch_root=os.getenv(SCRATCH_ROOT_ENV, DEFAULT_SCRATCH_ROOT),
    )
    server = create_server(config=config)
    print(f"[minimal-worker] listening on {LISTEN_HOST}:{LISTEN_PORT} (guard verified)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
