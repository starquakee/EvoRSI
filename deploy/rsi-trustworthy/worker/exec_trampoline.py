"""Trusted pre-exec sandbox trampoline for candidate sessions (US-008).

Runs as the unprivileged candidate uid INSIDE the guard-verified worker
container, between the supervisor's C-level uid drop and the candidate
command:

  supervisor (root) -> Popen(user=candidate) -> exec_trampoline.py
      -> Landlock filesystem confinement + narrow seccomp metadata filter
      -> execve(candidate command)

Confinement applied to the candidate process tree (inherited by every
descendant, cannot be escaped without new privileges, and
no-new-privileges is enforced container-wide):

  * Landlock (ABI >= 3, REQUIRED): read/execute only on system library
    roots, read-only on the public data/input mounts, read-write ONLY on
    the current job's scratch root (plus the GPU/null
    device nodes CUDA needs). Other jobs' scratch trees, the evaluator
    tree (never mounted anyway) and any future writable share are
    unreachable for open/read/write/rename/unlink/symlink/truncate.
  * Seccomp notify-free filter (EPERM) on the metadata-mutation syscall
    families Landlock ABI 3 does not mediate: chmod/fchmod/fchmodat/
    fchmodat2, chown/fchown/lchown/fchownat, utime/utimes/futimesat/
    utimensat and setxattr/lsetxattr/fsetxattr/removexattr/lremovexattr/
    fremovexattr. This closes the proven cross-job chmod attack
    (.runtime/cross-job-boundary-review.json) and its mtime/xattr
    variants. The dispatcher's marker protocol therefore creates markers
    with shell redirection (open/write), never `touch`. x32-ABI syscall
    numbers (nr | __X32_SYSCALL_BIT) fail closed with KILL.
  * Inherited descriptors beyond stdin/stdout/stderr are closed before
    exec (already-open descriptors escape later path restriction).
  * HOME and TMPDIR are redirected to a job-local directory; the shared
    /tmp tmpfs and /dev/shm are intentionally NOT granted, so there is no
    common writable escape hatch between jobs.

FAIL CLOSED: if the architecture, kernel Landlock ABI, scratch-root
containment or any filter setup is unavailable, the trampoline exits
nonzero BEFORE exec and the candidate command never runs unconfined.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

EXIT_FAILED = 87

# --- Landlock (x86_64 syscall numbers) -----------------------------------
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
LANDLOCK_RULE_PATH_BENEATH = 1

ACCESS_FS_EXECUTE = 1 << 0
ACCESS_FS_WRITE_FILE = 1 << 1
ACCESS_FS_READ_FILE = 1 << 2
ACCESS_FS_READ_DIR = 1 << 3
ACCESS_FS_REMOVE_DIR = 1 << 4
ACCESS_FS_REMOVE_FILE = 1 << 5
ACCESS_FS_MAKE_CHAR = 1 << 6
ACCESS_FS_MAKE_DIR = 1 << 7
ACCESS_FS_MAKE_REG = 1 << 8
ACCESS_FS_MAKE_SOCK = 1 << 9
ACCESS_FS_MAKE_FIFO = 1 << 10
ACCESS_FS_MAKE_BLOCK = 1 << 11
ACCESS_FS_MAKE_SYM = 1 << 12
ACCESS_FS_REFER = 1 << 13  # ABI 2
ACCESS_FS_TRUNCATE = 1 << 14  # ABI 3

ALL_FS_RIGHTS = (
    ACCESS_FS_EXECUTE
    | ACCESS_FS_WRITE_FILE
    | ACCESS_FS_READ_FILE
    | ACCESS_FS_READ_DIR
    | ACCESS_FS_REMOVE_DIR
    | ACCESS_FS_REMOVE_FILE
    | ACCESS_FS_MAKE_CHAR
    | ACCESS_FS_MAKE_DIR
    | ACCESS_FS_MAKE_REG
    | ACCESS_FS_MAKE_SOCK
    | ACCESS_FS_MAKE_FIFO
    | ACCESS_FS_MAKE_BLOCK
    | ACCESS_FS_MAKE_SYM
    | ACCESS_FS_REFER
    | ACCESS_FS_TRUNCATE
)
READ_EXEC = ACCESS_FS_EXECUTE | ACCESS_FS_READ_FILE | ACCESS_FS_READ_DIR
READ_ONLY = ACCESS_FS_READ_FILE | ACCESS_FS_READ_DIR
DEVICE_RW = ACCESS_FS_READ_FILE | ACCESS_FS_WRITE_FILE
MIN_LANDLOCK_ABI = 3

SYSTEM_READ_EXEC_ROOTS = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/opt", "/proc", "/sys")
PUBLIC_READ_ROOTS = ("/mnt/rsi_data", "/mnt/rsi_storage")
DEVICE_RW_PATHS = (
    "/dev/null",
    "/dev/zero",
    "/dev/full",
    "/dev/random",
    "/dev/urandom",
    "/dev/ptmx",
    # WSL2 paravirtual GPU device (this engine: Docker Desktop WSL2 backend,
    # no /dev/nvidia* nodes); native Linux uses the nvidia globs below.
    "/dev/dxg",
)
DEVICE_RW_GLOBS = ("/dev/nvidia*", "/dev/nvidia-caps/*")

# --- Seccomp (BPF) --------------------------------------------------------
AUDIT_ARCH_X86_64 = 0xC000003E
X32_SYSCALL_BIT = 0x40000000
BLOCKED_METADATA_SYSCALLS = (
    ("chmod", 90),
    ("fchmod", 91),
    ("fchmodat", 268),
    ("fchmodat2", 452),
    ("chown", 92),
    ("fchown", 93),
    ("lchown", 94),
    ("fchownat", 260),
    ("utime", 132),
    ("utimes", 235),
    ("futimesat", 261),
    ("utimensat", 280),
    ("setxattr", 188),
    ("lsetxattr", 189),
    ("fsetxattr", 190),
    ("removexattr", 197),
    ("lremovexattr", 198),
    ("fremovexattr", 199),
)
PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
EPERM = 1

BPF_LD = 0x00
BPF_W = 0x00
BPF_ABS = 0x20
BPF_JMP = 0x05
BPF_JEQ = 0x10
BPF_JSET = 0x40
BPF_K = 0x00
BPF_RET = 0x06


class TrampolineError(Exception):
    """Fail-closed refusal; the candidate command must not run."""


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8), ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def landlock_abi(libc: ctypes.CDLL | None = None) -> int:
    """Read-only Landlock ABI query (no policy created or changed)."""
    lib = libc or _libc()
    result = lib.syscall(ctypes.c_long(SYS_LANDLOCK_CREATE_RULESET), None, ctypes.c_size_t(0), ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION))
    if result < 0:
        return 0
    return int(result)


def build_access_plan(scratch_root: Path, job_root: Path) -> List[Tuple[str, int]]:
    """Ordered (path, allowed_rights) rules; missing paths are skipped.

    Deliberately NO write grant on /tmp, /dev/shm or the scratch root
    itself: the only writable tree is the current job root.
    """
    plan: List[Tuple[str, int]] = []
    for root in SYSTEM_READ_EXEC_ROOTS:
        if os.path.exists(root):
            plan.append((root, READ_EXEC))
    for root in PUBLIC_READ_ROOTS:
        if os.path.isdir(root):
            plan.append((root, READ_ONLY))
    devices = list(DEVICE_RW_PATHS)
    for pattern in DEVICE_RW_GLOBS:
        devices.extend(sorted(glob.glob(pattern)))
    for path in devices:
        if os.path.exists(path):
            plan.append((path, DEVICE_RW))
    plan.append((str(job_root), ALL_FS_RIGHTS))
    return plan


def _apply_landlock(plan: Sequence[Tuple[str, int]]) -> None:
    lib = _libc()
    attr = _RulesetAttr(ALL_FS_RIGHTS)
    ruleset_fd = lib.syscall(
        ctypes.c_long(SYS_LANDLOCK_CREATE_RULESET),
        ctypes.byref(attr),
        ctypes.c_size_t(ctypes.sizeof(attr)),
        ctypes.c_uint(0),
    )
    if ruleset_fd < 0:
        err = ctypes.get_errno()
        raise TrampolineError(f"landlock_create_ruleset_failed: errno={err}")
    try:
        for path, rights in plan:
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = _PathBeneathAttr(rights, fd)
                rc = lib.syscall(
                    ctypes.c_long(SYS_LANDLOCK_ADD_RULE),
                    ctypes.c_int(ruleset_fd),
                    ctypes.c_uint(LANDLOCK_RULE_PATH_BENEATH),
                    ctypes.byref(rule),
                    ctypes.c_uint(0),
                )
            finally:
                os.close(fd)
            if rc != 0:
                err = ctypes.get_errno()
                raise TrampolineError(f"landlock_add_rule_failed: path={path} errno={err}")
        if lib.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise TrampolineError("no_new_privs_failed")
        rc = lib.syscall(ctypes.c_long(SYS_LANDLOCK_RESTRICT_SELF), ctypes.c_int(ruleset_fd), ctypes.c_uint(0))
        if rc != 0:
            err = ctypes.get_errno()
            raise TrampolineError(f"landlock_restrict_self_failed: errno={err}")
    finally:
        os.close(ruleset_fd)


def build_seccomp_program() -> List[Tuple[int, int, int, int]]:
    """(code, jt, jf, k) BPF: EPERM on metadata-mutation syscalls.

    x32-ABI numbers (nr | __X32_SYSCALL_BIT) fail closed with KILL so the
    compat entry points cannot bypass the per-number checks. Ordinary
    syscalls (read/write/open/utime-free file creation, landlock setup)
    stay ALLOWED; the dispatcher's marker protocol creates markers with
    shell redirection instead of `touch`.
    """
    blocked = [nr for _, nr in BLOCKED_METADATA_SYSCALLS]
    count = len(blocked)
    # Layout: 0 ld arch, 1 jeq arch, 2 ld nr, 3 jset x32, 4..4+count-1 jeq,
    # then ret ALLOW, ret EPERM, ret KILL.
    idx_allow = 4 + count
    idx_eperm = idx_allow + 1
    idx_kill = idx_eperm + 1
    program: List[Tuple[int, int, int, int]] = []
    program.append((BPF_LD | BPF_W | BPF_ABS, 0, 0, 4))  # seccomp_data.arch
    program.append((BPF_JMP | BPF_JEQ | BPF_K, 0, idx_kill - 2, AUDIT_ARCH_X86_64))
    program.append((BPF_LD | BPF_W | BPF_ABS, 0, 0, 0))  # seccomp_data.nr
    program.append((BPF_JMP | BPF_JSET | BPF_K, idx_kill - 4, 0, X32_SYSCALL_BIT))
    for i, nr in enumerate(blocked):
        at = 4 + i
        program.append((BPF_JMP | BPF_JEQ | BPF_K, idx_eperm - (at + 1), 0, nr))
    program.append((BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ALLOW))
    program.append((BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | EPERM))
    program.append((BPF_RET | BPF_K, 0, 0, SECCOMP_RET_KILL_PROCESS))
    return program


def _apply_seccomp() -> None:
    lib = _libc()
    statements = build_seccomp_program()
    array_type = _SockFilter * len(statements)
    filters = array_type(*[_SockFilter(code, jt, jf, k) for code, jt, jf, k in statements])
    prog = _SockFprog(len(statements), filters)
    if lib.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise TrampolineError("no_new_privs_failed")
    if lib.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(prog)) != 0:
        err = ctypes.get_errno()
        raise TrampolineError(f"seccomp_install_failed: errno={err}")


def _close_inherited_fds(max_fd: int = 1024) -> None:
    for fd in range(3, max_fd):
        try:
            os.close(fd)
        except OSError:
            pass


def resolve_contained(scratch_root: Path, job_root: Path) -> Path:
    """Realpath containment proof; refuses anything outside the scratch root."""
    scratch_real = Path(os.path.realpath(str(scratch_root)))
    job_real = Path(os.path.realpath(str(job_root)))
    if job_real == scratch_real or scratch_real not in job_real.parents:
        raise TrampolineError("job_root_outside_scratch")
    if not job_real.is_dir():
        raise TrampolineError("job_root_not_found")
    return job_real


def run(argv: Sequence[str]) -> int:
    args = list(argv)
    job_root_raw: str | None = None
    command: List[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item == "--job-root" and index + 1 < len(args):
            job_root_raw = args[index + 1]
            index += 2
        elif item == "--":
            command = args[index + 1 :]
            break
        else:
            raise TrampolineError(f"unexpected_argument: {item}")
    if not job_root_raw:
        raise TrampolineError("job_root_required")
    if not command:
        raise TrampolineError("command_required")
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise TrampolineError("non_posix_host")
    if os.geteuid() == 0:
        raise TrampolineError("trampoline_must_not_run_as_root")
    if os.uname().machine != "x86_64":
        raise TrampolineError("unsupported_arch")
    abi = landlock_abi()
    if abi < MIN_LANDLOCK_ABI:
        raise TrampolineError(f"landlock_abi_too_old: {abi}")

    scratch_root = Path(os.environ.get("RSI_WORKER_SCRATCH_ROOT", "/mnt/local_sandbox_workdir"))
    job_root = resolve_contained(scratch_root, Path(job_root_raw))

    home_dir = job_root / ".home"
    tmp_dir = job_root / ".tmp"
    for path in (home_dir, tmp_dir):
        path.mkdir(mode=0o700, exist_ok=True)

    _close_inherited_fds()
    _apply_landlock(build_access_plan(scratch_root, job_root))
    _apply_seccomp()

    env: Dict[str, str] = dict(os.environ)
    env["HOME"] = str(home_dir)
    env["TMPDIR"] = str(tmp_dir)
    os.execvpe(command[0], command, env)
    return EXIT_FAILED  # unreachable; execvpe raises on failure


def main() -> None:
    try:
        code = run(sys.argv[1:])
    except TrampolineError as exc:
        print(f"exec-trampoline: REFUSED: {exc}", file=sys.stderr, flush=True)
        code = EXIT_FAILED
    except OSError as exc:
        print(f"exec-trampoline: REFUSED: os_error: {exc}", file=sys.stderr, flush=True)
        code = EXIT_FAILED
    sys.exit(code)


if __name__ == "__main__":
    main()
