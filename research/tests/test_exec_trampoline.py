"""US-008 offline tests for the candidate pre-exec sandbox trampoline.

Exercises deploy/rsi-trustworthy/worker/exec_trampoline.py on the host
WITHOUT applying any real confinement: Landlock/seccomp application and
exec are mocked or simulated. The REAL confinement proof (cross-job
read/write/chmod denial, GPU still working) runs inside the new worker
container via deploy/rsi-trustworthy/probe_us008.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAMPOLINE_PATH = REPO_ROOT / "deploy" / "rsi-trustworthy" / "worker" / "exec_trampoline.py"

spec = importlib.util.spec_from_file_location("exec_trampoline", TRAMPOLINE_PATH)
assert spec is not None and spec.loader is not None
tramp = importlib.util.module_from_spec(spec)
sys.modules["exec_trampoline"] = tramp
spec.loader.exec_module(tramp)


# --- access plan -----------------------------------------------------------


def test_access_plan_scopes_writes_to_job_root(tmp_path):
    scratch = tmp_path / "scratch"
    job = scratch / "jobs-candidate-v1" / "job-1"
    job.mkdir(parents=True)
    plan = dict(tramp.build_access_plan(scratch, job))
    assert plan[str(job)] == tramp.ALL_FS_RIGHTS
    # The scratch parent itself and other job trees get NO rule at all.
    assert str(scratch) not in plan
    for root in ("/usr", "/bin", "/etc"):
        if Path(root).exists():
            assert plan[root] == tramp.READ_EXEC
    for public in ("/mnt/rsi_data", "/mnt/rsi_storage"):
        assert plan.get(public) in (None, tramp.READ_ONLY)
    # No shared writable escape: no /tmp, /dev/shm or scratch-root grant.
    assert "/tmp" not in plan
    assert "/dev/shm" not in plan
    assert str(scratch) not in plan
    # Device nodes needed by CUDA get exactly read+write, never exec/dir.
    for path, rights in plan.items():
        if path.startswith("/dev/nvidia") or path in ("/dev/null", "/dev/urandom"):
            assert rights == tramp.DEVICE_RW


def test_resolve_contained_fail_closed(tmp_path):
    scratch = tmp_path / "scratch"
    job = scratch / "jobs" / "job-1"
    job.mkdir(parents=True)
    assert tramp.resolve_contained(scratch, job) == job
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(tramp.TrampolineError, match="job_root_outside_scratch"):
        tramp.resolve_contained(scratch, outside)
    with pytest.raises(tramp.TrampolineError, match="job_root_outside_scratch"):
        tramp.resolve_contained(scratch, scratch)  # root itself is not a job
    with pytest.raises(tramp.TrampolineError, match="job_root_outside_scratch"):
        tramp.resolve_contained(scratch, scratch / ".." / "scratch" / "jobs" / ".." / ".." / "elsewhere")
    with pytest.raises(tramp.TrampolineError, match="job_root_not_found"):
        tramp.resolve_contained(scratch, scratch / "jobs" / "absent")
    # Symlink alias pointing outside the scratch root is resolved away.
    alias = scratch / "jobs" / "link"
    alias.symlink_to(outside)
    with pytest.raises(tramp.TrampolineError, match="job_root_outside_scratch"):
        tramp.resolve_contained(scratch, alias)


# --- seccomp BPF program (executed against a tiny simulator) ----------------

BPF_LD, BPF_W, BPF_ABS = 0x00, 0x00, 0x20
BPF_JMP, BPF_JEQ, BPF_K = 0x05, 0x10, 0x00
BPF_RET = 0x06


def _simulate(program, *, arch, nr):
    acc = 0
    pc = 0
    while True:
        code, jt, jf, k = program[pc]
        if code == BPF_LD | BPF_W | BPF_ABS:
            acc = {0: nr, 4: arch}[k]
            pc += 1
        elif code in (BPF_JMP | BPF_JEQ | BPF_K, BPF_JMP | 0x40 | BPF_K):
            match = (acc == k) if code == BPF_JMP | BPF_JEQ | BPF_K else bool(acc & k)
            pc += 1 + (jt if match else jf)
        elif code == BPF_RET | BPF_K:
            return k
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected opcode {code}")


def test_seccomp_program_blocks_metadata_mutations_only():
    program = tramp.build_seccomp_program()
    x86 = tramp.AUDIT_ARCH_X86_64
    for name, nr in tramp.BLOCKED_METADATA_SYSCALLS:
        result = _simulate(program, arch=x86, nr=nr)
        assert result & 0xFFFF0000 == tramp.SECCOMP_RET_ERRNO, name
        assert result & 0xFFFF == tramp.EPERM, name
    # chmod/chown + utime + xattr families are all covered.
    assert {nr for _, nr in tramp.BLOCKED_METADATA_SYSCALLS} == {
        90, 91, 268, 452, 92, 93, 94, 260, 132, 235, 261, 280, 188, 189, 190, 197, 198, 199
    }
    # Ordinary syscalls stay allowed (markers use open/write redirection).
    for nr in (0, 1, 2, 39, 444, 445, 446):  # read/write/open/getpid/landlock
        assert _simulate(program, arch=x86, nr=nr) == tramp.SECCOMP_RET_ALLOW, nr
    # x32-ABI numbers cannot bypass the per-number checks.
    for nr in (tramp.X32_SYSCALL_BIT, tramp.X32_SYSCALL_BIT | 90, tramp.X32_SYSCALL_BIT | 268):
        assert _simulate(program, arch=x86, nr=nr) == tramp.SECCOMP_RET_KILL_PROCESS, nr
    # Foreign architectures fail closed with KILL (syscall numbers differ).
    assert _simulate(program, arch=0x40000028, nr=90) == tramp.SECCOMP_RET_KILL_PROCESS


# --- fail-closed preconditions ----------------------------------------------


def test_landlock_abi_query_handles_unavailable():
    class FakeLibc:
        def syscall(self, *args):
            return -1

    assert tramp.landlock_abi(FakeLibc()) == 0

    class Abi3:
        def syscall(self, *args):
            return 3

    assert tramp.landlock_abi(Abi3()) == 3


def test_run_refuses_missing_arguments():
    with pytest.raises(tramp.TrampolineError, match="job_root_required"):
        tramp.run(["--", "/bin/true"])
    with pytest.raises(tramp.TrampolineError, match="command_required"):
        tramp.run(["--job-root", "/x", "--"])
    with pytest.raises(tramp.TrampolineError, match="unexpected_argument"):
        tramp.run(["--bogus"])


def test_run_refuses_root_and_foreign_arch(monkeypatch, tmp_path):
    scratch = tmp_path / "scratch"
    job = scratch / "job"
    job.mkdir(parents=True)
    monkeypatch.setenv("RSI_WORKER_SCRATCH_ROOT", str(scratch))
    monkeypatch.setattr(tramp.os, "geteuid", lambda: 0)
    with pytest.raises(tramp.TrampolineError, match="trampoline_must_not_run_as_root"):
        tramp.run(["--job-root", str(job), "--", "/bin/true"])
    monkeypatch.setattr(tramp.os, "geteuid", lambda: 65432)

    class Uname:
        machine = "aarch64"

    monkeypatch.setattr(tramp.os, "uname", lambda: Uname())
    with pytest.raises(tramp.TrampolineError, match="unsupported_arch"):
        tramp.run(["--job-root", str(job), "--", "/bin/true"])

    class X86:
        machine = "x86_64"

    monkeypatch.setattr(tramp.os, "uname", lambda: X86())
    monkeypatch.setattr(tramp, "landlock_abi", lambda: 2)
    with pytest.raises(tramp.TrampolineError, match="landlock_abi_too_old"):
        tramp.run(["--job-root", str(job), "--", "/bin/true"])


def test_run_applies_confinement_in_order_before_exec(monkeypatch, tmp_path):
    scratch = tmp_path / "scratch"
    job = scratch / "jobs-candidate-v1" / "job-9"
    job.mkdir(parents=True)
    monkeypatch.setenv("RSI_WORKER_SCRATCH_ROOT", str(scratch))
    monkeypatch.setattr(tramp.os, "geteuid", lambda: 65432)
    monkeypatch.setattr(tramp, "landlock_abi", lambda: 3)

    calls: list[str] = []
    monkeypatch.setattr(tramp, "_apply_landlock", lambda plan: calls.append("landlock"))
    monkeypatch.setattr(tramp, "_apply_seccomp", lambda: calls.append("seccomp"))
    monkeypatch.setattr(tramp, "_close_inherited_fds", lambda: calls.append("close_fds"))
    execed: dict = {}

    def fake_execvpe(file, args, env):
        execed.update(file=file, args=args, env=env)
        calls.append("exec")

    monkeypatch.setattr(tramp.os, "execvpe", fake_execvpe)

    tramp.run(["--job-root", str(job), "--", "/bin/bash", "-c", "echo hi"])

    assert calls == ["close_fds", "landlock", "seccomp", "exec"]
    assert execed["args"] == ["/bin/bash", "-c", "echo hi"]
    # HOME/TMPDIR are job-local, never the shared /tmp.
    assert execed["env"]["HOME"] == str(job / ".home")
    assert execed["env"]["TMPDIR"] == str(job / ".tmp")
    assert (job / ".home").is_dir() and (job / ".tmp").is_dir()
