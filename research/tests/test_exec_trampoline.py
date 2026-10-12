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
    # EXP-003: NO broad public read roots exist any more, even when the
    # directories exist; only exact validated grants get READ_ONLY rules.
    assert not hasattr(tramp, "PUBLIC_READ_ROOTS")
    assert not any(key.startswith("/mnt/rsi_") for key in plan)
    # No shared writable escape: no /tmp, /dev/shm or scratch-root grant.
    assert "/tmp" not in plan
    assert "/dev/shm" not in plan
    assert str(scratch) not in plan
    # Device nodes needed by CUDA get exactly read+write, never exec/dir.
    for path, rights in plan.items():
        if path.startswith("/dev/nvidia") or path in ("/dev/null", "/dev/urandom"):
            assert rights == tramp.DEVICE_RW


def test_access_plan_adds_exact_validated_grants_only(tmp_path):
    scratch = tmp_path / "scratch"
    job = scratch / "job"
    job.mkdir(parents=True)
    grant = tmp_path / "data" / "task" / "public"
    grant.mkdir(parents=True)
    plan = dict(tramp.build_access_plan(scratch, job, [grant, grant]))  # duplicates collapse
    assert plan[str(grant)] == tramp.READ_ONLY
    assert len([key for key in plan if str(grant) in key]) == 1
    # A grant never widens to its parents.
    assert str(grant.parent) not in plan
    # The grant gets no write rights of any kind.
    assert plan[str(grant)] & (
        tramp.ACCESS_FS_WRITE_FILE | tramp.ACCESS_FS_MAKE_REG | tramp.ACCESS_FS_MAKE_DIR
    ) == 0


# --- read grant validation (fail closed) -------------------------------------


def test_validate_read_grant_accepts_canonical_existing_dir(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    root = tmp_path / "mnt" / "rsi_data"
    grant = root / "task" / "data" / "public"
    grant.mkdir(parents=True)
    validated = tramp.validate_read_grant(str(grant), scratch, allowed_leaves={str(grant)})
    assert validated == grant


def test_validate_read_grant_rejects_non_leaf_paths(tmp_path):
    # Rejection derives from the pinned exact-leaf policy, not from any
    # missing-directory accident: parents, unregistered task-shaped paths
    # and children of leaves are refused even when they exist.
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    root = tmp_path / "mnt" / "rsi_data"
    leaf = root / "task" / "data" / "public"
    (leaf / "child").mkdir(parents=True)
    for bad in (str(root), str(root / "task"), str(root / "unregistered" / "data" / "public"), str(leaf / "child")):
        Path(bad).mkdir(parents=True, exist_ok=True)
        with pytest.raises(tramp.TrampolineError, match="read_grant_not_allowlisted"):
            tramp.validate_read_grant(bad, scratch, allowed_leaves={str(leaf)})


def test_validate_read_grant_rejects_noncanonical_and_relative(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    root = tmp_path / "mnt" / "rsi_data"
    grant = root / "task"
    grant.mkdir(parents=True)
    leaves = {str(grant)}
    for bad in (
        "relative/path",
        str(grant) + "/",
        str(grant) + "/../task",
        str(root) + "//task",
        "",
        123,
    ):
        with pytest.raises(tramp.TrampolineError):
            tramp.validate_read_grant(bad, scratch, allowed_leaves=leaves)


def test_validate_read_grant_rejects_symlink_alias(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    root = tmp_path / "mnt" / "rsi_data"
    target = tmp_path / "secret"
    target.mkdir()
    (root / "task").mkdir(parents=True)
    alias = root / "task" / "alias"
    alias.symlink_to(target)
    with pytest.raises(tramp.TrampolineError, match="read_grant_symlink"):
        tramp.validate_read_grant(str(alias), scratch, allowed_leaves={str(alias)})


def test_validate_read_grant_rejects_scratch_outside_roots_and_missing(tmp_path):
    scratch = tmp_path / "scratch"
    inside = scratch / "jobs" / "job-1"
    inside.mkdir(parents=True)
    root = tmp_path / "mnt" / "rsi_data"
    root.mkdir(parents=True)
    with pytest.raises(tramp.TrampolineError, match="read_grant_inside_scratch"):
        tramp.validate_read_grant(str(inside), scratch, allowed_leaves={str(inside)})
    with pytest.raises(tramp.TrampolineError, match="read_grant_not_allowlisted"):
        tramp.validate_read_grant(str(tmp_path / "elsewhere"), scratch, allowed_leaves={str(root / "task")})
    with pytest.raises(tramp.TrampolineError, match="read_grant_not_found"):
        tramp.validate_read_grant(str(root / "absent"), scratch, allowed_leaves={str(root / "absent")})


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
    grant_root = tmp_path / "mnt" / "rsi_data"
    grant = grant_root / "task" / "data" / "public"
    grant.mkdir(parents=True)
    monkeypatch.setenv("RSI_WORKER_SCRATCH_ROOT", str(scratch))
    monkeypatch.setattr(tramp.os, "geteuid", lambda: 65432)
    monkeypatch.setattr(tramp, "landlock_abi", lambda: 3)
    monkeypatch.setattr(tramp, "ALLOWED_READ_GRANT_LEAVES", frozenset({str(grant)}))

    calls: list[str] = []
    plans: list = []
    monkeypatch.setattr(tramp, "_apply_landlock", lambda plan: (calls.append("landlock"), plans.append(plan)))
    monkeypatch.setattr(tramp, "_apply_seccomp", lambda: calls.append("seccomp"))
    monkeypatch.setattr(tramp, "_close_inherited_fds", lambda: calls.append("close_fds"))
    execed: dict = {}

    def fake_execvpe(file, args, env):
        execed.update(file=file, args=args, env=env)
        calls.append("exec")

    monkeypatch.setattr(tramp.os, "execvpe", fake_execvpe)

    tramp.run([
        "--job-root", str(job),
        "--read-grant", str(grant),
        "--", "/bin/bash", "-c", "echo hi",
    ])

    assert calls == ["close_fds", "landlock", "seccomp", "exec"]
    assert execed["args"] == ["/bin/bash", "-c", "echo hi"]
    # The exact grant gets READ_ONLY; nothing else under the data root.
    plan = dict(plans[0])
    assert plan[str(grant)] == tramp.READ_ONLY
    assert str(grant_root) not in plan
    # HOME/TMPDIR are job-local, never the shared /tmp.
    assert execed["env"]["HOME"] == str(job / ".home")
    assert execed["env"]["TMPDIR"] == str(job / ".tmp")
    assert (job / ".home").is_dir() and (job / ".tmp").is_dir()


def test_run_refuses_invalid_or_excess_grants_before_exec(monkeypatch, tmp_path):
    scratch = tmp_path / "scratch"
    job = scratch / "job"
    job.mkdir(parents=True)
    root = tmp_path / "mnt" / "rsi_data"
    root.mkdir(parents=True)
    monkeypatch.setenv("RSI_WORKER_SCRATCH_ROOT", str(scratch))
    monkeypatch.setattr(tramp.os, "geteuid", lambda: 65432)
    monkeypatch.setattr(tramp, "landlock_abi", lambda: 3)
    monkeypatch.setattr(tramp, "_apply_landlock", lambda plan: None)
    monkeypatch.setattr(tramp, "_apply_seccomp", lambda: None)
    monkeypatch.setattr(tramp.os, "execvpe", lambda *a: None)
    # Granting a broad parent is refused by the exact-leaf policy.
    monkeypatch.setattr(tramp, "ALLOWED_READ_GRANT_LEAVES", frozenset({str(root / "leaf")}))
    with pytest.raises(tramp.TrampolineError, match="read_grant_not_allowlisted"):
        tramp.run(["--job-root", str(job), "--read-grant", str(root), "--", "/bin/true"])
    # Two distinct task grants conflict even when both are allowlisted.
    leaf_a = root / "a"
    leaf_b = root / "b"
    leaf_a.mkdir()
    leaf_b.mkdir()
    monkeypatch.setattr(tramp, "ALLOWED_READ_GRANT_LEAVES", frozenset({str(leaf_a), str(leaf_b)}))
    with pytest.raises(tramp.TrampolineError, match="conflicting_read_grants"):
        tramp.run([
            "--job-root", str(job),
            "--read-grant", str(leaf_a), "--read-grant", str(leaf_b),
            "--", "/bin/true",
        ])
    # An allowlisted leaf that does not exist fails closed.
    monkeypatch.setattr(tramp, "ALLOWED_READ_GRANT_LEAVES", frozenset({str(root / "absent")}))
    with pytest.raises(tramp.TrampolineError, match="read_grant_not_found"):
        tramp.run(["--job-root", str(job), "--read-grant", str(root / "absent"), "--", "/bin/true"])
    excess = [arg for _ in range(tramp.MAX_READ_GRANTS + 1) for arg in ("--read-grant", str(root / "x"))]
    with pytest.raises(tramp.TrampolineError, match="too_many_read_grants"):
        tramp.run(["--job-root", str(job), *excess, "--", "/bin/true"])
