"""Supervisor regressions: pure mocks/BPF interpretation, never host confinement."""
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'deploy/rsi-trustworthy/worker' / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

trampoline = load('supervisor_trampoline', 'exec_trampoline.py')
worker = load('supervisor_worker_confinement', 'minimal_worker.py')

def bpf_verdict(nr, arch=0xC000003E):
    program = trampoline.build_seccomp_program()
    acc = 0
    pc = 0
    for _ in range(len(program) + 2):
        code, jt, jf, k = program[pc]
        if code == 0x20:  # LD W ABS
            acc = {0: nr, 4: arch}[k]
        elif code in (0x15, 0x25, 0x35, 0x45):
            match = {0x15: acc == k, 0x25: acc > k, 0x35: acc >= k, 0x45: bool(acc & k)}[code]
            pc += jt if match else jf
        elif code == 0x06:
            return k
        else:
            raise AssertionError(f'unsupported BPF instruction {code}')
        pc += 1
    raise AssertionError('BPF did not terminate')

@pytest.mark.parametrize('nr', [90, 91, 268, 452, 92, 93, 94, 260,
                                  132, 235, 261, 280, 188, 189, 190, 197, 198, 199])
def test_landlock_unmediated_metadata_mutation_denied(nr):
    assert bpf_verdict(nr) != 0x7FFF0000

@pytest.mark.parametrize('nr', [0x40000000, 0x40000000 | 90, 0x40000000 | 268])
def test_x32_numbers_cannot_bypass_metadata_filter(nr):
    assert bpf_verdict(nr) != 0x7FFF0000

def test_filter_preserves_ordinary_read_write_and_getpid():
    for nr in (0, 1, 39):
        assert bpf_verdict(nr) == 0x7FFF0000

def test_no_global_shared_memory_write_grant(tmp_path):
    scratch = tmp_path / 'scratch'
    job = scratch / 'job'
    job.mkdir(parents=True)
    plan = trampoline.build_access_plan(scratch, job)
    for path, rights in plan:
        if path in ('/tmp', '/dev/shm', str(scratch)):
            assert rights & (trampoline.ACCESS_FS_WRITE_FILE | trampoline.ACCESS_FS_MAKE_REG) == 0

def test_python_confinement_helper_starts_isolated_without_site(monkeypatch, tmp_path):
    captured = {}
    class StopBeforeSpawn(Exception):
        pass
    def popen(argv, **kwargs):
        captured['argv'] = argv
        captured['kwargs'] = kwargs
        raise StopBeforeSpawn()
    monkeypatch.setattr(worker.subprocess, 'Popen', popen)
    cfg = worker.WorkerConfig(control_key=b'example-control-key', candidate_uid=65432,
                              candidate_gid=65432, guard_active=True)
    session = worker.Session('fixture', 'echo fixture', str(tmp_path), cfg, None,
                             exec_class='candidate', job_root=str(tmp_path))
    with pytest.raises(StopBeforeSpawn):
        session.start()
    argv = captured['argv']
    flags = ''.join(a[1:] for a in argv[1:argv.index(cfg.trampoline_path)] if a.startswith('-'))
    assert 'I' in flags and 'S' in flags, argv
    assert captured['kwargs'].get('close_fds', True) is True

def test_missing_worker_session_is_not_cleanup_proof(monkeypatch):
    import test_dispatcher_worker_control as fixture
    td = fixture.td
    class Controller:
        def __init__(self, *args): pass
        def exec_command(self, **kwargs):
            return {'session_id': 'missing-after-start', 'status': 'running'}
        def kill(self, session_id): return None
        def cleanup(self, session_id): return False
    monkeypatch.setattr(td, 'SandboxHTTPController', Controller)
    with pytest.raises(td.SandboxCommandTimeoutError) as error:
        td.execute_shell_command_with_deadline(worker_endpoint='http://fixture.invalid',
            command='never executed', exec_dir=None, job_id='fixture', redis_client=None,
            db_conn=None, job_deadline=0)
    assert error.value.cleanup_verified is False


def test_wsl_dxg_gpu_device_has_specific_read_write_rule(monkeypatch, tmp_path):
    # WSL CUDA uses this real device rather than /dev/nvidia0. No syscall,
    # CUDA invocation, or sandbox setup runs on the development host.
    monkeypatch.setattr(trampoline.os.path, 'exists', lambda p: p == '/dev/dxg')
    monkeypatch.setattr(trampoline.os.path, 'isdir', lambda p: False)
    monkeypatch.setattr(trampoline.glob, 'glob', lambda pattern: [])
    plan = trampoline.build_access_plan(tmp_path, tmp_path / 'job')
    assert ('/dev/dxg', trampoline.DEVICE_RW) in plan
