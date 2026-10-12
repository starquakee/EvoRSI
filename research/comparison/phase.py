"""Server-side evaluator phase activation for the comparison (EXP-007).

The dispatcher loads its evaluator registry ONCE at startup from
``EVALUATOR_REGISTRY_PATH``. The search phase uses the tracked search
registry; the final phase uses the hash-bound phase registry published
under the ignored runtime publication root (mounted read-only at
``/opt/rsi_final``). Switching phases therefore means recreating ONLY the
dispatcher service with the registry override — always while the stack is
idle, always bound to the ORIGINAL experiment deadline, and always with a
durable failure sink (production passes the original ledger's ``stop``).

This module is the ONLY phase-switch mechanism: the live CLI
(run/resume) and the no-model preflight probes both use it, so the probe
exercises exactly what the live experiment will run.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn

PHASE_SEARCH = "search"
PHASE_FINAL = "final"

SEARCH_REGISTRY_CONTAINER_PATH = "/opt/rsi_repo/research/evaluator/registry.v1.json"
FINAL_REGISTRY_CONTAINER_PATH = "/opt/rsi_final/registry.final.v1.json"

COMPOSE_FILE = Path("deploy") / "rsi-trustworthy" / "docker-compose.yaml"
DISPATCHER = "rsi_dispatcher"
WORKER = "rsi_worker"
REGISTRY_ENV = "EVALUATOR_REGISTRY_PATH"
REGISTRY_OVERRIDE_ENV = "RSI_EVALUATOR_REGISTRY_PATH"
CANDIDATE_UID = "65432"

#: This checkout's root: the stack and its private auth file belong to it.
REPO_ROOT = Path(__file__).resolve().parents[2]

_HEALTH_TIMEOUT = 120.0
_DOCKER_TIMEOUT = 180.0
_COMPOSE_TIMEOUT = 300.0
_HTTP_TIMEOUT = 30.0


class PhaseActivationError(RuntimeError):
    """Phase activation/verification failure; the message IS the rule."""


def _bounded_timeout(
    default: float,
    deadline_epoch: float | None,
    now_fn: Callable[[], float],
) -> float:
    """Bound ONE blocking wait by the ORIGINAL remaining deadline.

    Never hands a subprocess/HTTP/poll wait a fresh independent duration when
    an experiment deadline applies; raises on expiry BEFORE the wait starts.
    """
    if deadline_epoch is None:
        return default
    remaining = deadline_epoch - now_fn()
    if remaining <= 0.0:
        raise PhaseActivationError("phase_activation_deadline_expired")
    return min(default, remaining)


def _docker(
    *args: str, timeout: float = _DOCKER_TIMEOUT, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return proc


def _docker_ok(*args: str, timeout: float = _DOCKER_TIMEOUT) -> str:
    proc = _docker(*args, timeout=timeout)
    if proc.returncode != 0:
        raise PhaseActivationError(
            f"docker_command_failed:{args[0]}:{proc.stderr.strip()[-160:]}"
        )
    return proc.stdout


def _dispatcher_env(timeout: float = _DOCKER_TIMEOUT) -> Mapping[str, str]:
    proc = _docker("inspect", DISPATCHER, "--format", "{{json .Config.Env}}", timeout=timeout)
    if proc.returncode != 0:
        raise PhaseActivationError("dispatcher_inspect_failed")
    env_list = json.loads(proc.stdout) or []
    return {
        entry.split("=", 1)[0]: entry.split("=", 1)[1]
        for entry in env_list
        if isinstance(entry, str) and "=" in entry
    }


def current_phase(timeout: float = _DOCKER_TIMEOUT) -> str:
    """The dispatcher's ACTIVE evaluator phase, from its container env."""
    registry = _dispatcher_env(timeout=timeout).get(REGISTRY_ENV)
    if registry == SEARCH_REGISTRY_CONTAINER_PATH:
        return PHASE_SEARCH
    if registry == FINAL_REGISTRY_CONTAINER_PATH:
        return PHASE_FINAL
    raise PhaseActivationError(f"dispatcher_registry_unknown:{registry}")


def _running(name: str, timeout: float = _DOCKER_TIMEOUT) -> bool:
    proc = _docker("inspect", name, "--format", "{{.State.Running}}", timeout=timeout)
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _candidate_processes(timeout: float = _DOCKER_TIMEOUT) -> list[str]:
    """Candidate-uid processes INSIDE the dedicated worker container only."""
    proc = _docker("exec", WORKER, "ps", "-eo", "uid=,args=", timeout=timeout)
    if proc.returncode != 0:
        raise PhaseActivationError("worker_scan_failed")
    return [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip().startswith(f"{CANDIDATE_UID} ")
    ]


def _active_jobs(
    repo_root: Path,
    *,
    deadline_epoch: float | None = None,
    now_fn: Callable[[], float] = time.time,
) -> int:
    """Queued+running jobs via the authenticated gateway (probe parity).

    A queued job has no candidate PID yet, so the process scan alone is not
    an idle proof. Uses the sandbox API key from the private auth file of the
    SUPPLIED repo root. Each HTTP wait is bounded by the original deadline
    when one applies.
    """
    import urllib.request

    from research.comparison import deploy
    from research.search.acceptance_inner import load_sandbox_api_key

    key = load_sandbox_api_key(Path(repo_root) / deploy.AUTH_FILE)
    total = 0
    for status in ("queued", "running"):
        request = urllib.request.Request(
            f"{deploy.GATEWAY_URL}/api/v1/jobs?status={status}&limit=50",
            headers={"X-API-Key": key},
        )
        with urllib.request.urlopen(
            request,
            timeout=_bounded_timeout(_HTTP_TIMEOUT, deadline_epoch, now_fn),
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        count = payload.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise PhaseActivationError("job_count_unreadable")
        total += count
    return total


def verify_idle(
    repo_root: Path,
    *,
    deadline_epoch: float | None = None,
    now_fn: Callable[[], float] = time.time,
) -> None:
    """Both services running; zero candidate processes; zero active jobs."""
    for name in (DISPATCHER, WORKER):
        if not _running(
            name, timeout=_bounded_timeout(_DOCKER_TIMEOUT, deadline_epoch, now_fn)
        ):
            raise PhaseActivationError(f"stack_not_idle:{name}_down")
    leftovers = _candidate_processes(
        timeout=_bounded_timeout(_DOCKER_TIMEOUT, deadline_epoch, now_fn)
    )
    if leftovers:
        raise PhaseActivationError(f"stack_not_idle:candidate_processes:{len(leftovers)}")
    active = _active_jobs(repo_root, deadline_epoch=deadline_epoch, now_fn=now_fn)
    if active:
        raise PhaseActivationError(f"stack_not_idle:active_jobs:{active}")


def _wait_dispatcher_running(
    now_fn: Callable[[], float], deadline_epoch: float | None = None
) -> None:
    wait_deadline = now_fn() + _HEALTH_TIMEOUT
    if deadline_epoch is not None:
        wait_deadline = min(wait_deadline, deadline_epoch)
    while True:
        if _running(
            DISPATCHER,
            timeout=_bounded_timeout(_DOCKER_TIMEOUT, deadline_epoch, now_fn),
        ):
            return
        if now_fn() >= wait_deadline:
            if deadline_epoch is not None and now_fn() >= deadline_epoch:
                raise PhaseActivationError("phase_activation_deadline_expired")
            raise PhaseActivationError("dispatcher_not_running_after_recreate")
        time.sleep(min(2.0, max(0.0, wait_deadline - now_fn())))


def _container_registry_sha256(path: str, timeout: float = _DOCKER_TIMEOUT) -> str:
    out = _docker_ok("exec", DISPATCHER, "sha256sum", path, timeout=timeout).strip()
    digest = out.split()[0] if out else ""
    if len(digest) != 64:
        raise PhaseActivationError("phase_registry_unreadable")
    return digest


def _compose_recreate_dispatcher(
    repo_root: Path, registry_path: str | None, *, timeout_seconds: float = _COMPOSE_TIMEOUT
) -> None:
    """Recreate ONLY the dispatcher; None selects the compose default.

    The blocking wait is bounded by ``timeout_seconds`` — production passes
    the ORIGINAL experiment deadline's remaining time, never an independent
    fresh duration.
    """
    import os

    env = dict(os.environ)
    env.pop(REGISTRY_OVERRIDE_ENV, None)
    if registry_path is not None:
        env[REGISTRY_OVERRIDE_ENV] = registry_path
    proc = subprocess.run(
        ["docker", "compose", "-f", str(Path(repo_root) / COMPOSE_FILE), "up", "-d", "dispatcher"],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env,
        cwd=Path(repo_root) / COMPOSE_FILE.parent,
    )
    if proc.returncode != 0:
        raise PhaseActivationError(
            f"dispatcher_recreate_failed:{proc.stderr.strip()[-160:]}"
        )


def _switch_phase(
    repo_root: Path,
    *,
    target: str,
    registry_path: str,
    expected_registry_sha256: str | None,
    deadline_epoch: float | None,
    now_fn: Callable[[], float],
    on_failure: Callable[[str], None],
) -> str:
    def _fail(reason: str) -> NoReturn:
        on_failure(reason)
        raise PhaseActivationError(reason)

    if deadline_epoch is not None and now_fn() >= deadline_epoch:
        _fail("phase_activation_deadline_expired")
    try:
        verify_idle(repo_root, deadline_epoch=deadline_epoch, now_fn=now_fn)
        phase_now = current_phase(
            timeout=_bounded_timeout(_DOCKER_TIMEOUT, deadline_epoch, now_fn)
        )
    except PhaseActivationError as exc:
        _fail(f"phase_precondition_failed:{exc}")
    except (subprocess.TimeoutExpired, OSError) as exc:
        _fail(f"phase_precondition_failed:{type(exc).__name__}")
    if phase_now == target:
        if expected_registry_sha256 is not None:
            try:
                actual = _container_registry_sha256(
                    registry_path,
                    timeout=_bounded_timeout(_DOCKER_TIMEOUT, deadline_epoch, now_fn),
                )
            except PhaseActivationError as exc:
                _fail(f"phase_registry_unverified:{exc}")
            except (subprocess.TimeoutExpired, OSError) as exc:
                _fail(f"phase_registry_unverified:{type(exc).__name__}")
            if actual != expected_registry_sha256:
                _fail("phase_registry_hash_mismatch")
        # The original deadline also bounds the SLOW verification work, not
        # just the entry check.
        if deadline_epoch is not None and now_fn() > deadline_epoch:
            _fail("phase_activation_deadline_expired")
        return f"already_{target}"
    try:
        _compose_recreate_dispatcher(
            repo_root,
            registry_path,
            timeout_seconds=_bounded_timeout(_COMPOSE_TIMEOUT, deadline_epoch, now_fn),
        )
        _wait_dispatcher_running(now_fn, deadline_epoch=deadline_epoch)
        phase_now = current_phase(
            timeout=_bounded_timeout(_DOCKER_TIMEOUT, deadline_epoch, now_fn)
        )
    except PhaseActivationError as exc:
        _fail(f"phase_switch_failed:{exc}")
    except (subprocess.TimeoutExpired, OSError) as exc:
        _fail(f"phase_switch_failed:{type(exc).__name__}")
    if phase_now != target:
        _fail("phase_env_mismatch_after_recreate")
    if deadline_epoch is not None and now_fn() >= deadline_epoch:
        _fail("phase_activation_deadline_expired")
    if expected_registry_sha256 is not None:
        try:
            actual_sha = _container_registry_sha256(
                registry_path,
                timeout=_bounded_timeout(_DOCKER_TIMEOUT, deadline_epoch, now_fn),
            )
        except PhaseActivationError as exc:
            _fail(f"phase_registry_unverified:{exc}")
        except (subprocess.TimeoutExpired, OSError) as exc:
            _fail(f"phase_registry_unverified:{type(exc).__name__}")
        if actual_sha != expected_registry_sha256:
            _fail("phase_registry_hash_mismatch")
    if deadline_epoch is not None and now_fn() >= deadline_epoch:
        _fail("phase_activation_deadline_expired")
    try:
        verify_idle(repo_root, deadline_epoch=deadline_epoch, now_fn=now_fn)
    except PhaseActivationError as exc:
        _fail(f"phase_postcondition_failed:{exc}")
    except (subprocess.TimeoutExpired, OSError) as exc:
        _fail(f"phase_postcondition_failed:{type(exc).__name__}")
    # The SLOW postcondition work itself must finish before the original
    # deadline; an activation that completes at/past it never returns success.
    if deadline_epoch is not None and now_fn() >= deadline_epoch:
        _fail("phase_activation_deadline_expired")
    return f"activated_{target}"


def activate_final_phase(
    repo_root: Path,
    *,
    expected_registry_sha256: str,
    deadline_epoch: float | None,
    now_fn: Callable[[], float] = time.time,
    on_failure: Callable[[str], None],
) -> str:
    """Switch the dispatcher to the hash-bound FINAL phase registry.

    Idempotent: an already-final dispatcher is accepted only when its
    container-visible registry bytes hash-match the published record.
    Every failure calls ``on_failure`` (production: the original ledger's
    durable stop) BEFORE raising.
    """
    return _switch_phase(
        repo_root,
        target=PHASE_FINAL,
        registry_path=FINAL_REGISTRY_CONTAINER_PATH,
        expected_registry_sha256=expected_registry_sha256,
        deadline_epoch=deadline_epoch,
        now_fn=now_fn,
        on_failure=on_failure,
    )


def restore_search_phase(
    repo_root: Path,
    *,
    now_fn: Callable[[], float] = time.time,
    on_failure: Callable[[str], None],
) -> str:
    """Switch the dispatcher back to the tracked SEARCH registry.

    Used by preflight probes to restore/prove the search phase; the live
    experiment itself never restores (the final phase is terminal).
    """
    return _switch_phase(
        repo_root,
        target=PHASE_SEARCH,
        registry_path=SEARCH_REGISTRY_CONTAINER_PATH,
        expected_registry_sha256=None,
        deadline_epoch=None,
        now_fn=now_fn,
        on_failure=on_failure,
    )
