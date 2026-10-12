"""Controller-side wrapper for the stateful native iStratDE driver (EXP-006).

The native algorithm runs ONLY as a subprocess under the pinned legacy
interpreter (``native-env.json``: JAX 0.4.33, CPU; the research venv's newer
JAX is incompatible and the legacy environment is never upgraded). This
wrapper owns the subprocess contract and strict output validation:

- exact batch shape (pop 4 x 5 dims), strict float typing, finiteness and
  encoded-bounds membership are enforced on every returned vector;
- the response mode and seed must echo the request (a stale or foreign
  driver output never enters the comparison);
- float32 trial values are converted to Python floats once, here, so the
  RANDOM method's paired first four and the ISTRATDE first batch consume
  the IDENTICAL persisted numbers (PRD: matched including dtype
  conversion and decoding).

The provider never sees scores for the first batch, and the second batch
consumes exactly the accepted fitness history of the SAME (task, search
seed) pair — never another cell's.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from research.comparison.protocol import COMPARISON_DIMS

#: Pinned native environment configuration (deployment topology, not an
#: experiment parameter). Resolved against the repo root, never a bare
#: module-level absolute literal.
NATIVE_ENV_FILE = Path("research") / "comparison" / "native-env.json"

#: The standalone driver executed by the legacy interpreter.
NATIVE_DRIVER_FILE = Path(__file__).with_name("native_driver.py")

POP_SIZE = 4
DIMS = len(COMPARISON_DIMS)

#: Hard wall-clock bound for one driver invocation (JAX import dominates).
NATIVE_DRIVER_TIMEOUT_SECONDS = 600


class NativeDriverError(RuntimeError):
    """The native driver could not produce a trustworthy batch."""


@dataclass(frozen=True)
class NativeCompletion:
    """Verifiable end-of-run evidence of the full two-batch feedback loop.

    ``first_batch``/``second_batch`` are the driver's echoed batches (the
    caller MUST verify they equal the accepted vectors bit-for-bit);
    ``final_population_sha256`` is the bit-exact sha256 of the post-second-
    tell population (float32 bytes), reproducible ONLY by replaying the
    frozen seed and BOTH accepted fitness histories through the real
    algorithm — the proof that the second tell actually happened.
    """

    first_batch: tuple[tuple[float, ...], ...]
    second_batch: tuple[tuple[float, ...], ...]
    final_population_sha256: str


class NativeProvider(Protocol):
    """Batch source consumed by the method theta plans (fakeable in tests)."""

    def setup_population(self, seed: int) -> tuple[tuple[float, ...], ...]: ...

    def first_batch(self, seed: int) -> tuple[tuple[float, ...], ...]: ...

    def second_batch(
        self, seed: int, fitness: Sequence[float]
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]: ...

    def complete_run(
        self,
        seed: int,
        first_fitness: Sequence[float],
        second_fitness: Sequence[float],
    ) -> NativeCompletion: ...


def load_native_python(repo_root: Path) -> Path:
    """Resolve and validate the pinned legacy interpreter path."""
    env_path = repo_root / NATIVE_ENV_FILE
    if env_path.is_symlink() or not env_path.exists():
        raise NativeDriverError("native_env_missing")
    try:
        payload = json.loads(env_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NativeDriverError("native_env_malformed") from exc
    if not isinstance(payload, dict) or payload.get("jax_version") != "0.4.33":
        raise NativeDriverError("native_env_malformed")
    python = payload.get("python")
    if not isinstance(python, str) or not python:
        raise NativeDriverError("native_env_malformed")
    path = Path(python).expanduser()
    if not path.is_file():
        raise NativeDriverError("native_python_missing")
    return path


def _validate_batch(value: Any, rule: str) -> tuple[tuple[float, ...], ...]:
    """Strict batch shape/type/finiteness/bounds validation (fail closed)."""
    if not isinstance(value, list) or len(value) != POP_SIZE:
        raise NativeDriverError(rule)
    rows: list[tuple[float, ...]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != DIMS:
            raise NativeDriverError(rule)
        vector: list[float] = []
        for dim, element in zip(COMPARISON_DIMS, row):
            if (
                isinstance(element, bool)
                or not isinstance(element, (int, float))
                or not math.isfinite(float(element))
            ):
                raise NativeDriverError(rule)
            number = float(element)
            if not dim.lower <= number <= dim.upper:
                raise NativeDriverError("native_vector_out_of_bounds")
            vector.append(number)
        rows.append(tuple(vector))
    return tuple(rows)


def _check_fitness(fitness: Sequence[float]) -> list[float]:
    if len(fitness) != POP_SIZE or any(
        isinstance(f, bool)
        or not isinstance(f, (int, float))
        or not math.isfinite(float(f))
        for f in fitness
    ):
        raise NativeDriverError("native_fitness_invalid")
    return [float(f) for f in fitness]


def _run_driver(
    repo_root: Path,
    mode: str,
    seed: int,
    *,
    fitness: Sequence[float] | None = None,
    second_fitness: Sequence[float] | None = None,
    timeout: int = NATIVE_DRIVER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    python = load_native_python(repo_root)
    argv = [str(python), str(NATIVE_DRIVER_FILE), str(repo_root), mode, str(seed)]
    if fitness is not None:
        argv.append(",".join(repr(f) for f in _check_fitness(fitness)))
    if second_fitness is not None:
        argv.append(",".join(repr(f) for f in _check_fitness(second_fitness)))
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(repo_root),
        )
    except subprocess.TimeoutExpired as exc:
        raise NativeDriverError("native_driver_timeout") from exc
    except OSError as exc:
        raise NativeDriverError("native_driver_unavailable") from exc
    if proc.returncode != 0:
        raise NativeDriverError("native_driver_failed")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise NativeDriverError("native_driver_silent")
    try:
        payload = json.loads(lines[-1])
    except ValueError as exc:
        raise NativeDriverError("native_driver_output_malformed") from exc
    if not isinstance(payload, dict):
        raise NativeDriverError("native_driver_output_malformed")
    if payload.get("mode") != mode or payload.get("seed") != seed:
        raise NativeDriverError("native_driver_identity_mismatch")
    return payload


class SubprocessNativeProvider:
    """The real native provider: one pinned-interpreter subprocess per call."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root

    def setup_population(self, seed: int) -> tuple[tuple[float, ...], ...]:
        payload = _run_driver(self._repo_root, "setup", seed)
        return _validate_batch(payload.get("population"), "native_batch_invalid")

    def first_batch(self, seed: int) -> tuple[tuple[float, ...], ...]:
        payload = _run_driver(self._repo_root, "first-batch", seed)
        return _validate_batch(payload.get("first_batch"), "native_batch_invalid")

    def second_batch(
        self, seed: int, fitness: Sequence[float]
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
        """Ask the feedback batch from state threaded through tell.

        Returns ``(echoed_first_batch, second_batch)``; the caller MUST
        verify the echo equals the accepted first batch bit-for-bit (the
        frozen seed + fitness history replay check).
        """
        payload = _run_driver(self._repo_root, "second-batch", seed, fitness=fitness)
        first = _validate_batch(payload.get("first_batch"), "native_batch_invalid")
        second = _validate_batch(payload.get("second_batch"), "native_batch_invalid")
        echoed_fitness = payload.get("fitness")
        if echoed_fitness != [float(f) for f in fitness]:
            raise NativeDriverError("native_driver_identity_mismatch")
        return first, second

    def complete_run(
        self,
        seed: int,
        first_fitness: Sequence[float],
        second_fitness: Sequence[float],
    ) -> NativeCompletion:
        """Replay BOTH accepted fitness histories and perform BOTH tells.

        Returns the echoed batches plus the final post-second-tell
        population hash; the caller MUST verify the echoes equal the
        accepted first/second batches bit-for-bit (the replayed vectors
        are the vectors actually evaluated) and persist the final hash as
        end-of-run evidence.
        """
        payload = _run_driver(
            self._repo_root,
            "complete",
            seed,
            fitness=first_fitness,
            second_fitness=second_fitness,
        )
        first = _validate_batch(payload.get("first_batch"), "native_batch_invalid")
        second = _validate_batch(payload.get("second_batch"), "native_batch_invalid")
        if payload.get("fitness_first") != [float(f) for f in first_fitness]:
            raise NativeDriverError("native_driver_identity_mismatch")
        if payload.get("fitness_second") != [float(f) for f in second_fitness]:
            raise NativeDriverError("native_driver_identity_mismatch")
        final_hash = payload.get("final_population_sha256")
        if (
            not isinstance(final_hash, str)
            or len(final_hash) != 64
            or any(c not in "0123456789abcdef" for c in final_hash)
        ):
            raise NativeDriverError("native_final_state_invalid")
        return NativeCompletion(
            first_batch=first,
            second_batch=second,
            final_population_sha256=final_hash,
        )
