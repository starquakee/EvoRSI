"""Standalone native iStratDE batch driver (EXP-006).

Executed ONLY as a subprocess under the pinned legacy interpreter
(``~/rsi-gpu/.venv/bin/python``, JAX 0.4.33, CPU backend —
pinned in ``native-env.json``). The research venv carries an incompatible
newer JAX, so the controller never imports the native module in-process.

The driver initializes the vendored ``IStratDE`` ONCE per invocation and
preserves the returned state into the next ask (the existing
``_NativeOuterAlgorithm`` reinitializes per ask and is deliberately NOT
used):

- ``setup``        : init(seed), print the setup population (NOT a batch).
- ``first-batch``  : init(seed), ask once, print the first trial batch.
- ``second-batch`` : init(seed), ask, tell the four first-batch fitness
                     values (native minimizes: fitness = -utility, failed
                     config utility 0), then ask again FROM THE PRESERVED
                     STATE and print both batches. Re-deriving the first
                     batch inside the same session is the approved replay:
                     the echoed first batch lets the controller verify the
                     frozen seed + fitness history bit-for-bit, and no model
                     call is ever issued to reconstruct state.
- ``complete``     : init(seed), ask, tell first-batch fitness, ask FROM
                     THE PRESERVED STATE, tell the second-batch fitness,
                     then print both batches, both fitness echoes and a
                     bit-exact sha256 of the FINAL post-second-tell
                     population (float32 bytes). This is the full two-batch
                     feedback replay: the second tell is a real algorithm
                     state transition, and the echoed batches let the
                     controller verify that the replayed vectors equal the
                     vectors it actually evaluated.

Output contract: exactly one JSON object on the last stdout line;
everything else (JAX banners) may only appear before it. Any violation,
version drift or non-vendored import exits nonzero with no JSON.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

POP_SIZE = 4
JAX_VERSION_PIN = "0.4.33"


def _fail(reason: str) -> "SystemExit":
    print(f"native_driver:{reason}", file=sys.stderr)
    return SystemExit(2)


def _load(repo_root: Path):
    vendor_src = (repo_root / "research" / "vendor" / "istratde" / "src").resolve()
    if not vendor_src.is_dir():
        raise _fail("vendor_src_missing")
    sys.path.insert(0, str(vendor_src))
    sys.path.insert(0, str(repo_root))
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import istratde  # noqa: PLC0415
    from istratde.algorithms.jax import IStratDE  # noqa: PLC0415

    if jax.__version__ != JAX_VERSION_PIN:
        raise _fail("jax_version_mismatch")
    if jax.default_backend() != "cpu":
        raise _fail("jax_platform_must_be_cpu")
    module_file = Path(istratde.__file__ or "").resolve()
    if not module_file.is_relative_to(vendor_src):
        raise _fail("istratde_not_vendored")

    from research.comparison.protocol import COMPARISON_DIMS  # noqa: PLC0415

    lb = jnp.array([dim.lower for dim in COMPARISON_DIMS], dtype=jnp.float32)
    ub = jnp.array([dim.upper for dim in COMPARISON_DIMS], dtype=jnp.float32)
    return jax, jnp, IStratDE, lb, ub


def _vectors(trials) -> list[list[float]]:
    rows = [[float(v) for v in row] for row in trials]
    if len(rows) != POP_SIZE or any(len(row) != 5 for row in rows):
        raise _fail("batch_shape_mismatch")
    return rows


def _parse_fitness(arg: str) -> list[float]:
    try:
        fitness = [float(part) for part in arg.split(",")]
    except ValueError:
        raise _fail("fitness_invalid") from None
    if len(fitness) != POP_SIZE or any(not math.isfinite(x) for x in fitness):
        raise _fail("fitness_invalid")
    return fitness


def _population_sha256(jax, state) -> str:
    import hashlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    population = np.asarray(jax.device_get(state.population), dtype=np.float32)
    if population.shape != (POP_SIZE, 5):
        raise _fail("batch_shape_mismatch")
    return hashlib.sha256(population.tobytes()).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        raise _fail("usage")
    repo_root = Path(argv[1]).resolve()
    mode = argv[2]
    try:
        seed = int(argv[3])
    except ValueError:
        raise _fail("seed_invalid") from None
    if seed < 0 or seed >= 2**32:
        raise _fail("seed_invalid")

    jax, jnp, IStratDE, lb, ub = _load(repo_root)
    algo = IStratDE(lb=lb, ub=ub, pop_size=POP_SIZE)
    # Initialize ONCE; the state object threads through every ask/tell.
    state = algo.init(jax.random.PRNGKey(seed))

    if mode == "setup":
        payload = {"mode": mode, "seed": seed, "population": _vectors(state.population)}
    elif mode == "first-batch":
        batch, _state = algo.ask(state)
        payload = {"mode": mode, "seed": seed, "first_batch": _vectors(batch)}
    elif mode == "second-batch":
        if len(argv) != 5:
            raise _fail("usage")
        fitness = _parse_fitness(argv[4])
        first, state = algo.ask(state)
        state = algo.tell(state, jnp.array(fitness, dtype=jnp.float32))
        second, _state = algo.ask(state)
        payload = {
            "mode": mode,
            "seed": seed,
            "fitness": fitness,
            "first_batch": _vectors(first),
            "second_batch": _vectors(second),
        }
    elif mode == "complete":
        if len(argv) != 6:
            raise _fail("usage")
        fitness_first = _parse_fitness(argv[4])
        fitness_second = _parse_fitness(argv[5])
        first, state = algo.ask(state)
        state = algo.tell(state, jnp.array(fitness_first, dtype=jnp.float32))
        second, state = algo.ask(state)
        # The second tell is a real state transition; the final population
        # hash is the persisted, replay-verifiable end-of-run evidence.
        state = algo.tell(state, jnp.array(fitness_second, dtype=jnp.float32))
        payload = {
            "mode": mode,
            "seed": seed,
            "fitness_first": fitness_first,
            "fitness_second": fitness_second,
            "first_batch": _vectors(first),
            "second_batch": _vectors(second),
            "final_population_sha256": _population_sha256(jax, state),
        }
    else:
        raise _fail("unknown_mode")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
