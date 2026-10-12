"""REAL native iStratDE state tests (EXP-006).

These execute the vendored iStratDE under the pinned legacy interpreter
(JAX 0.4.33, CPU) via research/comparison/native.py subprocesses — no JAX
in the research venv, no model calls. They prove:

- the first ask differs from the setup population (setup is NOT a batch);
- the first/second batches match the supervisor's independently generated
  offline reference (.runtime/comparison-native-state-reference.json);
- feedback through tell CHANGES the second ask (state is genuinely
  preserved from the first ask into the second within one session);
- seed + fitness replay reproduces both batches bit-for-bit (the approved
  resume mechanism never needs model calls to reconstruct state);
- strict output validation rejects malformed/foreign driver payloads.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from research.comparison import native
from research.comparison.native import (
    NativeDriverError,
    SubprocessNativeProvider,
    _validate_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PYTHON = Path.home() / "rsi-gpu" / ".venv" / "bin" / "python"
REFERENCE = REPO_ROOT / ".runtime" / "comparison-native-state-reference.json"

pytestmark = pytest.mark.skipif(
    not LEGACY_PYTHON.is_file() or shutil.which(str(LEGACY_PYTHON)) is None,
    reason="pinned legacy JAX 0.4.33 venv unavailable",
)


@pytest.fixture(scope="module")
def provider() -> SubprocessNativeProvider:
    return SubprocessNativeProvider(REPO_ROOT)


def _reference() -> dict[int, dict]:
    if not REFERENCE.is_file():
        pytest.skip("supervisor native reference evidence unavailable")
    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert payload["jax"] == "0.4.33" and payload["backend"] == "cpu"
    return {int(entry["seed"]): entry for entry in payload["results"]}


def test_first_batch_matches_supervisor_reference(provider):
    reference = _reference()
    for seed, entry in reference.items():
        batch = provider.first_batch(seed)
        assert [list(row) for row in batch] == entry["first_ask"], seed


def test_setup_population_is_not_first_batch(provider):
    reference = _reference()
    for seed, entry in reference.items():
        setup = provider.setup_population(seed)
        assert [list(row) for row in setup] == entry["setup_population"], seed
        assert [list(row) for row in setup] != entry["first_ask"], seed


def test_feedback_changes_second_batch(provider):
    # Mirrors the supervisor reference property
    # ``second_changes_with_fitness_order``: the SAME four vectors with a
    # reversed fitness ordering produce a different second ask, proving
    # tell state genuinely feeds the next ask.
    seed = 20261010
    fitness_a = [-0.9, -0.1, -0.5, -0.3]
    echo_a, second_a = provider.second_batch(seed, fitness_a)
    assert [list(row) for row in echo_a] == [
        list(row) for row in provider.first_batch(seed)
    ]
    _echo_b, second_b = provider.second_batch(seed, list(reversed(fitness_a)))
    assert second_a != second_b
    # Magnitude-insensitivity is NOT asserted: the reference guarantees
    # ordering sensitivity only.


def test_exact_replay_bit_for_bit(provider):
    seed = 11
    fitness = [-0.75, 0.0, -0.25, -0.5]
    echo1, second1 = provider.second_batch(seed, fitness)
    echo2, second2 = provider.second_batch(seed, fitness)
    assert echo1 == echo2
    assert second1 == second2
    assert [list(row) for row in echo1] == [
        list(row) for row in provider.first_batch(seed)
    ]


def test_pairing_property_same_seed_identical_numbers(provider):
    # RANDOM and ISTRATDE consume the same paired first batch: two provider
    # calls with one seed return byte-identical float values.
    seed = 33
    assert provider.first_batch(seed) == provider.first_batch(seed)


# ------------------------------------------------------- two-tell completion --
def test_complete_run_performs_both_tells_real(provider):
    """The full two-batch feedback loop through the REAL pinned provider."""
    seed = 20261010
    first_fitness = [-0.9, -0.1, -0.5, -0.3]
    second_fitness = [-0.2, -0.8, -0.4, -0.6]
    completion = provider.complete_run(seed, first_fitness, second_fitness)
    # Echoed first batch equals the plain first ask; echoed second batch
    # equals the second-batch ask under the same first fitness.
    assert [list(row) for row in completion.first_batch] == [
        list(row) for row in provider.first_batch(seed)
    ]
    _echo, expected_second = provider.second_batch(seed, first_fitness)
    assert completion.second_batch == expected_second
    assert len(completion.final_population_sha256) == 64
    # Bit-exact replay: identical seed + BOTH fitness histories reproduce
    # the identical final state (the approved resume mechanism).
    again = provider.complete_run(seed, first_fitness, second_fitness)
    assert again == completion


def test_complete_run_final_state_depends_on_second_tell(provider):
    """The second tell is real: second-batch fitness ORDER changes the
    final state while the second batch itself is unchanged (it depends
    only on the first tell)."""
    seed = 77
    first_fitness = [-0.5, -0.4, -0.3, -0.2]
    second_fitness = [-0.6, -0.1, -0.9, -0.2]
    base = provider.complete_run(seed, first_fitness, second_fitness)
    swapped = provider.complete_run(seed, first_fitness, list(reversed(second_fitness)))
    assert base.second_batch == swapped.second_batch
    assert base.final_population_sha256 != swapped.final_population_sha256
    # First-batch fitness order still feeds the second batch as well.
    other = provider.complete_run(seed, list(reversed(first_fitness)), second_fitness)
    assert other.second_batch != base.second_batch


def test_complete_run_validation_rejects_bad_fitness(provider):
    with pytest.raises(NativeDriverError, match="native_fitness_invalid"):
        provider.complete_run(11, [0.1, 0.2, 0.3, 0.4], [0.1, float("inf"), 0.3, 0.4])
    with pytest.raises(NativeDriverError, match="native_fitness_invalid"):
        provider.complete_run(11, [0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4])


def test_complete_run_rejects_tampered_driver_echo(provider, monkeypatch):
    payload = {
        "mode": "complete",
        "seed": 11,
        "fitness_first": [0.1, 0.2, 0.3, 0.4],
        "fitness_second": [0.0, 0.0, 0.0, 0.0],  # tampered echo
        "first_batch": [[0.5, 1.0, 0.4, 0.25, 1.0]] * 4,
        "second_batch": [[0.5, 1.0, 0.4, 0.25, 1.0]] * 4,
        "final_population_sha256": "ab" * 32,
    }

    def fake_run(*a, **k):
        class FakeProc:
            returncode = 0
            stdout = json.dumps(payload)
            stderr = ""

        return FakeProc()

    monkeypatch.setattr(native.subprocess, "run", fake_run)
    with pytest.raises(NativeDriverError, match="native_driver_identity_mismatch"):
        provider.complete_run(11, [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8])

    payload["fitness_second"] = [0.5, 0.6, 0.7, 0.8]
    payload["final_population_sha256"] = "not-a-hash"
    with pytest.raises(NativeDriverError, match="native_final_state_invalid"):
        provider.complete_run(11, [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8])


# ------------------------------------------------------------ validation --
def test_validate_batch_rejects_malformed():
    good = [[0.5, 1.0, 0.4, 0.25, 1.0]] * 4
    assert _validate_batch(good, "x") == ((0.5, 1.0, 0.4, 0.25, 1.0),) * 4
    for bad in (
        [],
        [[0.5, 1.0, 0.4, 0.25, 1.0]] * 3,  # wrong pop
        [[0.5, 1.0, 0.4, 0.25]] * 4,  # wrong dims
        [[0.5, 1.0, 0.4, 0.25, float("nan")]] * 4,
        [[0.5, 1.0, 0.4, 0.25, True]] * 4,
        [["0.5", 1.0, 0.4, 0.25, 1.0]] * 4,
    ):
        with pytest.raises(NativeDriverError):
            _validate_batch(bad, "x")
    with pytest.raises(NativeDriverError, match="native_vector_out_of_bounds"):
        _validate_batch([[0.5, 1.0, 0.4, 0.25, 2.5]] * 4, "x")
    with pytest.raises(NativeDriverError, match="native_vector_out_of_bounds"):
        _validate_batch([[0.01, 1.0, 0.4, 0.25, 1.0]] * 4, "x")


def test_driver_identity_mismatch_rejected(provider, monkeypatch):
    payload = {"mode": "setup", "seed": 999, "population": [[0.5] * 5] * 4}

    class FakeProc:
        returncode = 0
        stdout = json.dumps(payload)
        stderr = ""

    monkeypatch.setattr(native.subprocess, "run", lambda *a, **k: FakeProc())
    with pytest.raises(NativeDriverError, match="native_driver_identity_mismatch"):
        provider.first_batch(11)


def test_driver_nonzero_exit_and_silence_rejected(provider, monkeypatch):
    class FailProc:
        returncode = 2
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(native.subprocess, "run", lambda *a, **k: FailProc())
    with pytest.raises(NativeDriverError, match="native_driver_failed"):
        provider.first_batch(11)

    class SilentProc:
        returncode = 0
        stdout = "\n"
        stderr = ""

    monkeypatch.setattr(native.subprocess, "run", lambda *a, **k: SilentProc())
    with pytest.raises(NativeDriverError, match="native_driver_silent"):
        provider.first_batch(11)


def test_fitness_validation_rejects_nonfinite(provider):
    with pytest.raises(NativeDriverError, match="native_fitness_invalid"):
        provider.second_batch(11, [0.1, float("nan"), 0.2, 0.3])
    with pytest.raises(NativeDriverError, match="native_fitness_invalid"):
        provider.second_batch(11, [0.1, 0.2, 0.3])


def test_native_env_pin_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(
        native, "NATIVE_ENV_FILE", Path("missing") / "native-env.json"
    )
    with pytest.raises(NativeDriverError, match="native_env_missing"):
        native.load_native_python(tmp_path)
