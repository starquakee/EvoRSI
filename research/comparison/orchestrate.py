"""Top-level comparison orchestration (EXP-006).

Drives the frozen 216-position schedule against the ONE comparison ledger,
checkpointing ONLY at verified completed-config boundaries:

- before launch, a durable intent binds the COMPLETE serialized
  ComparisonRunSpec bytes (including job_timeout_seconds and the runner
  commit), not just the config identity;
- a config interrupted between intent and boundary is NEVER silently
  rerun, replaced or re-seeded — resume fails closed on it;
- every child failure (pre-ledger, nonzero exit, missing/tampered result,
  identity mismatch) durably stops the original ledger before raising;
- a truthfully cell-capped cell freezes its verified incumbent and the
  schedule continues for OTHER cells; a global stop is permanent;
- when all eight configs of a cell are terminal, the cell freezes exactly
  one winner: highest validation score, tie = earliest actual evaluation
  order (== lowest config index, guaranteed by the frozen schedule), with
  the exact winning program bytes recovered from the run journal and
  re-hashed against the incumbent binding;
- ``verify_final_barrier`` validates the full 27-cell roster and actual
  terminal evidence (re-hashing spec/result/winner bytes) — never a
  caller-supplied count or boolean.

No model calls, no live ledger creation and no final-data release happen
here; the live CLI entry stays fail-closed until the EXP-007 supervisor
review is wired.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from research.comparison.budget import ComparisonLedger, comparison_experiment_identity
from research.comparison.deploy import validate_job_snapshot
from research.comparison.methods import (
    feedback_theta_plan,
    initial_theta_plan,
    validate_encoded_vector,
)
from research.comparison.native import NativeProvider
from research.comparison.protocol import (
    CONFIGS_PER_CELL,
    ComparisonSpecError,
    canonical_json,
    decode_theta,
    identity_of,
    theta_gate,
)
from research.comparison.runner import (
    LEDGER_FILENAME,
    LIVE_JOBS_SCHEMA,
    RUN_RESULT_SCHEMA,
    RUNTIME_DIR,
    ComparisonRunSpec,
    comparison_config_identity,
    run_dir_name,
)
from research.comparison.schedule import (
    INITIAL_BATCH_CONFIGS,
    JOB_TIMEOUT_SECONDS,
    SchedulePosition,
    derive_native_seed,
    freeze_schedule,
    load_schedule,
    schedule_identity,
)
from research.comparison.specs import CellSpec, InnerRunSpec, iter_cells
from research.contracts.source_gate import load_policy
from research.search.sandbox_task import _single_file_tree_sha256

ORCHESTRATION_STATE_SCHEMA = "istratde-comparison-cell.v1"
INTENT_SCHEMA = "istratde-comparison-intent.v1"
BARRIER_SCHEMA = "istratde-comparison-barrier.v1"

_SPEC_DIR = "specs"
_INTENT_DIR = "intents"
_CELL_DIR = "cells"
_RUNS_DIR = "runs"
_WINNER_FILENAME = "winner.py"


class OrchestrationError(RuntimeError):
    """An orchestration/state integrity constraint was violated."""


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_durable(path: Path, text: str) -> None:
    """Atomic tmp+replace write with file AND parent-directory fsync.

    Never follows a pre-existing temporary symlink (a planted alias must
    not redirect the write onto unrelated evidence bytes).
    """
    tmp = path.with_name(path.name + ".tmp")
    if tmp.is_symlink():
        raise OrchestrationError(f"durable_write_tmp_symlink:{path.name}")
    if tmp.exists():
        tmp.unlink()  # regular crash leftover only
    with open(tmp, "x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _require_hex64(value: Any, rule: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise OrchestrationError(rule)
    return value


def _read_json_file(path: Path, rule: str) -> dict[str, Any]:
    if path.is_symlink() or not path.exists():
        raise OrchestrationError(rule)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OrchestrationError(rule) from exc
    if not isinstance(payload, dict):
        raise OrchestrationError(rule)
    return payload


#: Child-runner contract: given the frozen spec path, run one inner search
#: (production: ``python -m research.comparison.runner run --spec ...``
#: under the Evo venv) and return its exit code.
ChildRunner = Callable[[Path], int]


def evo_child_runner(
    repo_root: Path, *, deadline_fn: Callable[[], float | None] | None = None
) -> ChildRunner:
    """The production child: the Evo-venv interpreter running run_inner.

    ``deadline_fn`` returns the ORIGINAL experiment deadline epoch (from
    the ledger); when given, every child invocation is bounded by the
    remaining time — an expired deadline refuses BEFORE spawning, and a
    child exceeding the remainder is killed and reported as an
    infrastructure failure (the orchestrator stops the ledger).
    """
    import subprocess
    import sys

    python = repo_root / "OpenMLE-Evo" / ".venv" / "bin" / "python"
    interpreter = python if python.is_file() else Path(sys.executable)

    def _run(spec_path: Path) -> int:
        timeout: float | None = None
        if deadline_fn is not None:
            deadline = deadline_fn()
            if deadline is not None:
                timeout = float(deadline) - time.time()
                if timeout <= 0:
                    raise OrchestrationError("child_deadline_exceeded")
        try:
            proc = subprocess.run(
                [
                    str(interpreter),
                    "-m",
                    "research.comparison.runner",
                    "run",
                    "--spec",
                    str(spec_path),
                    "--repo-root",
                    str(repo_root),
                ],
                cwd=str(repo_root),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise OrchestrationError("child_deadline_exceeded") from exc
        return proc.returncode

    return _run


def _validate_runtime_paths(runtime_dir: Path, repo_root: Path) -> None:
    """Runtime/output ancestors must not escape via symlinks or aliases."""
    if runtime_dir.is_symlink() or repo_root.is_symlink():
        raise OrchestrationError("runtime_path_symlinked")
    ancestor = runtime_dir
    while ancestor != ancestor.parent:
        if ancestor.exists() and ancestor.is_symlink():
            raise OrchestrationError("runtime_path_symlinked")
        if ancestor == repo_root:
            break
        ancestor = ancestor.parent
    resolved_repo = repo_root.resolve()
    resolved_runtime = runtime_dir.resolve()
    if not resolved_runtime.is_relative_to(resolved_repo):
        raise OrchestrationError("runtime_dir_escapes_repo")


class Orchestrator:
    """Stateful driver of the frozen schedule over the one ledger."""

    def __init__(
        self,
        *,
        runtime_dir: Path,
        repo_root: Path,
        cells: Sequence[CellSpec],
        positions: Sequence[SchedulePosition],
        provider: NativeProvider,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.runtime_dir = runtime_dir
        self.repo_root = repo_root
        self.cells = tuple(cells)
        self.positions = tuple(positions)
        self.provider = provider
        self._now = now_fn
        _validate_runtime_paths(runtime_dir, repo_root)
        if len(self.cells) != 27 or len(self.positions) != 216:
            raise OrchestrationError("orchestration_roster_invalid")
        self._cells_by_key = {cell.cell_key(): cell for cell in self.cells}
        for pos in self.positions:
            if pos.cell_key not in self._cells_by_key:
                raise OrchestrationError("orchestration_roster_invalid")
        identities = {cell.identity() for cell in self.cells}
        if len(identities) != 27:
            raise OrchestrationError("orchestration_roster_invalid")
        commits = {cell.source_commit for cell in self.cells}
        if len(commits) != 1:
            raise OrchestrationError("source_commit_not_uniform")
        self.source_commit = commits.pop()
        self._schedule_identity = schedule_identity(self.positions, self.source_commit)

    # ------------------------------------------------------------ paths --
    def _cell_dir(self, cell_key: str) -> Path:
        return self.runtime_dir / _CELL_DIR / cell_key.replace(":", "-")

    def _cell_state_path(self, cell_key: str) -> Path:
        return self._cell_dir(cell_key) / "cell.json"

    def _spec_path(self, run: InnerRunSpec) -> Path:
        return self.runtime_dir / _SPEC_DIR / f"{run_dir_name(run)}.json"

    def _intent_path(self, run: InnerRunSpec) -> Path:
        return self.runtime_dir / _INTENT_DIR / f"{run_dir_name(run)}.json"

    def _run_dir(self, run: InnerRunSpec) -> Path:
        return self.runtime_dir / _RUNS_DIR / run_dir_name(run)

    def _result_path(self, run: InnerRunSpec) -> Path:
        return self._run_dir(run) / "run-result.json"

    # --------------------------------------------------------- lifecycle --
    @classmethod
    def initialize(
        cls,
        *,
        runtime_dir: Path,
        repo_root: Path,
        cells: Sequence[CellSpec],
        provider: NativeProvider,
        now_fn: Callable[[], float] = time.time,
    ) -> "Orchestrator":
        """Create fresh orchestration state: frozen schedule + cell states.

        Fails closed if ANY state already exists (an experiment is never
        silently reinitialized). All FIXED/RANDOM vectors and the paired
        ISTRATDE initial batches are accepted and persisted now; ISTRATDE
        feedback vectors become known only after first-batch evaluation.
        """
        _validate_runtime_paths(runtime_dir, repo_root)
        if runtime_dir.exists() and any(runtime_dir.iterdir()):
            raise OrchestrationError("orchestration_dir_not_fresh")
        for sub in (_SPEC_DIR, _INTENT_DIR, _CELL_DIR, _RUNS_DIR):
            (runtime_dir / sub).mkdir(parents=True, exist_ok=True)
        positions = freeze_schedule(runtime_dir / "schedule.json", cells)
        orch = cls(
            runtime_dir=runtime_dir,
            repo_root=repo_root,
            cells=cells,
            positions=positions,
            provider=provider,
            now_fn=now_fn,
        )
        for cell in orch.cells:
            vectors: dict[int, tuple[float, ...]] = dict(
                initial_theta_plan(cell, provider)
            )
            if cell.method != "ISTRATDE":
                # FIXED and RANDOM never consume scores: their feedback-half
                # vectors are knowable (and frozen) NOW.
                vectors.update(feedback_theta_plan(cell, provider, (), ()))
            state = {
                "schema": ORCHESTRATION_STATE_SCHEMA,
                "cell_key": cell.cell_key(),
                "cell_identity": cell.identity(),
                "task_id": cell.task_id,
                "method": cell.method,
                "search_seed": cell.search_seed,
                "schedule_identity": orch._schedule_identity,
                "vectors": {
                    str(index): [float(v) for v in vector]
                    for index, vector in sorted(vectors.items())
                },
                "configs": {},
                "winner": None,
                "terminal": None,
                "native_completion": None,
            }
            orch._persist_cell_state(cell.cell_key(), state)
        return orch

    @classmethod
    def load(
        cls,
        *,
        runtime_dir: Path,
        repo_root: Path,
        bindings: Mapping[str, Any],
        provider: NativeProvider,
        now_fn: Callable[[], float] = time.time,
    ) -> "Orchestrator":
        """Resume: reload and RE-VALIDATE every persisted artifact.

        The schedule is rebuilt from the roster and compared byte-for-byte;
        every cell state is re-checked (schema, identities, vector
        identities, completed-config spec/result hashes, winner bytes,
        terminal consistency); a launched-but-uncompleted config is an
        interrupted config and fails closed.
        """
        _validate_runtime_paths(runtime_dir, repo_root)
        schedule_path = runtime_dir / "schedule.json"
        # The frozen schedule binds the original source commit; resume with
        # anything else is impossible by construction.
        schedule_payload = _read_json_file(schedule_path, "schedule_missing")
        source_commit = schedule_payload.get("source_commit")
        if not isinstance(source_commit, str):
            raise OrchestrationError("schedule_source_commit_invalid")
        cells = iter_cells(bindings, source_commit)
        positions = load_schedule(schedule_path, cells)
        orch = cls(
            runtime_dir=runtime_dir,
            repo_root=repo_root,
            cells=cells,
            positions=positions,
            provider=provider,
            now_fn=now_fn,
        )
        for cell in orch.cells:
            state = orch._load_cell_state(cell.cell_key())
            orch._validate_cell_state(cell, state)
        orch._validate_no_orphans()
        return orch

    def _validate_no_orphans(self) -> None:
        """Every intent/spec/run artifact must belong to a recorded config.

        A durable intent without a matching cell launch record means the
        process died between the intent and the boundary checkpoint; that
        config is interrupted and resume fails closed rather than silently
        rerunning it.
        """
        expected: dict[str, tuple[str, int]] = {}
        for cell in self.cells:
            for index in range(CONFIGS_PER_CELL):
                name = run_dir_name(cell.inner_run(index))
                expected[name] = (cell.cell_key(), index)
        for subdir in (_INTENT_DIR, _SPEC_DIR):
            directory = self.runtime_dir / subdir
            if not directory.is_dir():
                continue
            for entry in sorted(directory.iterdir()):
                if entry.suffix == ".tmp":
                    raise OrchestrationError(f"orphan_artifact:{entry.name}")
                if entry.name.endswith(".json") is False or entry.is_symlink():
                    raise OrchestrationError(f"orphan_artifact:{entry.name}")
                stem = entry.stem
                if stem not in expected:
                    raise OrchestrationError(f"orphan_artifact:{entry.name}")
                cell_key, index = expected[stem]
                state = self._cell_state(cell_key)
                configs = state.get("configs")
                record = (
                    configs.get(str(index)) if isinstance(configs, dict) else None
                )
                if not isinstance(record, dict) or record.get("status") != "completed":
                    raise OrchestrationError(
                        f"orphan_artifact:{cell_key}:{index}"
                    )
                if subdir == _INTENT_DIR:
                    try:
                        text = entry.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError) as exc:
                        raise OrchestrationError(
                            f"intent_unreadable:{cell_key}:{index}"
                        ) from exc
                    if _sha256_text(text) != record.get("intent_sha256"):
                        raise OrchestrationError(
                            f"intent_tampered:{cell_key}:{index}"
                        )
        runs_dir = self.runtime_dir / _RUNS_DIR
        if runs_dir.is_dir():
            for entry in sorted(runs_dir.iterdir()):
                if entry.name not in expected:
                    raise OrchestrationError(f"orphan_artifact:{entry.name}")
                cell_key, index = expected[entry.name]
                state = self._cell_state(cell_key)
                configs = state.get("configs")
                record = (
                    configs.get(str(index)) if isinstance(configs, dict) else None
                )
                if not isinstance(record, dict) or record.get("status") != "completed":
                    raise OrchestrationError(
                        f"orphan_artifact:{cell_key}:{index}"
                    )

    # ------------------------------------------------------- cell state --
    def _persist_cell_state(self, cell_key: str, state: Mapping[str, Any]) -> None:
        path = self._cell_state_path(cell_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_durable(path, canonical_json(dict(state)) + "\n")

    def _load_cell_state(self, cell_key: str) -> dict[str, Any]:
        return _read_json_file(self._cell_state_path(cell_key), "cell_state_missing")

    def _validate_cell_state(
        self,
        cell: CellSpec,
        state: Mapping[str, Any],
        ledger_view: Mapping[str, Any] | None = None,
    ) -> None:
        cell_key = cell.cell_key()
        if state.get("schema") != ORCHESTRATION_STATE_SCHEMA:
            raise OrchestrationError("cell_state_schema_mismatch")
        if (
            state.get("cell_key") != cell_key
            or state.get("cell_identity") != cell.identity()
            or state.get("task_id") != cell.task_id
            or state.get("method") != cell.method
            or state.get("search_seed") != cell.search_seed
            or state.get("schedule_identity") != self._schedule_identity
        ):
            raise OrchestrationError(f"cell_state_identity_mismatch:{cell_key}")
        configs = state.get("configs")
        if not isinstance(configs, dict):
            raise OrchestrationError(f"cell_state_configs_invalid:{cell_key}")
        records: dict[int, Mapping[str, Any]] = {}
        for key, record in configs.items():
            index = self._validate_config_record(cell, key, record, ledger_view)
            if index in records:
                raise OrchestrationError(f"cell_state_configs_invalid:{cell_key}")
            records[index] = record
        if ledger_view is not None and any(
            record.get("status") == "not_launched_cell_exhausted"
            for record in records.values()
        ):
            # A skipped suffix is accepted ONLY with the ORIGINAL ledger's
            # durable cell termination record (truthful cap, partial counts
            # retained per run); never manufactured.
            cells_view = ledger_view.get("cells")
            cell_view = (
                cells_view.get(cell_key) if isinstance(cells_view, Mapping) else None
            )
            if not isinstance(cell_view, Mapping) or cell_view.get("terminated") is None:
                raise OrchestrationError(f"cell_cap_claim_unproven:{cell_key}")
        terminal = state.get("terminal")
        winner = state.get("winner")
        if terminal is not None:
            if not isinstance(terminal, dict) or terminal.get("status") not in (
                "frozen",
                "failed_no_valid_program",
            ):
                raise OrchestrationError(f"cell_state_terminal_invalid:{cell_key}")
            # A terminal cell must account for EVERY config slot with real
            # boundary evidence; a bare caller-written flag never suffices.
            if set(records) != set(range(CONFIGS_PER_CELL)):
                raise OrchestrationError(f"cell_state_terminal_unproven:{cell_key}")
            if terminal["status"] == "frozen":
                if not any(
                    record.get("status") == "completed"
                    and record.get("incumbent") is not None
                    for record in records.values()
                ):
                    raise OrchestrationError(
                        f"cell_state_terminal_unproven:{cell_key}"
                    )
                self._validate_winner_record(cell, state, winner)
            elif winner is not None:
                raise OrchestrationError(f"cell_state_terminal_invalid:{cell_key}")
        elif winner is not None:
            raise OrchestrationError(f"cell_state_winner_premature:{cell_key}")
        self._validate_vectors_against_plan(cell, state, records)
        completion = state.get("native_completion")
        fully_resolved = set(records) == set(range(CONFIGS_PER_CELL)) and all(
            self._record_evaluated(record) for record in records.values()
        )
        if completion is not None:
            if cell.method != "ISTRATDE":
                raise OrchestrationError(f"native_completion_invalid:{cell_key}")
            self._validate_native_completion(cell, state, records, completion)
        elif cell.method == "ISTRATDE" and terminal is not None and fully_resolved:
            # A fully evaluated iStratDE cell MUST carry the two-tell
            # completion evidence; its absence is tampering/incompleteness.
            raise OrchestrationError(f"native_completion_missing:{cell_key}")

    @staticmethod
    def _record_evaluated(record: Mapping[str, Any]) -> bool:
        """Real-evaluation eligibility for native feedback fitness.

        A genuinely gate-rejected vector is a truthful known-zero outcome; a
        completed config counts ONLY when its run actually sent at least one
        main request (the persisted flag, re-derived from the verified result
        and the ORIGINAL ledger on every audit). A valid theta whose first
        main reservation was refused by the cell cap was NEVER evaluated and
        must never contribute an invented zero fitness.
        """
        status = record.get("status")
        if status == "gate_rejected":
            return True
        if status == "completed":
            return record.get("evaluated") is True
        return False

    def _native_completion_record(
        self, cell: CellSpec, state: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Replay BOTH accepted fitness histories through the SAME provider.

        The second tell is a real algorithm state transition; the returned
        record persists the accepted fitness histories and the bit-exact
        final-population hash. Config slots without a real resolved outcome
        never contribute invented fitness: this is only called when all
        eight configs are completed or truthfully gate-rejected.
        """
        cell_key = cell.cell_key()
        configs = state.get("configs")
        vectors = state.get("vectors")
        if not isinstance(configs, dict) or not isinstance(vectors, dict):
            raise OrchestrationError(f"native_completion_unavailable:{cell_key}")
        utilities: list[float] = []
        accepted: list[list[float]] = []
        for index in range(CONFIGS_PER_CELL):
            record = configs.get(str(index))
            vector = vectors.get(str(index))
            if (
                not isinstance(record, dict)
                or not self._record_evaluated(record)
                or not isinstance(vector, list)
            ):
                raise OrchestrationError(f"native_completion_unavailable:{cell_key}")
            utilities.append(float(record["utility"]))
            accepted.append([float(v) for v in vector])
        first_fitness = [-u for u in utilities[:INITIAL_BATCH_CONFIGS]]
        second_fitness = [-u for u in utilities[INITIAL_BATCH_CONFIGS:]]
        seed = derive_native_seed(cell.task_id, cell.search_seed)
        try:
            completion = self.provider.complete_run(
                seed, first_fitness, second_fitness
            )
        except Exception as exc:
            raise OrchestrationError(
                f"native_completion_failed:{cell_key}"
            ) from exc
        # The replayed batches must equal the vectors actually evaluated.
        if [list(map(float, row)) for row in completion.first_batch] != (
            accepted[:INITIAL_BATCH_CONFIGS]
        ) or [list(map(float, row)) for row in completion.second_batch] != (
            accepted[INITIAL_BATCH_CONFIGS:]
        ):
            raise OrchestrationError(f"native_replay_mismatch:{cell_key}")
        return {
            "seed": seed,
            "first_fitness": first_fitness,
            "second_fitness": second_fitness,
            "final_population_sha256": completion.final_population_sha256,
        }

    def _validate_native_completion(
        self,
        cell: CellSpec,
        state: Mapping[str, Any],
        records: Mapping[int, Mapping[str, Any]],
        completion: Any,
    ) -> None:
        """Persisted two-tell evidence must reproduce EXACTLY on replay."""
        cell_key = cell.cell_key()
        if not isinstance(completion, dict):
            raise OrchestrationError(f"native_completion_invalid:{cell_key}")
        seed = derive_native_seed(cell.task_id, cell.search_seed)
        final_hash = completion.get("final_population_sha256")
        if (
            completion.get("seed") != seed
            or not isinstance(final_hash, str)
            or len(final_hash) != 64
            or any(c not in "0123456789abcdef" for c in final_hash)
        ):
            raise OrchestrationError(f"native_completion_invalid:{cell_key}")
        utilities: list[float] = []
        for index in range(CONFIGS_PER_CELL):
            record = records.get(index)
            if record is None or not self._record_evaluated(record):
                raise OrchestrationError(f"native_completion_invalid:{cell_key}")
            utilities.append(float(record["utility"]))
        expected_first = [-u for u in utilities[:INITIAL_BATCH_CONFIGS]]
        expected_second = [-u for u in utilities[INITIAL_BATCH_CONFIGS:]]
        if completion.get("first_fitness") != expected_first or completion.get(
            "second_fitness"
        ) != expected_second:
            raise OrchestrationError(f"native_completion_mismatch:{cell_key}")
        fresh = self._native_completion_record(cell, state)
        if fresh["final_population_sha256"] != final_hash:
            raise OrchestrationError(f"native_completion_mismatch:{cell_key}")

    def _validate_vectors_against_plan(
        self,
        cell: CellSpec,
        state: Mapping[str, Any],
        records: Mapping[int, Mapping[str, Any]],
    ) -> None:
        """Persisted vectors must reproduce the deterministic frozen plan.

        FIXED/RANDOM vectors are fully recomputed; the ISTRATDE initial
        batch is recomputed from the frozen native seed, and its feedback
        batch (once present) is recomputed from the frozen seed plus the
        recorded first-batch utilities — tampering with either fails closed.
        """
        cell_key = cell.cell_key()
        vectors = state.get("vectors")
        if not isinstance(vectors, dict):
            raise OrchestrationError(f"cell_state_vectors_invalid:{cell_key}")
        persisted: dict[int, tuple[float, ...]] = {}
        for key, vector in vectors.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                raise OrchestrationError(
                    f"cell_state_vectors_invalid:{cell_key}"
                ) from None
            if not isinstance(vector, list):
                raise OrchestrationError(f"cell_state_vectors_invalid:{cell_key}")
            try:
                persisted[index] = validate_encoded_vector(vector)
            except ComparisonSpecError as exc:
                raise OrchestrationError(
                    f"cell_state_vectors_invalid:{cell_key}"
                ) from exc
        expected: dict[int, tuple[float, ...]] = dict(
            initial_theta_plan(cell, self.provider)
        )
        if cell.method != "ISTRATDE":
            expected.update(feedback_theta_plan(cell, self.provider, (), ()))
        elif any(index >= INITIAL_BATCH_CONFIGS for index in persisted):
            utilities: list[float] = []
            accepted_first: list[list[float]] = []
            for index in range(INITIAL_BATCH_CONFIGS):
                record = records.get(index)
                if (
                    record is None
                    or record.get("status") not in ("completed", "gate_rejected")
                    or index not in persisted
                ):
                    raise OrchestrationError(
                        f"cell_state_vectors_invalid:{cell_key}"
                    )
                utilities.append(float(record["utility"]))
                accepted_first.append([float(v) for v in persisted[index]])
            expected.update(
                feedback_theta_plan(cell, self.provider, utilities, accepted_first)
            )
        if set(persisted) != set(expected) or any(
            tuple(float(v) for v in persisted[index])
            != tuple(float(v) for v in expected[index])
            for index in expected
        ):
            raise OrchestrationError(f"cell_state_vectors_mismatch:{cell_key}")
        # A completed config's accepted vector must still be recorded.
        for index, record in records.items():
            if record.get("status") == "completed" and index not in persisted:
                raise OrchestrationError(f"cell_state_vector_missing:{cell_key}")
            if record.get("status") == "gate_rejected":
                # The gate claim must still hold for the persisted vector.
                if index not in persisted:
                    raise OrchestrationError(
                        f"cell_state_vector_missing:{cell_key}"
                    )
                try:
                    theta_gate(decode_theta(persisted[index]))
                except ComparisonSpecError:
                    pass
                else:
                    raise OrchestrationError(
                        f"cell_state_gate_claim_false:{cell_key}"
                    ) from None

    def _validate_config_record(
        self,
        cell: CellSpec,
        key: str,
        record: Any,
        ledger_view: Mapping[str, Any] | None = None,
    ) -> int:
        cell_key = cell.cell_key()
        if not isinstance(record, dict):
            raise OrchestrationError(f"cell_state_configs_invalid:{cell_key}")
        try:
            index = int(key)
        except (TypeError, ValueError):
            raise OrchestrationError(
                f"cell_state_configs_invalid:{cell_key}"
            ) from None
        if not 0 <= index < CONFIGS_PER_CELL or record.get("config_index") != index:
            raise OrchestrationError(f"cell_state_configs_invalid:{cell_key}")
        status = record.get("status")
        run = cell.inner_run(index)
        if status == "launched":
            # Interrupted config: never silently rerun/replaced/re-seeded.
            raise OrchestrationError(f"config_interrupted:{run.run_key()}")
        if status not in ("completed", "not_launched_cell_exhausted", "gate_rejected"):
            raise OrchestrationError(f"cell_state_configs_invalid:{cell_key}")
        position = record.get("position")
        expected_position = next(
            p.position
            for p in self.positions
            if p.cell_key == cell_key and p.config_index == index
        )
        if position != expected_position:
            raise OrchestrationError(f"cell_state_position_mismatch:{cell_key}")
        if status == "not_launched_cell_exhausted":
            self._require_zero_run_requests(cell, run, ledger_view)
            return index
        if status == "gate_rejected":
            # A truthful no-launch boundary: utility 0, no artifacts; the
            # gate claim itself is re-verified against the persisted vector
            # in _validate_vectors_against_plan.
            if (
                record.get("utility") != 0.0
                or record.get("raw_score") is not None
                or record.get("incumbent") is not None
                or not isinstance(record.get("reason"), str)
            ):
                raise OrchestrationError(f"cell_state_configs_invalid:{cell_key}")
            self._require_zero_run_requests(cell, run, ledger_view)
            return index
        spec_path = self._spec_path(run)
        result_path = self._result_path(run)
        _require_hex64(record.get("spec_sha256"), f"cell_state_configs_invalid:{cell_key}")
        _require_hex64(
            record.get("result_sha256"), f"cell_state_configs_invalid:{cell_key}"
        )
        try:
            spec_bytes = spec_path.read_bytes()
            result_bytes = result_path.read_bytes()
        except OSError as exc:
            raise OrchestrationError(f"config_artifacts_missing:{run.run_key()}") from exc
        if spec_path.is_symlink() or result_path.is_symlink():
            raise OrchestrationError(f"config_artifacts_missing:{run.run_key()}")
        if _sha256_bytes(spec_bytes) != record["spec_sha256"]:
            raise OrchestrationError(f"config_spec_tampered:{run.run_key()}")
        if _sha256_bytes(result_bytes) != record["result_sha256"]:
            raise OrchestrationError(f"config_result_tampered:{run.run_key()}")
        utility = record.get("utility")
        if (
            isinstance(utility, bool)
            or not isinstance(utility, (int, float))
            or not math.isfinite(float(utility))
            or not 0.0 <= float(utility) <= 1.0
        ):
            raise OrchestrationError(f"cell_state_utility_invalid:{cell_key}")
        raw_score = record.get("raw_score")
        if raw_score is not None:
            if (
                isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(float(raw_score))
                or not 0.0 <= float(raw_score) <= 1.0
                or float(raw_score) != float(utility)
            ):
                raise OrchestrationError(f"cell_state_utility_invalid:{cell_key}")
        # Full run-evidence re-verification (shared by completion, resume
        # audit and the final barrier): decoded spec binds the persisted
        # method vector, request/trace/journal/live-job artifacts re-hash,
        # and claims reconcile with the ORIGINAL per-run ledger records.
        try:
            result = json.loads(result_bytes)
        except ValueError as exc:
            raise OrchestrationError(f"config_result_tampered:{run.run_key()}") from exc
        main_claimed = self._verify_result_evidence(cell, run, record, result, ledger_view)
        # The persisted real-evaluation flag must reproduce from the verified
        # evidence (never trusted as a bare claim).
        if record.get("evaluated") is not (main_claimed > 0):
            raise OrchestrationError(f"cell_state_configs_invalid:{cell_key}")
        return index

    # ---------------------------------------------------- run evidence --
    def _require_zero_run_requests(
        self,
        cell: CellSpec,
        run: InnerRunSpec,
        ledger_view: Mapping[str, Any] | None,
    ) -> None:
        """A no-launch record (gate_rejected / skipped) must have ZERO
        original ledger requests or reservations for its run."""
        if ledger_view is None:
            return
        run_identity = run.identity()
        for section in ("requests", "reservations"):
            entries = ledger_view.get(section)
            if not isinstance(entries, list):
                raise OrchestrationError(f"ledger_view_invalid:{cell.cell_key()}")
            if any(
                isinstance(entry, dict)
                and entry.get("run_identity") == run_identity
                for entry in entries
            ):
                raise OrchestrationError(f"run_requests_unexpected:{run.run_key()}")

    @staticmethod
    def _strict_nonneg_int(value: Any) -> bool:
        return type(value) is int and value >= 0

    def _verify_result_evidence(
        self,
        cell: CellSpec,
        run: InnerRunSpec,
        record: Mapping[str, Any],
        result: Mapping[str, Any],
        ledger_view: Mapping[str, Any] | None,
    ) -> int:
        """Complete per-config evidence binding for a completed boundary.

        Verifies: the decoded spec reproduces the persisted method vector,
        the paired inner seed, exact request counts against the ORIGINAL
        ledger (ordinary completion = SIX main attempts plus an actual
        0..2 debug total), request/generation/operator/journal trace bytes
        re-hashed, the durable live-jobs record fully resolved, the runner's
        trusted dataset/policy pins against the frozen task binding and the
        pinned source policy, every claimed prediction snapshot's bytes and
        hash, and EVERY actual job record — never-submitted local failures
        (job_id None) stay null with no phantom job, while every executed
        job's source-gate/evaluator/answer/registry/row-count/snapshot
        provenance is bound to pinned values and any explicit
        source-identity/cleanup contradiction fails closed. Returns the
        verified main-request count (the real-evaluation signal). Fail
        closed on any contradiction; never convert corrupt evidence into
        an ordinary zero-utility outcome.
        """
        cell_key = cell.cell_key()
        run_key = run.run_key()

        def _bad(reason: str) -> OrchestrationError:
            return OrchestrationError(f"{reason}:{run_key}")

        config_index = record.get("config_index")
        if isinstance(config_index, bool) or not isinstance(config_index, int):
            raise _bad("cell_state_configs_invalid")
        vector = self._vector_for(cell, config_index)
        if result.get("decoded") != decode_theta(vector):
            raise _bad("result_decoded_mismatch")
        if result.get("inner_seed") != run.inner_seed:
            raise _bad("result_seed_mismatch")
        termination = result.get("termination")
        if not isinstance(termination, str):
            raise _bad("result_not_terminal")
        requests = result.get("requests")
        if (
            not isinstance(requests, dict)
            or not self._strict_nonneg_int(requests.get("main"))
            or not self._strict_nonneg_int(requests.get("debug"))
            or not self._strict_nonneg_int(requests.get("charged_tokens"))
        ):
            raise _bad("result_requests_invalid")
        main_claimed = int(requests["main"])
        debug_claimed = int(requests["debug"])
        if debug_claimed > 2:
            raise _bad("result_requests_invalid")
        result_status = result.get("status")
        if termination == "completed":
            if result_status != "completed":
                raise _bad("result_not_terminal")
            # Ordinary completion is exactly the frozen inner protocol:
            # THREE generations x TWO main attempts (plus 0..2 debug total).
            if main_claimed != 6:
                raise _bad("run_main_count_mismatch")
        elif termination.startswith("CellBudgetExhausted"):
            if result_status != "cell_terminated" or main_claimed > 6:
                raise _bad("run_main_count_mismatch")
        else:
            raise _bad("result_not_terminal")
        # Reconcile with ORIGINAL per-run ledger records.
        ledger_kinds: dict[str, str] | None = None
        if ledger_view is not None:
            all_requests = ledger_view.get("requests")
            if not isinstance(all_requests, list):
                raise _bad("ledger_view_invalid")
            ledger_records = [
                entry
                for entry in all_requests
                if isinstance(entry, dict)
                and entry.get("run_identity") == run.identity()
            ]
            actual_main = sum(1 for r in ledger_records if r["kind"] == "main")
            actual_debug = sum(1 for r in ledger_records if r["kind"] == "debug")
            if (actual_main, actual_debug) != (main_claimed, debug_claimed):
                raise _bad("ledger_run_requests_mismatch")
            if sum(int(r["charged_tokens"]) for r in ledger_records) != int(
                requests["charged_tokens"]
            ):
                raise _bad("ledger_run_tokens_mismatch")
            ledger_kinds = {
                str(r["request_id"]): str(r["kind"]) for r in ledger_records
            }
        # Trace/journal artifacts re-hash against the pinned result bytes.
        run_dir = self._run_dir(run)
        for field, name in (
            ("journal_sha256", "journal.jsonl"),
            ("operator_trace_sha256", "operator-trace.jsonl"),
            ("request_trace_sha256", "request-trace.jsonl"),
            ("generation_outcomes_sha256", "generation-outcomes.jsonl"),
        ):
            claimed = result.get(field)
            path = run_dir / name
            if claimed == "" and not path.exists() and not path.is_symlink():
                continue  # no such artifact legitimately produced
            _require_hex64(claimed, f"config_evidence_invalid:{run_key}")
            if path.is_symlink() or not path.exists():
                raise _bad("trace_missing")
            try:
                actual_bytes = path.read_bytes()
            except OSError as exc:
                raise _bad("trace_missing") from exc
            if _sha256_bytes(actual_bytes) != claimed:
                raise _bad("trace_tampered")
        # Request trace: every line binds to an ORIGINAL ledger request id
        # with the matching kind; counts equal the claimed totals.
        total_claimed = main_claimed + debug_claimed
        if total_claimed > 0:
            trace_path = run_dir / "request-trace.jsonl"
            try:
                lines = [
                    line
                    for line in trace_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, UnicodeDecodeError) as exc:
                raise _bad("trace_missing") from exc
            if len(lines) != total_claimed:
                raise _bad("request_trace_count_mismatch")
            seen_ids: set[str] = set()
            for line in lines:
                try:
                    entry = json.loads(line)
                except ValueError as exc:
                    raise _bad("request_trace_malformed") from exc
                request_id = entry.get("request_id") if isinstance(entry, dict) else None
                kind = entry.get("kind") if isinstance(entry, dict) else None
                if (
                    not isinstance(entry, dict)
                    or entry.get("run_key") != run_key
                    or not isinstance(request_id, str)
                    or request_id in seen_ids
                    or kind not in ("main", "debug")
                ):
                    raise _bad("request_trace_malformed")
                if ledger_kinds is not None and ledger_kinds.get(request_id) != kind:
                    raise _bad("request_trace_ledger_mismatch")
                seen_ids.add(request_id)
            if ledger_kinds is not None and seen_ids != set(ledger_kinds):
                raise _bad("request_trace_ledger_mismatch")
        # Durable live-job/submission-intent record: fully resolved.
        live = _read_json_file(
            run_dir / "live-jobs.json", f"live_jobs_missing:{run_key}"
        )
        if (
            live.get("schema") != LIVE_JOBS_SCHEMA
            or live.get("run_key") != run_key
            or live.get("submission_intents") != []
            or live.get("live_jobs") != []
            or live.get("uncertain_submissions") != []
        ):
            raise _bad("live_jobs_unresolved")
        # Trusted dataset/policy pins recorded by the runner at launch from
        # the REAL registry and pinned policy: every job's evaluator
        # evidence below is checked against THESE (and the frozen task
        # binding), never against itself.
        dataset = result.get("dataset")
        binding = cell.task_binding
        if (
            not isinstance(dataset, dict)
            or dataset.get("registry_task_id") != f"comparison_{cell.task_id}"
            or dataset.get("split") != "validation"
            or dataset.get("train_csv_sha256") != binding.train_csv_sha256
            or dataset.get("validation_features_sha256")
            != binding.validation_features_sha256
            or dataset.get("validation_labels_sha256")
            != binding.validation_labels_sha256
            or dataset.get("task_binding_identity") != binding.identity()
            or not isinstance(dataset.get("registry_version"), str)
            or not dataset["registry_version"]
        ):
            raise _bad("result_dataset_mismatch")
        registry_version = str(dataset["registry_version"])
        policy = load_policy()
        if (
            result.get("policy_version") != policy.policy_version
            or result.get("policy_sha256") != policy.policy_sha256
        ):
            raise _bad("result_policy_mismatch")
        # Every claimed prediction artifact must exist on disk with the
        # claimed hash (a hash without snapshot bytes is rejected) — shared
        # by the initial boundary AND every resume audit.
        artifacts_claimed = result.get("prediction_artifacts")
        if not isinstance(artifacts_claimed, dict):
            raise _bad("result_malformed")
        snapshot_bytes: dict[str, bytes] = {}
        for claimed_job, claimed_sha in artifacts_claimed.items():
            if not isinstance(claimed_job, str) or not claimed_job:
                raise _bad("result_malformed")
            _require_hex64(claimed_sha, f"config_evidence_invalid:{run_key}")
            snapshot_path = run_dir / "predictions" / f"{claimed_job}.csv"
            if snapshot_path.is_symlink() or not snapshot_path.exists():
                raise _bad("prediction_snapshot_missing")
            try:
                snapshot_data = snapshot_path.read_bytes()
            except OSError as exc:
                raise _bad("prediction_snapshot_missing") from exc
            if _sha256_bytes(snapshot_data) != claimed_sha:
                raise _bad("prediction_snapshot_mismatch")
            snapshot_bytes[claimed_job] = snapshot_data
        # Jobs: EVERY actual adapter record is verified, not only the
        # incumbent. A None job_id is a legitimate never-submitted local
        # failure (admission denial / pre-submission timeout): null score,
        # no phantom job. An explicit source-identity/cleanup contradiction
        # or evidence mismatch on ANY executed job is a global integrity
        # failure; ABSENT keys on ordinary failed jobs are not.
        jobs = result.get("jobs")
        job_ids = result.get("job_ids")
        if not isinstance(jobs, list) or any(
            not isinstance(job, dict) for job in jobs
        ):
            raise _bad("jobs_evidence_invalid")
        submitted_jobs = [job for job in jobs if job.get("job_id") is not None]
        if any(
            not isinstance(job["job_id"], str) or not job["job_id"]
            for job in submitted_jobs
        ):
            raise _bad("jobs_evidence_invalid")
        if job_ids != sorted({job["job_id"] for job in submitted_jobs}):
            raise _bad("jobs_evidence_invalid")
        journal_code_by_hash: dict[str, bytes] | None = None

        def _journal_map() -> dict[str, bytes]:
            # Actual candidate code bytes from the run journal, keyed by
            # their recomputed submitted-source hash: every executed scored
            # job must bind to one of these (never a digest-shaped claim).
            nonlocal journal_code_by_hash
            if journal_code_by_hash is None:
                journal_path = run_dir / "journal.jsonl"
                if journal_path.is_symlink() or not journal_path.exists():
                    raise _bad("journal_missing")
                try:
                    journal_lines = journal_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                except (OSError, UnicodeDecodeError) as exc:
                    raise _bad("journal_missing") from exc
                mapping: dict[str, bytes] = {}
                for journal_line in journal_lines:
                    if not journal_line.strip():
                        continue
                    try:
                        node = json.loads(journal_line)
                    except ValueError as exc:
                        raise _bad("journal_malformed") from exc
                    if (
                        isinstance(node, dict)
                        and isinstance(node.get("code"), str)
                        and node["code"]
                    ):
                        node_bytes = node["code"].encode(
                            "utf-8", errors="surrogatepass"
                        )
                        mapping[_sha256_bytes(node_bytes)] = node_bytes
                journal_code_by_hash = mapping
            return journal_code_by_hash

        for job in jobs:
            job_id = job.get("job_id")
            if job_id is None:
                if (
                    job.get("score") is not None
                    or job.get("evidence_verified") is True
                    or not isinstance(job.get("status"), str)
                ):
                    raise _bad("jobs_evidence_invalid")
                continue
            if (
                job.get("source_identity_verified") is False
                or job.get("worker_cleanup_verified") is False
                or job.get("evidence_mismatch") is not None
            ):
                raise _bad("job_execution_integrity_contradiction")
            if job.get("evidence_verified") is not True:
                if job.get("score") is not None:
                    raise _bad("jobs_evidence_invalid")
                continue
            # Verified (scored) executed job: full pinned provenance.
            if (
                job.get("source_identity_verified") is not True
                or job.get("worker_cleanup_verified") is not True
            ):
                raise _bad("job_execution_integrity_contradiction")
            job_score = job.get("score")
            if (
                isinstance(job_score, bool)
                or not isinstance(job_score, (int, float))
                or not math.isfinite(float(job_score))
                or not 0.0 <= float(job_score) <= 1.0
            ):
                raise _bad("jobs_evidence_invalid")
            evaluation = job.get("evaluation")
            gate = job.get("source_gate")
            if not isinstance(evaluation, dict) or not isinstance(gate, dict):
                raise _bad("job_evidence_missing")
            eval_score = evaluation.get("score")
            if (
                evaluation.get("evaluator") != "external_trusted"
                or evaluation.get("task_id") != f"comparison_{cell.task_id}"
                or evaluation.get("split") != "validation"
                or evaluation.get("metric") != "accuracy"
                or evaluation.get("direction") != "maximize"
                or evaluation.get("answer_sha256")
                != binding.validation_labels_sha256
                or evaluation.get("registry_version") != registry_version
                or isinstance(eval_score, bool)
                or not isinstance(eval_score, (int, float))
                or float(eval_score) != float(job_score)
            ):
                raise _bad("job_evaluation_mismatch")
            prediction_sha256 = _require_hex64(
                evaluation.get("prediction_sha256"),
                f"config_evidence_invalid:{run_key}",
            )
            if artifacts_claimed.get(job_id) != prediction_sha256:
                raise _bad("job_prediction_unbound")
            try:
                validate_job_snapshot(
                    evaluation.get("prediction_snapshot"), job_id
                )
            except ComparisonSpecError:
                raise _bad("job_prediction_path_invalid") from None
            try:
                data_rows = len(
                    [
                        line
                        for line in snapshot_bytes[job_id]
                        .decode("utf-8")
                        .splitlines()
                        if line.strip()
                    ]
                ) - 1
            except UnicodeDecodeError as exc:
                raise _bad("prediction_snapshot_mismatch") from exc
            if data_rows < 1 or evaluation.get("row_count") != data_rows:
                raise _bad("job_row_count_mismatch")
            submitted_sha = _require_hex64(
                job.get("submitted_source_sha256"),
                f"config_evidence_invalid:{run_key}",
            )
            if (
                gate.get("allowed") is not True
                or gate.get("entrypoint") != "main.py"
                or gate.get("entrypoint_sha256") != submitted_sha
                or gate.get("policy_version") != policy.policy_version
                or gate.get("policy_sha256") != policy.policy_sha256
            ):
                raise _bad("job_source_mismatch")
            # The executed source must be ACTUAL journal candidate bytes:
            # the submitted-source hash selects them and the whole-artifact
            # tree hash is recomputed from those bytes.
            job_code = _journal_map().get(submitted_sha)
            if job_code is None:
                raise _bad("job_source_mismatch")
            if gate.get("source_sha256") != _single_file_tree_sha256(
                str(gate["entrypoint"]), job_code
            ):
                raise _bad("job_source_mismatch")
        incumbent = result.get("incumbent")
        record_utility = record.get("utility")
        record_raw = record.get("raw_score")
        if incumbent is None:
            if record_utility != 0.0 or record_raw is not None:
                raise _bad("record_utility_mismatch")
            return main_claimed
        if not isinstance(incumbent, dict):
            raise _bad("incumbent_invalid")
        score = incumbent.get("validation_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or isinstance(record_utility, bool)
            or not isinstance(record_utility, (int, float))
            or float(record_utility) != float(score)
            or record_raw is None
            or isinstance(record_raw, bool)
            or not isinstance(record_raw, (int, float))
            or float(record_raw) != float(score)
        ):
            raise _bad("record_utility_mismatch")
        bound = record.get("incumbent")
        if not isinstance(bound, dict) or any(
            bound.get(field) != incumbent.get(field)
            for field in (
                "node_id",
                "job_id",
                "code_sha256",
                "prediction_sha256",
                "validation_score",
            )
        ):
            raise _bad("record_incumbent_mismatch")
        job = next(
            (entry for entry in submitted_jobs if entry["job_id"] == incumbent["job_id"]),
            None,
        )
        if job is None or job.get("evidence_verified") is not True:
            raise _bad("incumbent_job_unproven")
        job_score = job.get("score")
        if (
            isinstance(job_score, bool)
            or not isinstance(job_score, (int, float))
            or float(job_score) != float(score)
        ):
            raise _bad("incumbent_score_mismatch")
        evaluation = job.get("evaluation")
        gate = job.get("source_gate")
        if not isinstance(evaluation, dict) or not isinstance(gate, dict):
            raise _bad("incumbent_evidence_missing")
        if evaluation.get("prediction_sha256") != incumbent["prediction_sha256"]:
            raise _bad("incumbent_evaluation_mismatch")
        # The executed source must be the EXACT journal code bytes of the
        # incumbent node — entrypoint hash AND whole-artifact tree hash.
        if (
            job.get("submitted_source_sha256") != incumbent["code_sha256"]
            or gate.get("entrypoint_sha256") != incumbent["code_sha256"]
        ):
            raise _bad("incumbent_source_mismatch")
        code_bytes = self._extract_winner_bytes(run, str(incumbent["node_id"]))
        if _sha256_bytes(code_bytes) != incumbent["code_sha256"]:
            raise _bad("incumbent_source_mismatch")
        if gate.get("source_sha256") != _single_file_tree_sha256(
            str(gate["entrypoint"]), code_bytes
        ):
            raise _bad("incumbent_source_mismatch")
        return main_claimed

    def _validate_winner_record(
        self, cell: CellSpec, state: Mapping[str, Any], winner: Any
    ) -> None:
        cell_key = cell.cell_key()
        if not isinstance(winner, dict):
            raise OrchestrationError(f"winner_record_invalid:{cell_key}")
        config_index = winner.get("config_index")
        configs = state.get("configs")
        if (
            not isinstance(config_index, int)
            or isinstance(config_index, bool)
            or not isinstance(configs, dict)
            or str(config_index) not in configs
        ):
            raise OrchestrationError(f"winner_record_invalid:{cell_key}")
        source_sha256 = _require_hex64(
            winner.get("source_sha256"), f"winner_record_invalid:{cell_key}"
        )
        code_sha256 = _require_hex64(
            winner.get("code_sha256"), f"winner_record_invalid:{cell_key}"
        )
        # The winner must be the recorded incumbent of its own config.
        config_record = configs[str(config_index)]
        incumbent = (
            config_record.get("incumbent") if isinstance(config_record, dict) else None
        )
        if not isinstance(incumbent, dict) or (
            incumbent.get("code_sha256") != code_sha256
            or incumbent.get("job_id") != winner.get("job_id")
            or incumbent.get("prediction_sha256") != winner.get("prediction_sha256")
            or incumbent.get("node_id") != winner.get("node_id")
            or incumbent.get("validation_score") != winner.get("validation_score")
        ):
            raise OrchestrationError(f"winner_record_invalid:{cell_key}")
        winner_path = self._cell_dir(cell_key) / _WINNER_FILENAME
        if winner_path.is_symlink() or not winner_path.exists():
            raise OrchestrationError(f"winner_bytes_missing:{cell_key}")
        actual = _sha256_bytes(winner_path.read_bytes())
        if actual != source_sha256 or actual != code_sha256:
            raise OrchestrationError(f"winner_bytes_tampered:{cell_key}")

    # ------------------------------------------------------------ launch --
    def _cell_state(self, cell_key: str) -> dict[str, Any]:
        return self._load_cell_state(cell_key)

    def _vector_for(self, cell: CellSpec, config_index: int) -> tuple[float, ...]:
        state = self._cell_state(cell.cell_key())
        vectors = state.get("vectors") if isinstance(state, dict) else None
        raw = vectors.get(str(config_index)) if isinstance(vectors, dict) else None
        if raw is None:
            raise OrchestrationError(
                f"config_vector_unavailable:{cell.cell_key()}:{config_index}"
            )
        return validate_encoded_vector(raw)

    def _ensure_feedback_vectors(self, cell: CellSpec, state: dict[str, Any]) -> None:
        """ISTRATDE configs 4..7: derive and accept the second batch ONCE."""
        vectors = state["vectors"]
        if str(INITIAL_BATCH_CONFIGS) in vectors:
            return
        configs = state.get("configs")
        if not isinstance(configs, dict):
            raise OrchestrationError(f"feedback_prerequisite_missing:{cell.cell_key()}")
        utilities: list[float] = []
        accepted_first: list[list[float]] = []
        for index in range(INITIAL_BATCH_CONFIGS):
            record = configs.get(str(index))
            if not isinstance(record, dict) or not self._record_evaluated(record):
                raise OrchestrationError(
                    f"feedback_prerequisite_missing:{cell.cell_key()}"
                )
            utilities.append(float(record["utility"]))
            accepted_first.append(list(vectors[str(index)]))
        plan = feedback_theta_plan(cell, self.provider, utilities, accepted_first)
        for index, vector in plan.items():
            vectors[str(index)] = [float(v) for v in vector]
        self._persist_cell_state(cell.cell_key(), state)

    def prepare_config(
        self, position: SchedulePosition, ledger: ComparisonLedger
    ) -> Path | None:
        """Build, durably bind and record the intent for one config launch.

        Returns None when the encoded vector fails the launch-time theta
        gate: the config is then recorded as a truthful gate-rejected
        boundary (utility 0, no intent, no launch, no model call).
        """
        cell = self._cells_by_key[position.cell_key]
        run = cell.inner_run(position.config_index)
        run_key = run.run_key()
        if ledger.stopped:
            raise OrchestrationError("ledger_stopped")
        state = self._cell_state(position.cell_key)
        if state.get("terminal") is not None:
            raise OrchestrationError(f"cell_already_terminal:{run_key}")
        configs = state.get("configs")
        if not isinstance(configs, dict):
            ledger.stop(f"orchestration_spec_invalid:{run_key}")
            raise OrchestrationError(f"cell_state_configs_invalid:{run_key}")
        existing = configs.get(str(position.config_index))
        if existing is not None:
            status = existing.get("status")
            if status == "launched":
                raise OrchestrationError(f"config_interrupted:{run_key}")
            raise OrchestrationError(f"config_already_recorded:{run_key}")
        spec_path = self._spec_path(run)
        intent_path = self._intent_path(run)
        if spec_path.exists() or spec_path.is_symlink() or intent_path.exists() or intent_path.is_symlink():
            # Partial preparation from a crashed attempt: never continue silently.
            raise OrchestrationError(f"config_preparation_inconsistent:{run_key}")
        if position.config_index >= INITIAL_BATCH_CONFIGS and cell.method == "ISTRATDE":
            self._ensure_feedback_vectors(cell, state)
            state = self._cell_state(position.cell_key)
            configs = state.get("configs")
            if not isinstance(configs, dict):
                raise OrchestrationError(f"cell_state_configs_invalid:{run_key}")
        vector = self._vector_for(cell, position.config_index)
        # The weight-sum gate applies at LAUNCH time: a bounded native
        # vector that fails it is a truthful gate-rejected config (utility
        # 0, never launched, never resampled — pairing is preserved), not
        # an orchestration error and not a model call.
        try:
            decoded = decode_theta(vector)
            theta_gate(decoded)
        except ComparisonSpecError as exc:
            configs[str(position.config_index)] = {
                "status": "gate_rejected",
                "config_index": position.config_index,
                "position": position.position,
                "reason": str(exc),
                "utility": 0.0,
                "raw_score": None,
                "incumbent": None,
            }
            self._persist_cell_state(position.cell_key, state)
            return None
        try:
            spec = ComparisonRunSpec(
                run=run,
                decoded=decoded,
                run_dir=str(RUNTIME_DIR / _RUNS_DIR / run_dir_name(run)),
                ledger_path=str(RUNTIME_DIR / LEDGER_FILENAME),
                runner_commit=self.source_commit,
                resource_type="cpu",
                job_timeout_seconds=JOB_TIMEOUT_SECONDS,
            )
            spec.validate()
        except ComparisonSpecError as exc:
            ledger.stop(f"orchestration_spec_invalid:{run_key}")
            raise OrchestrationError(f"config_spec_invalid:{run_key}") from exc
        spec_text = canonical_json(spec.to_dict()) + "\n"
        # Durable pre-launch intent binding the COMPLETE spec bytes (incl.
        # job_timeout_seconds), the config identity and the schedule slot.
        intent = {
            "schema": INTENT_SCHEMA,
            "run_key": run_key,
            "cell_key": position.cell_key,
            "config_index": position.config_index,
            "position": position.position,
            "run_identity": run.identity(),
            "cell_identity": cell.identity(),
            "config_identity": comparison_config_identity(spec),
            "spec_sha256": _sha256_text(spec_text),
            "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
            "schedule_identity": self._schedule_identity,
            "launched_at": float(self._now()),
        }
        try:
            _write_durable(spec_path, spec_text)
            _write_durable(intent_path, canonical_json(intent) + "\n")
        except OSError as exc:
            ledger.stop(f"orchestration_intent_failed:{run_key}")
            raise OrchestrationError(f"config_intent_failed:{run_key}") from exc
        configs[str(position.config_index)] = {
            "status": "launched",
            "config_index": position.config_index,
            "position": position.position,
            "spec_sha256": intent["spec_sha256"],
            "config_identity": intent["config_identity"],
            "intent_sha256": _sha256_text(canonical_json(intent) + "\n"),
        }
        self._persist_cell_state(position.cell_key, state)
        return spec_path

    def complete_config(
        self, position: SchedulePosition, ledger: ComparisonLedger
    ) -> None:
        """Verify the child result and checkpoint the completed boundary."""
        cell = self._cells_by_key[position.cell_key]
        run = cell.inner_run(position.config_index)
        run_key = run.run_key()

        def _fail(reason: str) -> OrchestrationError:
            ledger.stop(f"orchestration_integrity:{reason}")
            return OrchestrationError(reason)

        state = self._cell_state(position.cell_key)
        configs = state.get("configs")
        if not isinstance(configs, dict):
            raise _fail(f"cell_state_configs_invalid:{run_key}")
        record = configs.get(str(position.config_index))
        if not isinstance(record, dict) or record.get("status") != "launched":
            raise _fail(f"config_not_launched:{run_key}")
        try:
            intent = _read_json_file(
                self._intent_path(run), f"intent_missing:{run_key}"
            )
        except OrchestrationError:
            raise _fail(f"intent_missing:{run_key}") from None
        if intent.get("schema") != INTENT_SCHEMA or intent.get("run_key") != run_key:
            raise _fail(f"intent_mismatch:{run_key}")
        if _sha256_text(canonical_json(intent) + "\n") != record.get("intent_sha256"):
            raise _fail(f"intent_tampered:{run_key}")
        try:
            spec_bytes = self._spec_path(run).read_bytes()
        except OSError:
            raise _fail(f"spec_missing:{run_key}") from None
        if _sha256_bytes(spec_bytes) != record.get("spec_sha256"):
            raise _fail(f"config_spec_tampered:{run_key}")
        try:
            spec = ComparisonRunSpec.from_dict(json.loads(spec_bytes))
        except (ComparisonSpecError, ValueError) as exc:
            raise _fail(f"config_spec_invalid:{run_key}") from exc
        if spec.run.identity() != run.identity():
            raise _fail(f"config_spec_invalid:{run_key}")
        result_path = self._result_path(run)
        if result_path.is_symlink() or not result_path.exists():
            raise _fail(f"result_missing:{run_key}")
        try:
            result_bytes = result_path.read_bytes()
            result = json.loads(result_bytes)
        except (OSError, ValueError):
            raise _fail(f"result_malformed:{run_key}") from None
        if not isinstance(result, dict) or result.get("schema") != RUN_RESULT_SCHEMA:
            raise _fail(f"result_malformed:{run_key}")
        if (
            result.get("run_key") != run_key
            or result.get("run_identity") != run.identity()
            or result.get("cell_identity") != cell.identity()
            or result.get("config_identity") != record.get("config_identity")
        ):
            raise _fail(f"result_identity_mismatch:{run_key}")
        status = result.get("status")
        termination = result.get("termination")
        if status not in ("completed", "cell_terminated") or not isinstance(
            termination, str
        ):
            raise _fail(f"result_not_terminal:{run_key}")
        if result.get("worker_cleanup_verified") is not True:
            raise _fail(f"cleanup_unverified:{run_key}")
        if ledger.stopped:
            raise _fail(f"ledger_stopped:{run_key}")
        # A cell-cap claim must be backed by the ORIGINAL ledger's durable
        # cell termination record; a bare result flag proves nothing.
        ledger_view = ledger.snapshot()
        if status == "cell_terminated" or termination.startswith("CellBudgetExhausted"):
            cell_view = ledger_view["cells"].get(position.cell_key)
            if not isinstance(cell_view, dict) or cell_view.get("terminated") is None:
                raise _fail(f"cell_cap_claim_unproven:{run_key}")
        # Every claimed prediction artifact is bound (bytes + hash) inside
        # the shared evidence verification below; only the incumbent's own
        # claim is cross-checked here against the claimed map.
        artifacts_claimed = result.get("prediction_artifacts")
        if not isinstance(artifacts_claimed, dict):
            raise _fail(f"result_malformed:{run_key}")
        incumbent = result.get("incumbent")
        utility = 0.0
        raw_score: float | None = None
        winner_binding: dict[str, Any] | None = None
        if incumbent is not None:
            if not isinstance(incumbent, dict):
                raise _fail(f"incumbent_invalid:{run_key}")
            score = incumbent.get("validation_score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise _fail(f"incumbent_score_invalid:{run_key}")
            node_id = incumbent.get("node_id")
            job_id = incumbent.get("job_id")
            code_sha256 = incumbent.get("code_sha256")
            prediction_sha256 = incumbent.get("prediction_sha256")
            if (
                not isinstance(node_id, str)
                or not node_id
                or not isinstance(job_id, str)
                or not job_id
            ):
                raise _fail(f"incumbent_invalid:{run_key}")
            _require_hex64(code_sha256, f"incumbent_invalid:{run_key}")
            _require_hex64(prediction_sha256, f"incumbent_invalid:{run_key}")
            if artifacts_claimed.get(job_id) != prediction_sha256:
                raise _fail(f"incumbent_prediction_mismatch:{run_key}")
            utility = float(score)
            raw_score = float(score)
            winner_binding = {
                "node_id": node_id,
                "job_id": job_id,
                "code_sha256": code_sha256,
                "prediction_sha256": prediction_sha256,
                "validation_score": float(score),
            }
        configs[str(position.config_index)] = {
            "status": "completed",
            "config_index": position.config_index,
            "position": position.position,
            "spec_sha256": record["spec_sha256"],
            "config_identity": record["config_identity"],
            "intent_sha256": record["intent_sha256"],
            "result_sha256": _sha256_bytes(result_bytes),
            "termination": termination,
            "utility": utility,
            "raw_score": raw_score,
            "incumbent": winner_binding,
        }
        try:
            # Full shared evidence verification (decoded vector binding,
            # six-main request reconciliation against the ORIGINAL ledger,
            # traces/journal/live-jobs/jobs/snapshots) BEFORE the boundary
            # persists. The verified main-request count decides whether this
            # config was REALLY evaluated (a valid theta whose first main
            # reservation hit the cell cap was never evaluated).
            main_claimed = self._verify_result_evidence(
                cell, run, configs[str(position.config_index)], result, ledger_view
            )
        except OrchestrationError as exc:
            raise _fail(str(exc)) from exc
        configs[str(position.config_index)]["evaluated"] = main_claimed > 0
        self._persist_cell_state(position.cell_key, state)
        if termination.startswith("CellBudgetExhausted") or status == "cell_terminated":
            self.freeze_cell(position.cell_key, ledger, reason="cell_budget_exhausted")
        else:
            self._maybe_freeze(position.cell_key, ledger)

    def _maybe_freeze(self, cell_key: str, ledger: ComparisonLedger) -> None:
        """Freeze when every config slot is resolved (any truthful status)."""
        state = self._cell_state(cell_key)
        if state.get("terminal") is not None:
            return
        configs = state.get("configs")
        if not isinstance(configs, dict):
            return
        resolved = sum(
            1
            for record in configs.values()
            if isinstance(record, dict)
            and record.get("status")
            in ("completed", "gate_rejected", "not_launched_cell_exhausted")
        )
        if resolved == CONFIGS_PER_CELL:
            self.freeze_cell(cell_key, ledger, reason="all_configs_completed")

    # ------------------------------------------------------------ freeze --
    def freeze_cell(
        self, cell_key: str, ledger: ComparisonLedger, *, reason: str
    ) -> None:
        """Freeze the cell winner: highest validation, tie earliest config.

        The exact winning program bytes are recovered from the run journal
        and re-hashed against the incumbent binding; any contradiction stops
        the ledger globally. A cell with no valid program records a truthful
        ``failed_no_valid_program`` terminal state — never a synthesized
        fallback.
        """
        state = self._cell_state(cell_key)
        if state.get("terminal") is not None:
            return  # freezing is idempotent; first freeze wins
        configs = state.get("configs")
        if not isinstance(configs, dict):
            ledger.stop(f"orchestration_integrity:freeze_state_invalid:{cell_key}")
            raise OrchestrationError(f"freeze_state_invalid:{cell_key}")
        candidates: list[tuple[float, int, dict[str, Any]]] = []
        for key, record in configs.items():
            if not isinstance(record, dict) or record.get("status") != "completed":
                continue
            incumbent = record.get("incumbent")
            raw_score = record.get("raw_score")
            if incumbent is None or raw_score is None:
                continue
            candidates.append((float(raw_score), int(key), dict(incumbent)))
        # Remaining pending configs of an exhausted cell are permanently
        # skipped (truthful shortfall, never silently rerun).
        for index in range(CONFIGS_PER_CELL):
            if str(index) not in configs:
                configs[str(index)] = {
                    "status": "not_launched_cell_exhausted",
                    "config_index": index,
                    "position": next(
                        p.position
                        for p in self.positions
                        if p.cell_key == cell_key and p.config_index == index
                    ),
                }
        cell = self._cells_by_key[cell_key]
        if cell.method == "ISTRATDE" and all(
            self._record_evaluated(configs[str(index)])
            for index in range(CONFIGS_PER_CELL)
        ):
            # All eight configs have REAL resolved outcomes (an evaluated
            # run sent >=1 main request; gate-rejected vectors are truthful
            # known zeros): perform and persist the full two-tell native
            # feedback evidence. A cell with an UNEVALUATED slot (zero-request
            # cap, skipped suffix) skips this — unevaluated slots never get
            # invented fitness.
            try:
                state["native_completion"] = self._native_completion_record(
                    cell, state
                )
            except OrchestrationError as exc:
                ledger.stop(f"orchestration_integrity:{exc}")
                raise
        if not candidates:
            state["terminal"] = {"status": "failed_no_valid_program", "reason": reason}
            state["winner"] = None
            self._persist_cell_state(cell_key, state)
            return
        # Highest validation score; tie -> lowest config index == earliest
        # actual evaluation order (schedule guarantees per-cell config order).
        best_score, best_index, best = max(candidates, key=lambda c: (c[0], -c[1]))
        winner_bytes = self._extract_winner_bytes(
            self._cells_by_key[cell_key].inner_run(best_index), best["node_id"]
        )
        actual_sha = _sha256_bytes(winner_bytes)
        if actual_sha != best["code_sha256"]:
            ledger.stop(f"orchestration_integrity:winner_bytes_mismatch:{cell_key}")
            raise OrchestrationError(f"winner_bytes_mismatch:{cell_key}")
        winner_path = self._cell_dir(cell_key) / _WINNER_FILENAME
        try:
            _write_durable(
                winner_path, winner_bytes.decode("utf-8", errors="surrogatepass")
            )
        except (OSError, UnicodeDecodeError) as exc:
            ledger.stop(f"orchestration_integrity:winner_persist_failed:{cell_key}")
            raise OrchestrationError(f"winner_persist_failed:{cell_key}") from exc
        state["winner"] = {
            "config_index": best_index,
            "node_id": best["node_id"],
            "validation_score": best_score,
            "code_sha256": best["code_sha256"],
            "source_sha256": actual_sha,
            "job_id": best["job_id"],
            "prediction_sha256": best["prediction_sha256"],
        }
        state["terminal"] = {"status": "frozen", "reason": reason}
        self._persist_cell_state(cell_key, state)

    def _extract_winner_bytes(self, run: InnerRunSpec, node_id: str) -> bytes:
        journal_path = self._run_dir(run) / "journal.jsonl"
        if journal_path.is_symlink() or not journal_path.exists():
            raise OrchestrationError(f"journal_missing:{run.run_key()}")
        try:
            lines = journal_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise OrchestrationError(f"journal_missing:{run.run_key()}") from exc
        matches: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                node = json.loads(line)
            except ValueError as exc:
                raise OrchestrationError(f"journal_malformed:{run.run_key()}") from exc
            if isinstance(node, dict) and node.get("id") == node_id:
                code = node.get("code")
                if not isinstance(code, str) or not code:
                    raise OrchestrationError(f"journal_malformed:{run.run_key()}")
                matches.append(code)
        if len(matches) != 1:
            raise OrchestrationError(f"winner_node_not_unique:{run.run_key()}")
        return matches[0].encode("utf-8", errors="surrogatepass")

    # ------------------------------------------------------------- drive --
    def _acquire_controller_lock(self):
        """Separate CONTROLLER ownership (distinct from the ledger flock).

        Held for the entire run_schedule operation so concurrent
        controllers cannot race while the ledger lock is handed to a child.
        """
        import fcntl

        lock_path = self.runtime_dir / "controller.lock"
        if lock_path.is_symlink():
            raise OrchestrationError("controller_lock_symlinked")
        handle = open(lock_path, "a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise OrchestrationError("orchestration_controller_active") from exc
        return handle

    def run_schedule(
        self, ledger: ComparisonLedger, child_runner: ChildRunner
    ) -> dict[str, Any]:
        """Execute the frozen schedule in order, boundary-checkpointed.

        The production child opens the ledger itself (exclusive flock), so
        the parent RELINQUISHES the lock around every child invocation and
        reacquires it afterwards with the identical clock. This method takes
        ownership of the passed handle: on return OR raise, every handle it
        touched is closed; callers reopen to inspect. A stale CLOSED handle
        is never used to overwrite freshly settled child accounting.

        The WHOLE operation — resume audit, state reads, gate-rejection
        freezing, launches, boundary completion and the final summary — is
        protected: any OrchestrationError (or unexpected I/O error) durably
        stops the original ledger through a VALID open handle.

        Resumable: completed configs are skipped; an interrupted
        (launched-but-uncompleted) config fails closed. Any child failure
        durably stops the original ledger before propagating.
        """
        controller_lock = self._acquire_controller_lock()
        try:
            return self._run_schedule_locked(ledger, child_runner)
        finally:
            import fcntl

            try:
                fcntl.flock(controller_lock.fileno(), fcntl.LOCK_UN)
            finally:
                controller_lock.close()

    def _run_schedule_locked(
        self, ledger: ComparisonLedger, child_runner: ChildRunner
    ) -> dict[str, Any]:
        current = ledger
        try:
            # Resume audit: re-verify EVERY persisted boundary (including
            # request/ledger reconciliation) against the freshly opened
            # ORIGINAL ledger before stepping any position. The original
            # 24h deadline is enforced per launch (and post-child) below.
            audit_view = current.snapshot()
            for cell in self.cells:
                self._validate_cell_state(
                    cell, self._cell_state(cell.cell_key()), audit_view
                )
            for position in self.positions:
                cell_key = position.cell_key
                run_key = f"{cell_key}:{position.config_index}"
                current = self._step_position(
                    position, run_key, cell_key, current, child_runner
                )
                if current.stopped:
                    raise OrchestrationError("ledger_stopped")
            terminal = self.terminal_summary()
            if len(terminal) != 27:
                current.stop("orchestration_incomplete_schedule")
                raise OrchestrationError("schedule_incomplete")
            return terminal
        except OrchestrationError as exc:
            # ANY boundary/state/summary integrity error durably stops the
            # original ledger through a VALID open handle (first stop wins;
            # a resume-state refusal makes the matrix permanently
            # INCOMPLETE). A handle closed for the child handoff is NEVER
            # used for the write.
            if not current.closed and not current.stopped:
                current.stop(f"orchestration_operation_failure:{exc}")
            raise
        except Exception as exc:
            if not current.closed and not current.stopped:
                current.stop(f"orchestration_error:{type(exc).__name__}")
            raise OrchestrationError(f"orchestration_error:{exc}") from exc
        finally:
            if not current.closed:
                current.close()

    def _check_deadline(self, ledger: ComparisonLedger) -> None:
        """The original 24h deadline, checked at every transition boundary."""
        deadline = ledger.deadline_epoch
        if deadline is not None and ledger.now_fn() > deadline:
            ledger.stop("global_exhausted:deadline_expired")
            raise OrchestrationError("deadline_expired")

    def _step_position(
        self,
        position: SchedulePosition,
        run_key: str,
        cell_key: str,
        current: ComparisonLedger,
        child_runner: ChildRunner,
    ) -> ComparisonLedger:
        """One schedule position; returns the open ledger handle (the
        child handoff may replace it)."""
        if current.stopped:
            raise OrchestrationError("ledger_stopped")
        state = self._cell_state(cell_key)
        if state.get("terminal") is not None:
            return current
        configs = state.get("configs")
        record = (
            configs.get(str(position.config_index))
            if isinstance(configs, dict)
            else None
        )
        if isinstance(record, dict) and record.get("status") in (
            "completed",
            "not_launched_cell_exhausted",
            "gate_rejected",
        ):
            return current
        if record is not None:
            # Launched but never completed: interrupted, never rerun.
            raise OrchestrationError(f"config_interrupted:{run_key}")
        # The original 24h deadline gates EVERY new launch; the
        # clock function is the ledger's own (persisted start).
        deadline = current.deadline_epoch
        if deadline is not None and current.now_fn() > deadline:
            current.stop("global_exhausted:deadline_expired")
            raise OrchestrationError("deadline_expired")
        try:
            spec_path = self.prepare_config(position, current)
        except OrchestrationError:
            raise
        except Exception as exc:
            current.stop(f"orchestration_error:{type(exc).__name__}")
            raise OrchestrationError(f"orchestration_error:{run_key}") from exc
        if spec_path is None:
            # Gate-rejected config: boundary recorded, no launch.
            self._maybe_freeze(position.cell_key, current)
            return current
        return self._launch_and_complete(
            position, run_key, spec_path, current, child_runner
        )

    def _launch_and_complete(
        self,
        position: SchedulePosition,
        run_key: str,
        spec_path: Path,
        current: ComparisonLedger,
        child_runner: ChildRunner,
    ) -> ComparisonLedger:
        """Lock handoff + child + boundary completion for one config.

        Closes ``current`` so the production child can own the ledger
        (exclusive flock), reacquires it with the identical clock, and
        returns the NEW open handle (the caller's old handle stays closed
        and is NEVER written through again). Every handle acquired here is
        closed on every exception path.
        """
        ledger_path, now_fn = current.path, current.now_fn
        current.close()
        exit_code: int | None = None
        child_error: BaseException | None = None
        try:
            exit_code = child_runner(spec_path)
        except BaseException as exc:  # noqa: BLE001 - re-stop then raise
            child_error = exc
        try:
            fresh = ComparisonLedger.open(
                ledger_path, cells=self.cells, now_fn=now_fn
            )
        except Exception as exc:
            # Reopen validation stops durably on unresolved state; never
            # let a raw ledger error masquerade as progress, and never
            # write through the stale CLOSED parent handle.
            raise OrchestrationError(f"ledger_reopen_failed:{run_key}") from exc
        try:
            if child_error is not None:
                fresh.stop(f"orchestration_child_error:{type(child_error).__name__}")
                raise OrchestrationError(f"child_error:{run_key}") from child_error
            if exit_code != 0:
                fresh.stop(f"orchestration_child_failed:{run_key}")
                raise OrchestrationError(f"child_failed:{run_key}")
            # The original deadline is re-checked immediately AFTER the
            # child, before its boundary is accepted.
            self._check_deadline(fresh)
            try:
                self.complete_config(position, fresh)
            except OrchestrationError:
                # Boundary integrity failure AFTER the child: stop the
                # reacquired (original) ledger durably before propagating.
                fresh.stop(f"orchestration_boundary_failure:{run_key}")
                raise
            return fresh
        except BaseException:
            fresh.close()
            raise

    def terminal_summary(self) -> dict[str, Any]:
        """cell_key -> terminal record for every terminal cell."""
        summary: dict[str, Any] = {}
        for cell in self.cells:
            state = self._cell_state(cell.cell_key())
            terminal = state.get("terminal")
            if isinstance(terminal, dict):
                summary[cell.cell_key()] = terminal
        return summary

    # ------------------------------------------------------------ barrier --
    def verify_final_barrier(self, ledger: ComparisonLedger) -> dict[str, Any]:
        """The all-27-cell frozen barrier; returns hash-bound evidence.

        Re-validates the COMPLETE roster and the actual terminal evidence
        (cell states, winner bytes, spec and result hashes, ledger roster
        identity and zero open reservations). A cell with no valid program
        is terminal with a truthful failure — never a fallback classifier.
        """
        if ledger.stopped:
            raise OrchestrationError("ledger_stopped")
        # The original deadline gates the barrier transition too.
        deadline = ledger.deadline_epoch
        if deadline is not None and ledger.now_fn() > deadline:
            ledger.stop("global_exhausted:deadline_expired")
            raise OrchestrationError("deadline_expired")
        # Compare the ledger's persisted experiment identity against the
        # ORCHESTRATOR's actual cell roster — never the ledger against
        # itself.
        if ledger.experiment_identity != comparison_experiment_identity(self.cells):
            raise OrchestrationError("barrier_experiment_mismatch")
        return self.compute_barrier_payload(ledger.snapshot())

    def compute_barrier_payload(
        self, snapshot: Mapping[str, Any]
    ) -> dict[str, Any]:
        """READ-ONLY barrier computation over a validated ledger snapshot.

        The same roster/terminal-evidence validation as
        verify_final_barrier, without the live-ledger gates (stop
        transitions, deadline enforcement): used by the offline audit
        (EXP-008) with a ReadOnlyComparisonLedger view so reporting never
        mutates the original ledger. Never call this to AUTHORIZE the
        final phase — only verify_final_barrier does that.
        """
        if snapshot["global"]["open_reservations"] != 0:
            raise OrchestrationError("barrier_open_reservations")
        if snapshot["experiment_identity"] != comparison_experiment_identity(
            self.cells
        ):
            raise OrchestrationError("barrier_experiment_mismatch")
        cells_evidence: dict[str, Any] = {}
        for cell in self.cells:
            cell_key = cell.cell_key()
            state = self._cell_state(cell_key)
            self._validate_cell_state(cell, state, snapshot)
            terminal = state.get("terminal")
            if terminal is None:
                raise OrchestrationError(f"barrier_cell_not_terminal:{cell_key}")
            winner = state.get("winner")
            entry: dict[str, Any] = {"terminal_status": terminal["status"]}
            if terminal["status"] == "frozen":
                if not isinstance(winner, dict):
                    raise OrchestrationError(f"winner_record_invalid:{cell_key}")
                entry["winner"] = {
                    "config_index": winner["config_index"],
                    "validation_score": winner["validation_score"],
                    "code_sha256": winner["code_sha256"],
                    "source_sha256": winner["source_sha256"],
                    "job_id": winner["job_id"],
                    "prediction_sha256": winner["prediction_sha256"],
                }
            else:
                entry["winner"] = None
            cells_evidence[cell_key] = entry
        payload: dict[str, Any] = {
            "schema": BARRIER_SCHEMA,
            "schedule_identity": self._schedule_identity,
            "experiment_identity": snapshot["experiment_identity"],
            "source_commit": self.source_commit,
            "cells": cells_evidence,
        }
        payload["barrier_identity"] = identity_of(payload)
        return payload


def load_orchestrator(
    *,
    runtime_dir: Path | None = None,
    repo_root: Path,
    bindings: Mapping[str, Any],
    provider: NativeProvider,
    now_fn: Callable[[], float] = time.time,
) -> Orchestrator:
    """Resume helper bound to the default comparison runtime dir."""
    return Orchestrator.load(
        runtime_dir=runtime_dir or (repo_root / RUNTIME_DIR),
        repo_root=repo_root,
        bindings=bindings,
        provider=provider,
        now_fn=now_fn,
    )
