"""Offline orchestration tests (EXP-006).

Drive the Orchestrator with a full-evidence fake child runner (no model
calls, no sandbox) and real tmp ledgers, proving boundary-only resume,
durable intents, tamper detection, winner freezing, cell-cap shortfall,
native two-tell completion evidence, the global barrier and fail-closed
stop semantics.

The positive fixture is the SAME evidence contract required of real runs:
every completed config charges the ORIGINAL ledger (six main attempts for
an ordinary completion), writes request/operator/generation traces bound
to those ledger request ids, a journal with the exact winning bytes,
prediction snapshots, a resolved live-jobs record and full trusted
job/evaluator evidence. A module-scoped "golden" completed experiment is
built ONCE and reused (read-only or via copies) so hundreds of durable
mock runs are not repeated per test.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pytest

from research.comparison import deploy
from research.comparison.budget import ComparisonLedger, ProviderUsage
from research.comparison.orchestrate import (
    JOB_TIMEOUT_SECONDS,
    Orchestrator,
    OrchestrationError,
)
from research.comparison.runner import (
    LIVE_JOBS_SCHEMA,
    RUN_RESULT_SCHEMA,
    ComparisonRunSpec,
    comparison_config_identity,
)
from research.comparison.specs import iter_cells
from research.tests.test_comparison_specs import COMMIT, make_bindings


class ScriptedNative:
    """Deterministic fake provider; records second-batch/completion fitness."""

    def __init__(self) -> None:
        self.second_calls: list[tuple[int, tuple[float, ...]]] = []
        self.complete_calls: list[
            tuple[int, tuple[float, ...], tuple[float, ...]]
        ] = []

    def setup_population(self, seed: int):
        return self.first_batch(seed)

    def first_batch(self, seed: int):
        base = (seed % 100) / 1000.0
        return tuple(
            (0.05 + base + 0.01 * i, 1.0, 0.4, 0.25, 1.0) for i in range(4)
        )

    def _feedback_batch(self, seed: int):
        base = (seed % 100) / 1000.0
        return tuple((0.5 + base + 0.01 * i, 0.9, 0.3, 0.2, 1.0) for i in range(4))

    def second_batch(self, seed: int, fitness):
        self.second_calls.append((seed, tuple(fitness)))
        return self.first_batch(seed), self._feedback_batch(seed)

    def complete_run(self, seed: int, first_fitness, second_fitness):
        from research.comparison.native import NativeCompletion

        self.complete_calls.append(
            (seed, tuple(first_fitness), tuple(second_fitness))
        )
        digest = hashlib.sha256(
            json.dumps(
                [
                    seed,
                    [float(f) for f in first_fitness],
                    [float(f) for f in second_fitness],
                ]
            ).encode()
        ).hexdigest()
        return NativeCompletion(
            first_batch=self.first_batch(seed),
            second_batch=self._feedback_batch(seed),
            final_population_sha256=digest,
        )


def make_orch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider=None) -> Orchestrator:
    monkeypatch.setattr(deploy, "git_head_commit", lambda root: COMMIT)
    return Orchestrator.initialize(
        runtime_dir=tmp_path / "state",
        repo_root=tmp_path,
        cells=iter_cells(make_bindings(), COMMIT),
        provider=provider or ScriptedNative(),
        now_fn=lambda: 1000.0,
    )


def open_ledger(tmp_path: Path, orch: Orchestrator) -> ComparisonLedger:
    return ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: 1000.0)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def charge_main_requests(
    orch: Orchestrator, spec: ComparisonRunSpec, ledger_path: Path, count: int
) -> list[str]:
    """Charge the ORIGINAL ledger exactly like the real runner would."""
    request_ids: list[str] = []
    with ComparisonLedger.open(
        ledger_path, cells=orch.cells, now_fn=lambda: 1000.0
    ) as owned:
        for index in range(count):
            reservation = owned.reserve(spec.run, "main", estimated_input_tokens=100)
            owned.commit(
                reservation.request_id,
                ProviderUsage(60, 40),
                provider_request_id=f"{spec.run.run_key()}-{index}",
            )
            request_ids.append(reservation.request_id)
    return request_ids


def write_fake_result(
    orch: Orchestrator,
    spec: ComparisonRunSpec,
    score: float | None,
    *,
    termination: str = "completed",
    status: str = "completed",
    request_ids: list[str] | tuple[str, ...] = (),
) -> None:
    """Full-evidence synthetic run result (the SAME contract the real
    runner produces and the orchestrator boundary verifies): decoded spec
    binding, traces bound to the charged request ids, journal, prediction
    snapshot, resolved live-jobs and trusted job/evaluator evidence."""
    run_dir = orch._run_dir(spec.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_key = spec.run.run_key()
    from research.contracts.source_gate import load_policy
    from research.search.sandbox_task import _single_file_tree_sha256

    policy = load_policy()
    operators = ("draft", "draft", "improve", "improve", "crossover", "crossover")
    trace_records = [
        {
            "run_key": run_key,
            "attempt": index + 1,
            "operator": operators[index % len(operators)],
            "kind": "main",
            "request_id": request_id,
            "reserved_tokens": 8292,
            "request": {"model": "k3", "stream": True},
        }
        for index, request_id in enumerate(request_ids)
    ]
    _write_jsonl(run_dir / "request-trace.jsonl", trace_records)
    _write_jsonl(
        run_dir / "operator-trace.jsonl",
        [{"operator": record["operator"]} for record in trace_records],
    )
    _write_jsonl(
        run_dir / "generation-outcomes.jsonl",
        [{"generation": g, "outcome": "evaluated"} for g in range(3)],
    )
    (run_dir / "live-jobs.json").write_text(
        json.dumps(
            {
                "schema": LIVE_JOBS_SCHEMA,
                "run_key": run_key,
                "submission_intents": [],
                "live_jobs": [],
                "uncertain_submissions": [],
                "updated_at": 1000.0,
            }
        ),
        encoding="utf-8",
    )
    incumbent = None
    artifacts: dict[str, str] = {}
    jobs: list[dict] = []
    journal_nodes: list[dict] = [{"id": "root-node", "code": "", "metric": None}]
    if score is not None:
        code = f"# candidate for {run_key}\nprint('ok')\n"
        code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
        pred_bytes = f"id,label\n0,1\n".encode()
        pred_sha = hashlib.sha256(pred_bytes).hexdigest()
        predictions_dir = run_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        job_id = f"job-{uuid.uuid5(uuid.NAMESPACE_URL, run_key).hex[:12]}"
        (predictions_dir / f"{job_id}.csv").write_bytes(pred_bytes)
        node_id = uuid.uuid5(uuid.NAMESPACE_URL, f"node-{run_key}").hex
        journal_nodes.append({"id": node_id, "code": code, "metric": score})
        incumbent = {
            "node_id": node_id,
            "validation_score": score,
            "code_sha256": code_sha,
            "job_id": job_id,
            "prediction_sha256": pred_sha,
            "operators_used": ["draft"],
        }
        artifacts = {job_id: pred_sha}
        jobs = [
            {
                "job_id": job_id,
                "status": "completed",
                "score": score,
                "evidence_verified": True,
                "submitted_source_sha256": code_sha,
                "source_gate": {
                    "allowed": True,
                    "entrypoint": "main.py",
                    "entrypoint_sha256": code_sha,
                    "source_sha256": _single_file_tree_sha256(
                        "main.py", code.encode("utf-8")
                    ),
                    "policy_version": policy.policy_version,
                    "policy_sha256": policy.policy_sha256,
                },
                "evaluation": {
                    "evaluator": "external_trusted",
                    "task_id": f"comparison_{spec.run.cell.task_id}",
                    "split": "validation",
                    "metric": "accuracy",
                    "direction": "maximize",
                    "score": score,
                    "row_count": 1,
                    "answer_sha256": spec.run.cell.task_binding.validation_labels_sha256,
                    "registry_version": "evaluator-registry.v1",
                    "prediction_sha256": pred_sha,
                    "prediction_snapshot": (
                        f"/mnt/rsi_storage/mlsandbox/jobs/{job_id}/prediction_snapshot.csv"
                    ),
                },
                "source_identity_verified": True,
                "worker_cleanup_verified": True,
            }
        ]
    _write_jsonl(run_dir / "journal.jsonl", journal_nodes)
    result = {
        "schema": RUN_RESULT_SCHEMA,
        "run_key": run_key,
        "run_identity": spec.run.identity(),
        "cell_identity": spec.run.cell.identity(),
        "config_identity": comparison_config_identity(spec),
        "decoded": dict(spec.decoded),
        "inner_seed": spec.run.inner_seed,
        "status": status,
        "termination": termination,
        "incumbent": incumbent,
        "prediction_artifacts": artifacts,
        "policy_version": policy.policy_version,
        "policy_sha256": policy.policy_sha256,
        "dataset": {
            "registry_task_id": f"comparison_{spec.run.cell.task_id}",
            "registry_version": "evaluator-registry.v1",
            "split": "validation",
            "worker_data_public_path": (
                f"/mnt/rsi_data/comparison/{spec.run.cell.task_id}/data/public"
            ),
            "train_csv_sha256": spec.run.cell.task_binding.train_csv_sha256,
            "validation_features_sha256": (
                spec.run.cell.task_binding.validation_features_sha256
            ),
            "validation_labels_sha256": (
                spec.run.cell.task_binding.validation_labels_sha256
            ),
            "task_binding_identity": spec.run.cell.task_binding.identity(),
        },
        "worker_cleanup_verified": True,
        "requests": {
            "main": len(request_ids),
            "debug": 0,
            "charged_tokens": 100 * len(request_ids),
        },
        "main_candidates": len(request_ids),
        "generations_completed": 3 if status == "completed" else 1,
        "journal_sha256": _sha256_file(run_dir / "journal.jsonl"),
        "operator_trace_sha256": _sha256_file(run_dir / "operator-trace.jsonl"),
        "request_trace_sha256": _sha256_file(run_dir / "request-trace.jsonl"),
        "generation_outcomes_sha256": _sha256_file(
            run_dir / "generation-outcomes.jsonl"
        ),
        "jobs": jobs,
        "job_counts": {"completed": len(jobs)},
        "job_ids": sorted(job["job_id"] for job in jobs),
        "budget": {},
        "started_at": 1000.0,
        "finished_at": 1000.0,
    }
    (run_dir / "run-result.json").write_text(json.dumps(result), encoding="utf-8")


def fake_child(
    orch: Orchestrator,
    scores: dict[str, float | None],
    calls: list[str],
    ledger_path: Path,
):
    """Full-evidence fake child: six charged main attempts + artifacts."""

    def child(spec_path: Path) -> int:
        spec = ComparisonRunSpec.from_dict(json.loads(spec_path.read_text()))
        run_key = spec.run.run_key()
        calls.append(run_key)
        outcome = scores.get(run_key, 0.5)
        if isinstance(outcome, dict):
            # Cap script: {"mains": n, "terminate": reason}
            request_ids = charge_main_requests(
                orch, spec, ledger_path, int(outcome["mains"])
            )
            with ComparisonLedger.open(
                ledger_path, cells=orch.cells, now_fn=lambda: 1000.0
            ) as owned:
                owned.terminate_cell(spec.run.cell.identity(), str(outcome["terminate"]))
            write_fake_result(
                orch,
                spec,
                outcome.get("score"),
                termination=f"CellBudgetExhausted:cell_terminated:{outcome['terminate']}",
                status="cell_terminated",
                request_ids=request_ids,
            )
            return 0
        request_ids = charge_main_requests(orch, spec, ledger_path, 6)
        write_fake_result(orch, spec, outcome, request_ids=request_ids)
        return 0

    return child


def default_scores(orch: Orchestrator) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for pos in orch.positions:
        scores[f"{pos.cell_key}:{pos.config_index}"] = round(
            0.5 + 0.01 * pos.config_index, 6
        )
    return scores


def step_positions(
    orch: Orchestrator,
    ledger: ComparisonLedger,
    positions,
    child,
) -> ComparisonLedger:
    """Drive selected schedule positions through the PRODUCTION handoff."""
    current = ledger
    for position in positions:
        current = orch._step_position(
            position,
            f"{position.cell_key}:{position.config_index}",
            position.cell_key,
            current,
            child,
        )
    return current


@pytest.fixture(scope="module")
def golden(tmp_path_factory) -> dict:
    """ONE complete, fully evidenced 216-config experiment, reused by the
    read-only/resume/tamper tests (copied when a test must mutate)."""
    base = tmp_path_factory.mktemp("golden")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(deploy, "git_head_commit", lambda root: COMMIT)
        provider = ScriptedNative()
        orch = Orchestrator.initialize(
            runtime_dir=base / "state",
            repo_root=base,
            cells=iter_cells(make_bindings(), COMMIT),
            provider=provider,
            now_fn=lambda: 1000.0,
        )
        ledger_path = base / "ledger.json"
        with ComparisonLedger.open(
            ledger_path, cells=orch.cells, now_fn=lambda: 1000.0
        ) as ledger:
            ledger.start_clock()
            summary = orch.run_schedule(
                ledger, fake_child(orch, default_scores(orch), [], ledger_path)
            )
    return {
        "base": base,
        "orch": orch,
        "provider": provider,
        "summary": summary,
        "ledger_path": ledger_path,
    }


def copy_golden(golden: dict, tmp_path: Path) -> dict:
    base = tmp_path / "work"
    shutil.copytree(golden["base"], base)
    return {
        "base": base,
        "ledger_path": base / "ledger.json",
        "runtime_dir": base / "state",
    }


def load_copied(copy: dict, provider=None) -> Orchestrator:
    return Orchestrator.load(
        runtime_dir=copy["runtime_dir"],
        repo_root=copy["base"],
        bindings=make_bindings(),
        provider=provider or ScriptedNative(),
        now_fn=lambda: 1000.0,
    )


def test_full_schedule_freezes_all_27_winners(golden):
    orch = golden["orch"]
    provider = golden["provider"]
    summary = golden["summary"]
    assert len(summary) == 27
    assert all(record["status"] == "frozen" for record in summary.values())
    # ISTRATDE second batches: exactly 9, each with negated first-batch utilities.
    assert len(provider.second_calls) == 9
    for _seed, fitness in provider.second_calls:
        assert fitness == (-0.5, -0.51, -0.52, -0.53)
    # ISTRATDE two-tell completions: exactly 9 (one per ISTRATDE cell), each
    # replaying BOTH accepted fitness histories (configs 0..3 and 4..7).
    assert len(provider.complete_calls) == 9
    for _seed, first_fitness, second_fitness in provider.complete_calls:
        assert first_fitness == (-0.5, -0.51, -0.52, -0.53)
        assert second_fitness == (-0.54, -0.55, -0.56, -0.57)
    # Winner of each cell is its config 7 (highest score), bytes on disk.
    for cell in orch.cells:
        state = json.loads(
            (orch.runtime_dir / "cells" / cell.cell_key().replace(":", "-") / "cell.json").read_text()
        )
        assert state["winner"]["config_index"] == 7
        assert state["winner"]["validation_score"] == pytest.approx(0.57)
        winner_path = orch.runtime_dir / "cells" / cell.cell_key().replace(":", "-") / "winner.py"
        assert winner_path.exists()
        assert hashlib.sha256(winner_path.read_bytes()).hexdigest() == state["winner"]["source_sha256"] == state["winner"]["code_sha256"]
        assert winner_path.read_text().startswith(f"# candidate for {cell.cell_key()}:7")
        if cell.method == "ISTRATDE":
            completion = state["native_completion"]
            assert len(completion["final_population_sha256"]) == 64
            assert completion["first_fitness"] == [-0.5, -0.51, -0.52, -0.53]
            assert completion["second_fitness"] == [-0.54, -0.55, -0.56, -0.57]
        else:
            assert state["native_completion"] is None
    # The original ledger recorded exactly 6 main requests per config.
    with ComparisonLedger.open(
        golden["ledger_path"], cells=orch.cells, now_fn=lambda: 1000.0
    ) as ledger:
        snapshot = ledger.snapshot()
        assert snapshot["global"]["committed"]["requests"] == 216 * 6
        barrier = orch.verify_final_barrier(ledger)
    assert barrier["schema"] == "istratde-comparison-barrier.v1"
    assert len(barrier["cells"]) == 27
    assert all(
        entry["terminal_status"] == "frozen" for entry in barrier["cells"].values()
    )
    # Resume re-validates everything and reruns nothing.
    resumed = Orchestrator.load(
        runtime_dir=orch.runtime_dir,
        repo_root=golden["base"],
        bindings=make_bindings(),
        provider=ScriptedNative(),
        now_fn=lambda: 1000.0,
    )
    calls2: list[str] = []
    with ComparisonLedger.open(
        golden["ledger_path"], cells=resumed.cells, now_fn=lambda: 1000.0
    ) as ledger:
        summary2 = resumed.run_schedule(
            ledger, fake_child(resumed, {}, calls2, golden["ledger_path"])
        )
    assert calls2 == []
    assert summary2 == summary


def test_winner_tie_breaks_earliest_config(tmp_path, monkeypatch):
    orch = make_orch(tmp_path, monkeypatch)
    cell_key = "banknote:11:FIXED"
    positions = [p for p in orch.positions if p.cell_key == cell_key]
    scores = {f"{cell_key}:{index}": 0.6 for index in range(8)}  # all tied
    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        current = step_positions(
            orch, ledger, positions, fake_child(orch, scores, [], tmp_path / "ledger.json")
        )
        current.close()
    state = json.loads(
        (orch.runtime_dir / "cells" / cell_key.replace(":", "-") / "cell.json").read_text()
    )
    assert state["terminal"]["status"] == "frozen"
    assert state["winner"]["config_index"] == 0


def test_cell_cap_freezes_incumbent_and_continues(tmp_path, monkeypatch):
    orch = make_orch(tmp_path, monkeypatch)
    scores = default_scores(orch)
    capped_cell = "wdbc:22:RANDOM"
    scores[f"{capped_cell}:3"] = {
        "mains": 4,
        "terminate": "cell_main_budget_insufficient",
        "score": None,
    }
    positions = [p for p in orch.positions if p.cell_key == capped_cell]
    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        current = step_positions(
            orch, ledger, positions, fake_child(orch, scores, [], tmp_path / "ledger.json")
        )
        assert not current.stopped
        current.close()
    state = json.loads(
        (orch.runtime_dir / "cells" / capped_cell.replace(":", "-") / "cell.json").read_text()
    )
    assert state["terminal"]["status"] == "frozen"
    assert state["terminal"]["reason"] == "cell_budget_exhausted"
    assert state["winner"]["config_index"] == 2  # best of configs 0..2
    for index in range(4, 8):
        assert state["configs"][str(index)]["status"] == "not_launched_cell_exhausted"
    # Truthful partial counts are retained in the ORIGINAL ledger.
    with open_ledger(tmp_path, orch) as ledger:
        snapshot = ledger.snapshot()
        cell_view = snapshot["cells"][capped_cell]
        assert cell_view["terminated"] is not None
        run_identities = {
            orch._cells_by_key[capped_cell].inner_run(i).identity() for i in range(4)
        }
        partial = [
            r for r in snapshot["requests"] if r["run_identity"] in run_identities
        ]
        assert len(partial) == 6 + 6 + 6 + 4


def test_cell_with_no_valid_program_is_truthful_failure(tmp_path, monkeypatch):
    orch = make_orch(tmp_path, monkeypatch)
    scores = default_scores(orch)
    failed_cell = "spambase:33:ISTRATDE"
    for index in range(8):
        scores[f"{failed_cell}:{index}"] = None  # every config failed
    positions = [p for p in orch.positions if p.cell_key == failed_cell]
    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        current = step_positions(
            orch, ledger, positions, fake_child(orch, scores, [], tmp_path / "ledger.json")
        )
        current.close()
    state = json.loads(
        (orch.runtime_dir / "cells" / failed_cell.replace(":", "-") / "cell.json").read_text()
    )
    assert state["terminal"]["status"] == "failed_no_valid_program"
    assert state["winner"] is None
    assert not (orch.runtime_dir / "cells" / failed_cell.replace(":", "-") / "winner.py").exists()
    # Both tells still ran with the truthful zero utilities.
    completion = state["native_completion"]
    assert completion["first_fitness"] == [-0.0] * 4
    assert completion["second_fitness"] == [-0.0] * 4


def test_interrupted_config_never_reruns(tmp_path, monkeypatch):
    orch = make_orch(tmp_path, monkeypatch)

    def crashing_child(spec_path: Path) -> int:
        return 2  # child died before writing any result

    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        with pytest.raises(OrchestrationError, match="child_failed"):
            orch.run_schedule(ledger, crashing_child)
    with open_ledger(tmp_path, orch) as ledger:
        assert ledger.stopped
    # Resume: the launched-but-uncompleted config fails closed.
    with pytest.raises(OrchestrationError, match="config_interrupted"):
        Orchestrator.load(
            runtime_dir=orch.runtime_dir,
            repo_root=tmp_path,
            bindings=make_bindings(),
            provider=ScriptedNative(),
            now_fn=lambda: 1000.0,
        )


def test_resume_after_partial_completion_fails_closed_on_interrupted(tmp_path, monkeypatch):
    orch = make_orch(tmp_path, monkeypatch)
    scores = default_scores(orch)
    calls: list[str] = []
    child = fake_child(orch, scores, calls, tmp_path / "ledger.json")

    def stop_after_five(spec_path: Path) -> int:
        code = child(spec_path)
        if len(calls) >= 5:
            # Process death after the result bytes landed but BEFORE the
            # parent checkpointed the boundary.
            raise RuntimeError("simulated_power_loss")
        return code

    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        with pytest.raises(OrchestrationError, match="child_error"):
            orch.run_schedule(ledger, stop_after_five)
    assert len(calls) == 5
    with open_ledger(tmp_path, orch) as reopened:
        assert reopened.stopped
    # Resume: configs 1-4 are completed boundaries; the 5th was launched
    # but never checkpointed -> resume itself refuses (never rerun).
    with pytest.raises(OrchestrationError, match="config_interrupted"):
        Orchestrator.load(
            runtime_dir=orch.runtime_dir,
            repo_root=tmp_path,
            bindings=make_bindings(),
            provider=ScriptedNative(),
            now_fn=lambda: 1000.0,
        )


def test_tampered_result_rejected_on_resume(golden, tmp_path):
    copy = copy_golden(golden, tmp_path)
    run_dir = copy["runtime_dir"] / "runs" / "banknote-11-FIXED-0"
    result_path = run_dir / "run-result.json"
    payload = json.loads(result_path.read_text())
    payload["termination"] = "tampered"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OrchestrationError, match="config_result_tampered"):
        load_copied(copy)


def test_tampered_request_trace_rejected_on_resume(golden, tmp_path):
    copy = copy_golden(golden, tmp_path)
    run_dir = copy["runtime_dir"] / "runs" / "banknote-11-FIXED-0"
    (run_dir / "request-trace.jsonl").write_text("[]\n", encoding="utf-8")
    with pytest.raises(OrchestrationError, match="trace_tampered"):
        load_copied(copy)


def test_tampered_winner_bytes_rejected(golden, tmp_path):
    copy = copy_golden(golden, tmp_path)
    winner_path = copy["runtime_dir"] / "cells" / "banknote-11-FIXED" / "winner.py"
    winner_path.write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(OrchestrationError, match="winner_bytes_tampered"):
        load_copied(copy)


def test_barrier_refused_before_all_cells_terminal(tmp_path, monkeypatch):
    orch = make_orch(tmp_path, monkeypatch)
    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        with pytest.raises(OrchestrationError, match="barrier_cell_not_terminal"):
            orch.verify_final_barrier(ledger)


def test_barrier_refused_with_open_reservation(golden, tmp_path):
    copy = copy_golden(golden, tmp_path)
    orch = load_copied(copy)
    with ComparisonLedger.open(
        copy["ledger_path"], cells=orch.cells, now_fn=lambda: 1000.0
    ) as ledger:
        reservation = ledger.reserve(
            orch.cells[0].inner_run(0), "debug", estimated_input_tokens=10, debug_limit=2
        )
        assert isinstance(reservation.request_id, str)
        with pytest.raises(OrchestrationError, match="barrier_open_reservations"):
            orch.verify_final_barrier(ledger)


def test_barrier_refused_when_ledger_stopped(golden, tmp_path):
    copy = copy_golden(golden, tmp_path)
    orch = load_copied(copy)
    with ComparisonLedger.open(
        copy["ledger_path"], cells=orch.cells, now_fn=lambda: 1000.0
    ) as ledger:
        ledger.stop("test_stop")
        with pytest.raises(OrchestrationError, match="ledger_stopped"):
            orch.verify_final_barrier(ledger)


def test_barrier_experiment_identity_is_orchestrator_derived(golden, tmp_path):
    """The barrier compares the ledger against the ORCHESTRATOR's roster,
    never against itself."""
    copy = copy_golden(golden, tmp_path)
    orch = load_copied(copy)
    other_commit = "f" * 40
    other_cells = iter_cells(make_bindings(), other_commit)
    with ComparisonLedger.open(
        tmp_path / "other-ledger.json", cells=other_cells, now_fn=lambda: 1000.0
    ) as foreign:
        foreign.start_clock()
        with pytest.raises(OrchestrationError, match="barrier_experiment_mismatch"):
            orch.verify_final_barrier(foreign)


def test_clean_boundary_resume_reruns_nothing(golden):
    """Resume of the completed experiment re-verifies all evidence and
    launches zero children."""
    resumed = Orchestrator.load(
        runtime_dir=golden["orch"].runtime_dir,
        repo_root=golden["base"],
        bindings=make_bindings(),
        provider=ScriptedNative(),
        now_fn=lambda: 1000.0,
    )
    calls: list[str] = []
    with ComparisonLedger.open(
        golden["ledger_path"], cells=resumed.cells, now_fn=lambda: 1000.0
    ) as ledger:
        summary = resumed.run_schedule(
            ledger, fake_child(resumed, {}, calls, golden["ledger_path"])
        )
    assert calls == []
    assert summary == golden["summary"]


def test_gate_rejected_vector_is_truthful_zero_without_launch(tmp_path, monkeypatch):
    """A bounded all-zero-weight native vector: retained, never launched."""

    class ZeroWeightNative(ScriptedNative):
        def first_batch(self, seed: int):
            valid = super().first_batch(seed)
            return ((0.5, 0.0, 0.0, 0.0, 1.0), *valid[1:])

        def _feedback_batch(self, seed: int):
            return tuple((0.6, 0.9, 0.3, 0.2, 1.0) for _ in range(4))

    provider = ZeroWeightNative()
    orch = make_orch(tmp_path, monkeypatch, provider)
    calls: list[str] = []
    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        summary = orch.run_schedule(
            ledger, fake_child(orch, default_scores(orch), calls, tmp_path / "ledger.json")
        )
    # Config 0 of every RANDOM/ISTRATDE cell (9+9) was gate-rejected:
    # retained in the plan, zero utility, never launched, no model call.
    rejected = [k for k in summary]
    assert len(rejected) == 27
    for method in ("RANDOM", "ISTRATDE"):
        for task in ("banknote", "wdbc", "spambase"):
            for seed in (11, 22, 33):
                key = f"{task}:{seed}:{method}"
                assert f"{key}:0" not in calls
                state = json.loads(
                    (
                        orch.runtime_dir / "cells" / key.replace(":", "-") / "cell.json"
                    ).read_text()
                )
                record = state["configs"]["0"]
                assert record["status"] == "gate_rejected"
                assert record["utility"] == 0.0 and record["raw_score"] is None
                assert state["vectors"]["0"] == [0.5, 0.0, 0.0, 0.0, 1.0]
                # ISTRATDE feedback saw utility 0 for the rejected slot.
    for _seed, fitness in provider.second_calls:
        assert fitness[0] == -0.0
    # ISTRATDE config 4 vectors derive from the second batch (0.6 crossover).
    state = json.loads(
        (orch.runtime_dir / "cells" / "banknote-11-ISTRATDE" / "cell.json").read_text()
    )
    assert state["vectors"]["4"][0] == pytest.approx(0.6)
    # Winners still freeze from the valid configs (config 7, score 0.57);
    # resume re-validates the gate claims.
    resumed = Orchestrator.load(
        runtime_dir=orch.runtime_dir,
        repo_root=tmp_path,
        bindings=make_bindings(),
        provider=ZeroWeightNative(),
        now_fn=lambda: 1000.0,
    )
    assert resumed.terminal_summary() == summary


def test_gate_claim_tampering_rejected_on_resume(tmp_path, monkeypatch):
    """A gate_rejected record whose vector actually PASSES the gate is
    tampering and fails closed on resume."""

    class ZeroWeightNative(ScriptedNative):
        def first_batch(self, seed: int):
            valid = super().first_batch(seed)
            return ((0.5, 0.0, 0.0, 0.0, 1.0), *valid[1:])

        def _feedback_batch(self, seed: int):
            return tuple((0.6, 0.9, 0.3, 0.2, 1.0) for _ in range(4))

    orch = make_orch(tmp_path, monkeypatch, ZeroWeightNative())
    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        orch.run_schedule(
            ledger,
            fake_child(orch, default_scores(orch), [], tmp_path / "ledger.json"),
        )
    path = orch.runtime_dir / "cells" / "banknote-11-RANDOM" / "cell.json"
    state = json.loads(path.read_text())
    state["vectors"]["0"] = [0.5, 1.0, 0.4, 0.25, 1.0]  # valid AND gate-passing
    path.write_text(json.dumps(state))
    with pytest.raises(OrchestrationError):
        Orchestrator.load(
            runtime_dir=orch.runtime_dir,
            repo_root=tmp_path,
            bindings=make_bindings(),
            provider=ZeroWeightNative(),
            now_fn=lambda: 1000.0,
        )


def test_child_failure_stops_original_ledger(tmp_path, monkeypatch):
    orch = make_orch(tmp_path, monkeypatch)
    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        with pytest.raises(OrchestrationError, match="child_failed"):
            orch.run_schedule(ledger, lambda spec_path: 2)
    with open_ledger(tmp_path, orch) as reopened:
        assert reopened.stopped


def test_intent_binds_complete_spec_bytes(tmp_path, monkeypatch):
    orch = make_orch(tmp_path, monkeypatch)
    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        spec_path = orch.prepare_config(orch.positions[0], ledger)
        intent = json.loads(
            (orch.runtime_dir / "intents" / spec_path.name).read_text()
        )
        spec_bytes = spec_path.read_bytes()
        assert intent["spec_sha256"] == hashlib.sha256(spec_bytes).hexdigest()
        assert intent["job_timeout_seconds"] == JOB_TIMEOUT_SECONDS
        spec = json.loads(spec_bytes)
        assert spec["job_timeout_seconds"] == JOB_TIMEOUT_SECONDS
        assert spec["resource_type"] == "cpu"
        assert spec["runner_commit"] == COMMIT
        assert intent["config_identity"]
        assert intent["schedule_identity"]
        # Drifted on-disk spec is detected at completion.
        spec_path.write_text(
            json.dumps({**spec, "job_timeout_seconds": 601}), encoding="utf-8"
        )
        with pytest.raises(OrchestrationError, match="config_spec_tampered"):
            orch.complete_config(orch.positions[0], ledger)
        assert ledger.stopped


def test_malformed_incumbent_stops_ledger(tmp_path, monkeypatch):
    orch = make_orch(tmp_path, monkeypatch)

    def child(spec_path: Path) -> int:
        spec = ComparisonRunSpec.from_dict(json.loads(spec_path.read_text()))
        request_ids = charge_main_requests(orch, spec, tmp_path / "ledger.json", 6)
        write_fake_result(orch, spec, 0.5, request_ids=request_ids)
        result_path = orch._result_path(spec.run)
        payload = json.loads(result_path.read_text())
        payload["incumbent"]["validation_score"] = 1.7  # out of range
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return 0

    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        with pytest.raises(OrchestrationError, match="incumbent_score_invalid"):
            orch.run_schedule(ledger, child)
    # The caller's handle was closed by the lock handoff; the durable stop
    # is verified through a fresh handle.
    with open_ledger(tmp_path, orch) as reopened:
        assert reopened.stopped


def test_winner_bytes_mismatch_stops_ledger(tmp_path, monkeypatch):
    orch = make_orch(tmp_path, monkeypatch)

    def child(spec_path: Path) -> int:
        spec = ComparisonRunSpec.from_dict(json.loads(spec_path.read_text()))
        request_ids = charge_main_requests(orch, spec, tmp_path / "ledger.json", 6)
        write_fake_result(orch, spec, 0.5, request_ids=request_ids)
        # Journal code that does NOT match the incumbent code hash.
        journal = orch._run_dir(spec.run) / "journal.jsonl"
        lines = journal.read_text().splitlines()
        node = json.loads(lines[-1])
        node["code"] = "# different bytes\n"
        lines[-1] = json.dumps(node)
        journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0

    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        with pytest.raises(OrchestrationError, match="trace_tampered|winner_bytes_mismatch"):
            orch.run_schedule(ledger, child)
    with open_ledger(tmp_path, orch) as reopened:
        assert reopened.stopped


class AllInvalidNative(ScriptedNative):
    """Every vector is bounded/finite but fails the weight-sum gate."""

    def first_batch(self, seed: int):
        return ((0.5, 0.0, 0.0, 0.0, 1.0),) * 4

    def _feedback_batch(self, seed: int):
        return ((0.5, 0.0, 0.0, 0.0, 1.0),) * 4


def test_all_gate_rejected_istratde_cell_zero_requests_two_tells(tmp_path, monkeypatch):
    """All-eight-invalid cells: ZERO requests, truthful failure, real DE feedback.

    Every frozen vector is bounded but gate-rejected: no child launches, no
    model requests, no resampling; the cell legitimately ends
    failed_no_valid_program and the zero utilities still participate in BOTH
    native tells (recorded as two-tell completion evidence).
    """
    provider = AllInvalidNative()
    orch = make_orch(tmp_path, monkeypatch, provider)
    calls: list[str] = []
    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        summary = orch.run_schedule(
            ledger, fake_child(orch, default_scores(orch), calls, tmp_path / "ledger.json")
        )
    # run_schedule closes its handle on return; reopen to inspect.
    with open_ledger(tmp_path, orch) as ledger:
        snapshot = ledger.snapshot()
    assert len(provider.complete_calls) == 9
    for cell in orch.cells:
        if cell.method != "ISTRATDE":
            continue
        key = cell.cell_key()
        assert summary[key]["status"] == "failed_no_valid_program"
        assert not any(call.startswith(f"{key}:") for call in calls)
        assert snapshot["cells"][key]["committed"] == {"requests": 0, "tokens": 0}
        state = json.loads(
            (orch.runtime_dir / "cells" / key.replace(":", "-") / "cell.json").read_text()
        )
        assert len(state["configs"]) == 8
        assert all(
            record["status"] == "gate_rejected" for record in state["configs"].values()
        )
        completion = state["native_completion"]
        assert completion["first_fitness"] == [-0.0] * 4
        assert completion["second_fitness"] == [-0.0] * 4
        assert len(completion["final_population_sha256"]) == 64
    # The barrier accepts zero-request truthful cells (no positive-request
    # heuristic) and resume re-verifies the completion evidence.
    with open_ledger(tmp_path, orch) as ledger:
        barrier = orch.verify_final_barrier(ledger)
    assert len(barrier["cells"]) == 27
    resumed = Orchestrator.load(
        runtime_dir=orch.runtime_dir,
        repo_root=tmp_path,
        bindings=make_bindings(),
        provider=AllInvalidNative(),
        now_fn=lambda: 1000.0,
    )
    assert resumed.terminal_summary() == summary


def test_capped_istratde_cell_has_no_invented_completion_fitness(tmp_path, monkeypatch):
    """A capped iStratDE cell with unevaluated slots gets NO completion record."""
    provider = ScriptedNative()
    orch = make_orch(tmp_path, monkeypatch, provider)
    scores = default_scores(orch)
    capped_cell = "banknote:11:ISTRATDE"
    scores[f"{capped_cell}:6"] = {
        "mains": 5,
        "terminate": "cell_main_budget_insufficient",
        "score": None,
    }
    positions = [p for p in orch.positions if p.cell_key == capped_cell]
    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        current = step_positions(
            orch, ledger, positions, fake_child(orch, scores, [], tmp_path / "ledger.json")
        )
        current.close()
    state = json.loads(
        (orch.runtime_dir / "cells" / capped_cell.replace(":", "-") / "cell.json").read_text()
    )
    assert state["terminal"]["status"] == "frozen"
    assert state["configs"]["7"]["status"] == "not_launched_cell_exhausted"
    # Unevaluated slots never receive invented fitness: no completion record.
    assert state["native_completion"] is None
    assert state["winner"]["config_index"] == 5
    # Resume accepts the truthful partial cell.
    resumed = Orchestrator.load(
        runtime_dir=orch.runtime_dir,
        repo_root=tmp_path,
        bindings=make_bindings(),
        provider=ScriptedNative(),
        now_fn=lambda: 1000.0,
    )
    assert resumed.terminal_summary()[capped_cell]["status"] == "frozen"


# ------------------------------------------------- ownership/recovery (EXP-006) --


def test_concurrent_controller_lock_refused(tmp_path, monkeypatch):
    """A second controller cannot race while the first holds ownership
    (including during the ledger handoff window)."""
    orch = make_orch(tmp_path, monkeypatch)
    held = orch._acquire_controller_lock()
    try:
        with open_ledger(tmp_path, orch) as ledger:
            ledger.start_clock()
            with pytest.raises(
                OrchestrationError, match="orchestration_controller_active"
            ):
                orch.run_schedule(
                    ledger,
                    fake_child(orch, default_scores(orch), [], tmp_path / "ledger.json"),
                )
            assert not ledger.stopped  # refused BEFORE any mutation
    finally:
        import fcntl

        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()
    # After release the same orchestrator can drive.
    positions = [p for p in orch.positions if p.cell_key == "banknote:11:FIXED"]
    with open_ledger(tmp_path, orch) as ledger:
        current = step_positions(
            orch,
            ledger,
            positions,
            fake_child(orch, default_scores(orch), [], tmp_path / "ledger.json"),
        )
        current.close()


def test_runtime_path_symlink_escape_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "git_head_commit", lambda root: COMMIT)
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = repo / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    # A symlinked ancestor redirecting the runtime dir is refused before
    # ANY state write.
    with pytest.raises(OrchestrationError, match="runtime_path_symlinked"):
        Orchestrator.initialize(
            runtime_dir=alias / "state",
            repo_root=repo,
            cells=iter_cells(make_bindings(), COMMIT),
            provider=ScriptedNative(),
            now_fn=lambda: 1000.0,
        )
    assert not (outside / "state").exists()
    # The runtime dir itself being a symlink is refused.
    with pytest.raises(OrchestrationError, match="runtime_path_symlinked"):
        Orchestrator.initialize(
            runtime_dir=alias,
            repo_root=repo,
            cells=iter_cells(make_bindings(), COMMIT),
            provider=ScriptedNative(),
            now_fn=lambda: 1000.0,
        )
    # A ".." escape out of the repo root is refused.
    with pytest.raises(OrchestrationError, match="runtime_dir_escapes_repo"):
        Orchestrator.initialize(
            runtime_dir=repo / ".." / "outside" / "state",
            repo_root=repo,
            cells=iter_cells(make_bindings(), COMMIT),
            provider=ScriptedNative(),
            now_fn=lambda: 1000.0,
        )


def test_child_bounded_by_original_remaining_time(tmp_path):
    """evo_child_runner refuses to spawn once the ORIGINAL deadline has
    passed, before any subprocess starts."""
    from research.comparison.orchestrate import evo_child_runner

    runner = evo_child_runner(tmp_path, deadline_fn=lambda: 1.0)  # long past
    with pytest.raises(OrchestrationError, match="child_deadline_exceeded"):
        runner(tmp_path / "spec.json")


def test_exception_closes_every_owned_handle(tmp_path, monkeypatch):
    """A boundary failure after the child must not leak the reacquired
    ledger handle (no stale-handle writes, immediate reopen works)."""
    orch = make_orch(tmp_path, monkeypatch)

    def child(spec_path: Path) -> int:
        spec = ComparisonRunSpec.from_dict(json.loads(spec_path.read_text()))
        request_ids = charge_main_requests(orch, spec, tmp_path / "ledger.json", 6)
        write_fake_result(orch, spec, 0.5, request_ids=request_ids)
        # Corrupt the result AFTER charging: boundary completion fails.
        result_path = orch._result_path(spec.run)
        payload = json.loads(result_path.read_text())
        payload["termination"] = "corrupted"
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        return 0

    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        with pytest.raises(OrchestrationError):
            orch.run_schedule(ledger, child)
    # If any acquired handle had leaked, this NB-flock open would fail.
    with open_ledger(tmp_path, orch) as reopened:
        assert reopened.stopped
        # The child's freshly settled accounting survived (never overwritten
        # by a stale closed parent handle).
        snapshot = reopened.snapshot()
        assert snapshot["global"]["committed"]["requests"] == 6
