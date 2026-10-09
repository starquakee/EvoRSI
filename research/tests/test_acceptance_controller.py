"""US-009 parent-controller tests (offline; fake algo/child, REAL ledger).

The fake child opens the SAME real BudgetLedger path the parent uses, which
also proves the flock protocol: the parent must not hold the lock while a
child runs. No model, sandbox or Docker activity. Tests use a temporary
repo_root with the SAME fixed relative runtime location as production —
the production contract pins the one acceptance ledger/run directory and is
never relaxed for tests.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import pytest

from research.contracts.budget import (
    BudgetLedger,
    LedgerStopped,
    ModelUsage,
)
from research.search import acceptance as ap
from research.search import acceptance_config as ac
from research.search.acceptance import (
    AcceptanceError,
    cleanup_child_jobs,
    run_acceptance,
)
from research.search.acceptance_inner import config_identity_hash
from research.search.run_index import RunIndex
from research.search.search_vector import decode_retained

REAL_REPO = Path(__file__).resolve().parents[2]

# The recorded supervisor preflight vectors (native seed-42/pop-4 ask).
VECTORS = [
    [0.7870834469795227, 0.3256833553314209, 0.23071765899658203, 0.9376811981201172, 0.9818315505981445],
    [0.5546902418136597, 0.03373408317565918, 1.503467321395874, 0.031850576400756836, 4.237701416015625],
    [0.462753027677536, 0.0, 0.0, 0.5288450717926025, 4.657905578613281],
    [0.9279928803443909, 0.4288446009159088, 0.0, 0.5781111717224121, 0.0],
]
COMMIT = "c" * 40
SCORES = {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.8}
POLICY_SHA = "b" * 64
ANSWER_SHA = "a" * 64


class FakeAlgo:
    def __init__(self, vectors):
        self._vectors = vectors
        self.told = None

    def ask(self):
        return [list(v) for v in self._vectors]

    def tell(self, fitness):
        self.told = list(fitness)


def _repo(tmp_path, monkeypatch):
    """A temporary repo root with the SAME fixed relative runtime layout as
    production plus the trusted registry file the run identity binds."""
    monkeypatch.setattr(ac, "git_head_commit", lambda root: COMMIT)
    runtime = tmp_path / ac.RUNTIME_DIR
    preflight = tmp_path / ac.PREFLIGHT_VECTORS_FILE
    preflight.parent.mkdir(parents=True, exist_ok=True)
    preflight.write_text(json.dumps({"seed": 42, "pop_size": 4, "vectors": VECTORS}), encoding="utf-8")
    registry_dst = tmp_path / ac.EVALUATOR_REGISTRY
    registry_dst.parent.mkdir(parents=True, exist_ok=True)
    registry_dst.write_bytes((REAL_REPO / ac.EVALUATOR_REGISTRY).read_bytes())
    return tmp_path, runtime


def _fake_child(runtime: Path, calls: list, *, behaviors=None, ledger_activity=True):
    """Fake inner child: opens the REAL ledger (proves flock handover),
    reserves+commits one request, writes the full bound artifact set and
    returns a schema-complete run-result dict."""

    def child(spec, spec_path, *, repo_root):
        calls.append(spec.run_id)
        if ledger_activity:
            with BudgetLedger.open(runtime / ac.LEDGER_FILENAME) as ledger:
                reservation = ledger.reserve(100, 100)
                ledger.commit(reservation.reservation_id, ModelUsage(input_tokens=60, output_tokens=40))
        behavior = (behaviors or {}).get(spec.run_id, {})
        position = int(spec.run_id.removeprefix("cfg"))
        status = behavior.get("status", "completed")
        score = SCORES[position] if status == "completed" else None
        counts = behavior.get(
            "operator_counts", {"draft": 6, "improve": 1, "debug": 1, "crossover": 1}
        )
        job_id = f"job_{spec.run_id}"
        result = {
            "schema": "acceptance-inner-result.v1",
            "run_id": spec.run_id,
            "decoded": dict(spec.decoded),
            "config_hash": config_identity_hash(spec.decoded),
            "policy_version": "source-gate.v2",
            "policy_sha256": POLICY_SHA,
            "runner_commit": COMMIT,
            "status": status,
            "termination": "completed" if status == "completed" else "BudgetError",
            "best_score": score,
            "best_job_id": job_id if status == "completed" else "",
            "main_candidates": 6 if status == "completed" else 0,
            "generations_completed": 3 if status == "completed" else 0,
            "operator_counts": counts,
            "operator_trace_sha256": "",
            "request_trace_sha256": "",
            "journal_sha256": "",
            "jobs": [],
            "job_ids": [],
            "worker_cleanup_verified": status == "completed",
            "model_usage": {"requests_delta": 1, "tokens_delta": 100},
            "budget": {"committed": {}, "remaining": {}, "stopped": None, "open_reservations": []},
            "finished_at": time.time(),
        }
        if status == "completed":
            run_dir = runtime / "runs" / spec.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            prediction_bytes = f"id,label\nprediction-{spec.run_id}\n".encode()
            prediction_sha = hashlib.sha256(prediction_bytes).hexdigest()
            artifacts = {
                "operator-trace.jsonl": json.dumps({"run": spec.run_id, "operator": "draft"}).encode() + b"\n",
                "request-trace.jsonl": json.dumps({"run": spec.run_id, "reservation_id": "r"}).encode() + b"\n",
                "journal.jsonl": json.dumps({"run": spec.run_id, "code": "print(1)"}).encode() + b"\n",
            }
            for name, data in artifacts.items():
                (run_dir / name).write_bytes(data)
                field = name.replace("-", "_").replace(".jsonl", "_sha256")
                result[field] = hashlib.sha256(data).hexdigest()
            predictions_dir = run_dir / "predictions"
            predictions_dir.mkdir(exist_ok=True)
            (predictions_dir / f"{job_id}.csv").write_bytes(prediction_bytes)
            result["prediction_artifacts"] = {job_id: prediction_sha}
            result["jobs"] = [{
                "job_id": job_id,
                "status": "completed",
                "score": score,
                "evidence_verified": True,
                "evaluation": {
                    "prediction_sha256": prediction_sha,
                    "answer_sha256": ANSWER_SHA,
                },
            }]
            result["job_ids"] = [job_id]
        run_dir = runtime / "runs" / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run-result.json").write_text(
            json.dumps(result, sort_keys=True), encoding="utf-8"
        )
        return result

    return child


def test_happy_path_cache_reuse_and_tell(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    algos = []

    def algo_factory():
        algo = FakeAlgo(VECTORS)
        algos.append(algo)
        return algo

    calls: list = []
    summary = run_acceptance(
        repo, runtime,
        algo_factory=algo_factory,
        child_runner=_fake_child(runtime, calls),
    )
    # standalone run + 3 outer evaluations; position 0 reused from the index
    assert calls == ["cfg0", "cfg1", "cfg2", "cfg3"]
    assert len(algos) == 2  # standalone ask + outer re-init ask
    assert algos[1].told == [-0.5, -0.6, -0.7, -0.8]  # fitness = -accuracy
    assert summary["outer"]["outer_steps_completed"] == 1
    assert summary["verdict"] == "complete"
    assert summary["operator_coverage"] == {"draft": 24, "improve": 4, "debug": 4, "crossover": 4}
    assert summary["budget"]["end_committed"]["requests"] == 4
    assert summary["budget"]["end_committed"]["tokens"] == 400
    assert summary["budget"]["limits"] == {
        "max_tokens": 200_000, "max_requests": 30, "max_elapsed_seconds": 5400.0
    }
    index = RunIndex(runtime / "run-index.json")
    assert len(index) == 4
    assert json.loads((runtime / "acceptance-summary.json").read_text())["verdict"] == "complete"


def test_fixed_runtime_dir_enforced(tmp_path, monkeypatch):
    """The one acceptance ledger/location is fixed; alternates fail closed."""
    repo, runtime = _repo(tmp_path, monkeypatch)
    with pytest.raises(AcceptanceError, match="acceptance_runtime_dir_fixed"):
        run_acceptance(
            repo, tmp_path / "elsewhere",
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=_fake_child(runtime, []),
        )


def test_reuse_incurs_zero_model_cost(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    calls: list = []
    run_acceptance(
        repo, runtime,
        algo_factory=lambda: FakeAlgo(VECTORS),
        child_runner=_fake_child(runtime, calls),
    )
    with BudgetLedger.open(runtime / ac.LEDGER_FILENAME) as ledger:
        snapshot = ledger.snapshot()
    assert snapshot["committed"]["requests"] == len(calls) == 4
    assert snapshot["open_reservations"] == []
    assert snapshot["stopped"] is None


def test_ask_mismatch_fails_closed_before_any_child(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    bad = [list(v) for v in VECTORS]
    bad[0][0] += 0.1
    calls: list = []
    with pytest.raises(AcceptanceError, match="outer_ask_mismatch"):
        run_acceptance(
            repo, runtime,
            algo_factory=lambda: FakeAlgo(bad),
            child_runner=_fake_child(runtime, calls),
        )
    assert calls == []
    checkpoint = json.loads((runtime / "acceptance-checkpoint.json").read_text())
    assert "outer_ask_mismatch" in checkpoint["reason"]


def test_missing_preflight_file_is_optional(tmp_path, monkeypatch):
    """Native derivation is the source of truth; the recorded preflight file
    may be absent (it is only a cross-check when present)."""
    repo, runtime = _repo(tmp_path, monkeypatch)
    (repo / ac.PREFLIGHT_VECTORS_FILE).unlink()
    calls: list = []
    summary = run_acceptance(
        repo, runtime,
        algo_factory=lambda: FakeAlgo(VECTORS),
        child_runner=_fake_child(runtime, calls),
    )
    assert summary["verdict"] == "complete"
    assert calls == ["cfg0", "cfg1", "cfg2", "cfg3"]


def test_second_ask_not_reproducible_fails_closed(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    algos = [FakeAlgo(VECTORS), FakeAlgo([[0.1, 1, 1, 1, 1]] * 4)]
    calls: list = []
    with pytest.raises(AcceptanceError, match="outer_ask_not_reproducible"):
        run_acceptance(
            repo, runtime,
            algo_factory=lambda: algos.pop(0),
            child_runner=_fake_child(runtime, calls),
        )
    assert calls == ["cfg0"]


def test_standalone_incomplete_stops_acceptance(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    calls: list = []
    behaviors = {"cfg0": {"status": "failed"}}
    with pytest.raises(AcceptanceError, match="standalone_run_incomplete"):
        run_acceptance(
            repo, runtime,
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=_fake_child(runtime, calls, behaviors=behaviors),
        )
    assert calls == ["cfg0"]
    assert not (runtime / "run-index.json").exists()


def test_result_divergence_between_return_and_file_fails(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)

    def diverging_child(spec, spec_path, *, repo_root):
        result = _fake_child(runtime, [])(spec, spec_path, repo_root=repo_root)
        result["best_score"] = 0.99  # returned payload != persisted file
        return result

    with pytest.raises(AcceptanceError, match="inner_run_result_diverged"):
        run_acceptance(
            repo, runtime,
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=diverging_child,
        )


def test_failed_outer_evaluation_maps_to_worst_fitness(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    algos = []

    def algo_factory():
        algo = FakeAlgo(VECTORS)
        algos.append(algo)
        return algo

    calls: list = []
    behaviors = {"cfg2": {"status": "failed"}}
    summary = run_acceptance(
        repo, runtime,
        algo_factory=algo_factory,
        child_runner=_fake_child(runtime, calls, behaviors=behaviors),
    )
    assert algos[1].told == [-0.5, -0.6, 0.0, -0.8]  # failed -> accuracy 0.0
    assert len(RunIndex(runtime / "run-index.json")) == 3  # failed not indexed
    assert summary["runs"][2]["status"] == "failed"
    assert summary["verdict"] != "complete"  # a failed run blocks completion


def test_missing_improve_crossover_marks_incomplete(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    calls: list = []
    no_ops = {"operator_counts": {"draft": 6, "improve": 0, "debug": 2, "crossover": 0}}
    behaviors = {f"cfg{i}": dict(no_ops) for i in range(4)}
    summary = run_acceptance(
        repo, runtime,
        algo_factory=lambda: FakeAlgo(VECTORS),
        child_runner=_fake_child(runtime, calls, behaviors=behaviors),
    )
    assert summary["verdict"] == "incomplete"
    assert summary["operator_coverage"]["improve"] == 0
    assert summary["operator_coverage"]["crossover"] == 0


def test_stopped_ledger_checkpoints_and_propagates(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    runtime.mkdir(parents=True, exist_ok=True)
    with BudgetLedger.open(
        runtime / ac.LEDGER_FILENAME, limits=ac.acceptance_budget_limits()
    ) as ledger:
        ledger.stop("budget_exhausted_elsewhere")
    calls: list = []
    with pytest.raises(LedgerStopped):
        run_acceptance(
            repo, runtime,
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=_fake_child(runtime, calls),
        )
    checkpoint = json.loads((runtime / "acceptance-checkpoint.json").read_text())
    assert "LedgerStopped" in checkpoint["reason"]
    assert checkpoint["budget"]["stopped"]["reason"] == "budget_exhausted_elsewhere"
    with BudgetLedger.open(runtime / ac.LEDGER_FILENAME) as ledger:
        assert ledger.snapshot()["stopped"]["reason"] == "budget_exhausted_elsewhere"


def test_run_id_mismatch_fails_closed(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)

    def bad_child(spec, spec_path, *, repo_root):
        result = _fake_child(runtime, [])(spec, spec_path, repo_root=repo_root)
        result["run_id"] = "wrong"
        (runtime / "runs" / spec.run_id / "run-result.json").write_text(
            json.dumps(result, sort_keys=True), encoding="utf-8"
        )
        return result

    with pytest.raises(AcceptanceError, match="inner_run_id_mismatch"):
        run_acceptance(
            repo, runtime,
            algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=bad_child,
        )


def test_decoded_configs_match_preflight_and_pass_gate():
    decoded = [decode_retained(v) for v in VECTORS]
    assert decoded[0] == {
        "crossover_prob": 0.7870834469795227,
        "score_weight": 0.3256833553314209,
        "delta_weight": 0.23071765899658203,
        "novelty_weight": 0.9376811981201172,
        "max_debug_depth": 1,
    }
    hashes = {config_identity_hash(d) for d in decoded}
    assert len(hashes) == 4


# --------------------------------------------------------------------------
# Deadline and emergency cleanup
# --------------------------------------------------------------------------

def test_ledger_deadline_property_uses_persisted_limits(tmp_path):
    limits = ac.acceptance_budget_limits()
    with BudgetLedger.open(tmp_path / "ledger.json", limits=limits) as ledger:
        expected = ledger.started_at + 5400.0
        assert ledger.deadline_epoch == expected
    # Reopen: the deadline is anchored at the ORIGINAL started_at, not now.
    time.sleep(0.05)
    with BudgetLedger.open(tmp_path / "ledger.json") as reopened:
        assert reopened.deadline_epoch == expected
        assert reopened.deadline_epoch < time.time() + 5400.0


def _write_live_registry(run_dir: Path, *, jobs=(), uncertain=(), intents=()):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "live-jobs.json").write_text(json.dumps({
        "schema": "acceptance-live-jobs.v1",
        "run_id": run_dir.name,
        "submission_intents": sorted(intents),
        "live_jobs": sorted(jobs),
        "uncertain_submissions": sorted(uncertain),
        "updated_at": time.time(),
    }), encoding="utf-8")


class _FakeCancelClient:
    def __init__(self, verified=True):
        self.verified = verified
        self.cancelled = []

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)
        return {"job_id": job_id, "cleanup_verified": self.verified}


def test_cleanup_child_jobs_verified(tmp_path):
    run_dir = tmp_path / "cfg0"
    _write_live_registry(run_dir, jobs=["job_a", "job_b"])
    client = _FakeCancelClient(verified=True)
    evidence = cleanup_child_jobs(run_dir, repo_root=tmp_path, client_factory=lambda: client)
    assert evidence["cleanup_verified"] is True
    assert client.cancelled == ["job_a", "job_b"]
    assert json.loads((run_dir / "emergency-cleanup.json").read_text())["cleanup_verified"] is True


def test_cleanup_child_jobs_unproven_cancel_fails_closed(tmp_path):
    run_dir = tmp_path / "cfg0"
    _write_live_registry(run_dir, jobs=["job_a"])
    client = _FakeCancelClient(verified=False)
    evidence = cleanup_child_jobs(run_dir, repo_root=tmp_path, client_factory=lambda: client)
    assert evidence["cleanup_verified"] is False
    assert evidence["reason"] == "child_cleanup_unproven"


def test_cleanup_unknown_submission_identity_never_clean(tmp_path):
    run_dir = tmp_path / "cfg0"
    _write_live_registry(run_dir, uncertain=["trace-1"], intents=["trace-2"])
    client = _FakeCancelClient()
    evidence = cleanup_child_jobs(run_dir, repo_root=tmp_path, client_factory=lambda: client)
    assert evidence["cleanup_verified"] is False
    assert len(evidence["unknown_identities"]) == 2
    assert client.cancelled == []  # nothing to cancel without a job id


def test_cleanup_missing_registry_fails_closed(tmp_path):
    evidence = cleanup_child_jobs(tmp_path / "cfg0", repo_root=tmp_path,
                                  client_factory=lambda: _FakeCancelClient())
    assert evidence["cleanup_verified"] is False
    assert evidence["reason"] == "live_job_registry_missing"


def test_cleanup_empty_registry_verified_without_calls(tmp_path):
    run_dir = tmp_path / "cfg0"
    _write_live_registry(run_dir)
    client = _FakeCancelClient()
    evidence = cleanup_child_jobs(run_dir, repo_root=tmp_path, client_factory=lambda: client)
    assert evidence["cleanup_verified"] is True
    assert client.cancelled == []


class _HangingProcess:
    """Fake child process that never finishes (drives the timeout path)."""

    def __init__(self):
        self.killed = False
        self.timeout_seen = None

    def communicate(self, timeout=None):
        if self.killed:
            return ("", "")
        self.timeout_seen = timeout
        raise subprocess.TimeoutExpired("child", timeout)

    def kill(self):
        self.killed = True


def test_child_timeout_kills_only_child_and_cancels_recorded_jobs(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    runtime.mkdir(parents=True, exist_ok=True)
    with BudgetLedger.open(runtime / ac.LEDGER_FILENAME, limits=ac.acceptance_budget_limits()):
        pass
    evo_python = repo / "OpenMLE-Evo" / ".venv" / "bin" / "python"
    evo_python.parent.mkdir(parents=True)
    evo_python.write_text("#!/bin/sh\n")
    run_dir = runtime / "runs" / "cfg0"
    _write_live_registry(run_dir, jobs=["job_live_1"])
    process = _HangingProcess()
    monkeypatch.setattr(ap.subprocess, "Popen", lambda *a, **k: process)
    client = _FakeCancelClient(verified=True)

    spec = ap.InnerRunSpec(
        run_id="cfg0",
        run_dir=(ac.RUNTIME_DIR / "runs" / "cfg0").as_posix(),
        decoded=decode_retained(VECTORS[0]),
        ledger_path=(ac.RUNTIME_DIR / ac.LEDGER_FILENAME).as_posix(),
        runner_commit=COMMIT,
    )
    spec_path = runtime / "runs" / "cfg0" / "run-spec.json"
    with pytest.raises(AcceptanceError, match="child_deadline_exceeded"):
        ap._default_child_runner(
            spec, spec_path, repo_root=repo,
            cleanup_client_factory=lambda: client,
        )
    assert process.killed is True
    assert client.cancelled == ["job_live_1"]
    # the subprocess timeout is the SHARED ledger deadline + bounded grace
    assert process.timeout_seen is not None
    remaining = json.loads((runtime / ac.LEDGER_FILENAME).read_text())
    started_at = remaining["started_at"]
    expected = started_at + 5400.0 - time.time() + ap.CHILD_CLEANUP_GRACE_SECONDS
    assert abs(process.timeout_seen - expected) < 5.0
    evidence = json.loads((run_dir / "emergency-cleanup.json").read_text())
    assert evidence["cleanup_verified"] is True


def test_child_runner_refuses_after_deadline_without_spawning(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    runtime.mkdir(parents=True, exist_ok=True)
    with BudgetLedger.open(
        runtime / ac.LEDGER_FILENAME,
        limits=type(ac.acceptance_budget_limits())(
            max_tokens=200_000, max_requests=30, max_elapsed_seconds=0.05
        ),
    ):
        pass
    time.sleep(0.1)
    evo_python = repo / "OpenMLE-Evo" / ".venv" / "bin" / "python"
    evo_python.parent.mkdir(parents=True)
    evo_python.write_text("#!/bin/sh\n")
    spawned = []
    monkeypatch.setattr(ap.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    spec = ap.InnerRunSpec(
        run_id="cfg0",
        run_dir=(ac.RUNTIME_DIR / "runs" / "cfg0").as_posix(),
        decoded=decode_retained(VECTORS[0]),
        ledger_path=(ac.RUNTIME_DIR / ac.LEDGER_FILENAME).as_posix(),
        runner_commit=COMMIT,
    )
    with pytest.raises(AcceptanceError, match="acceptance_deadline_exceeded"):
        ap._default_child_runner(spec, runtime / "runs" / "cfg0" / "run-spec.json", repo_root=repo)
    assert spawned == []
