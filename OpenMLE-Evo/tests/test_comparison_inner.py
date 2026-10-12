"""EXP-005 offline integration: PRODUCTION comparison runner with REAL Evo.

Drives research.comparison.runner.run_inner with the REAL Evolutionary
solver, REAL GenericLLM operators from the vendored dojo YAML templates and
the REAL comparison ledger — only the litellm streaming completion_fn and
the sandbox HTTP layer are deterministic fakes (trusted evaluator evidence
is computed truthfully by the fake service). No model, network or sandbox
calls; fixture ledgers live under tmp_path; the live review gate is
monkeypatched away BELOW the production entry (the gate itself is covered
by research/tests/test_comparison_runner.py).
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import pytest

from dojo.solvers.evo.evo import Evolutionary
from dojo.utils.logger import config_logger

from research.comparison import runner as cr
from research.comparison.budget import (
    ComparisonBudgetError,
    ComparisonLedger,
    ProviderUsage,
)
from research.comparison.data import load_task_bindings
from research.comparison.protocol import FIXED_THETA, ModelProfile, decode_theta
from research.comparison.runner import ComparisonRunSpec, run_dir_name
from research.comparison.specs import iter_cells
from research.evaluator.service import EvaluatorRegistry
from research.comparison import deploy as ac
from research.search.credentials import ManagedKimiCredentials

REPO_ROOT = Path(__file__).resolve().parents[2]
LITELLM_MODULE = "dojo.core.solvers.llm_helpers.backends.lite_llm"
COMMIT = "c" * 40
TOKEN = "offline-" + "comparison-" + "token-456"

import httpx as _httpx  # noqa: E402

#: The REAL httpx.Client, captured before any test monkeypatching so nested
#: fake factories never wrap each other (multiple runs per test).
_REAL_HTTPX_CLIENT = _httpx.Client


def _cells():
    return iter_cells(load_task_bindings(), COMMIT)


def _cell_index(task_id: str, seed: int = 11, method: str = "FIXED") -> int:
    cells = _cells()
    key = f"{task_id}:{seed}:{method}"
    return next(i for i, c in enumerate(cells) if c.cell_key() == key)


class _Stream:
    """Deterministic streaming fake: content chunks + optional usage chunk."""

    def __init__(self, content: str, usage: dict | None):
        self._content = content
        self._usage = usage

    def __iter__(self):
        midpoint = max(1, len(self._content) // 2)
        for part in (self._content[:midpoint], self._content[midpoint:]):
            yield {"choices": [{"delta": {"content": part}}]}
        if self._usage is not None:
            yield {"usage": dict(self._usage), "choices": []}

    def close(self):
        pass


def _make_completion(calls: list, *, usage: dict | None = None,
                     empty_calls: set[int] | None = None):
    """Fake completion_fn; marker ``call-<n>`` lets the fake sandbox fail
    specific provider calls deterministically."""
    usage = {"prompt_tokens": 10, "completion_tokens": 20} if usage is None else usage

    def completion(**kwargs):
        calls.append(kwargs)
        marker = len(calls)
        assert kwargs.get("stream") is True
        assert kwargs.get("max_retries") == 0 and kwargs.get("num_retries") == 0
        if empty_calls and marker in empty_calls:
            return _Stream(f"Plan {marker} with no code at all.", usage)
        code = (
            "import pandas as pd\n"
            f"MARKER = \"call-{marker}\"\n"
            "print('ok')\n"
        )
        return _Stream(f"Plan {marker}.\n```python\n{code}```\n", usage)

    return completion


def _expected_score(task_id: str) -> float:
    """Truthful fixture score: accuracy of the sample_submission prediction
    against the trusted validation answer (controller-side only)."""
    registry = EvaluatorRegistry.load()
    spec = registry.task(f"comparison_{task_id}")
    answer = [
        int(line.split(",")[1])
        for line in spec.answer_path.read_text().splitlines()[1:]
        if line.strip()
    ]
    sample = (
        REPO_ROOT / "tasks" / "comparison" / task_id / "data" / "public"
        / "sample_submission.csv"
    ).read_text()
    prediction = [
        int(line.split(",")[1]) for line in sample.splitlines()[1:] if line.strip()
    ]
    assert len(answer) == len(prediction)
    return sum(1 for a, p in zip(answer, prediction) if a == p) / len(answer)


def _make_sandbox(base: Path, task_id: str, submissions: list,
                  fail_markers: frozenset[str] = frozenset(),
                  job_counter: dict | None = None):
    """Trusted sandbox fake: truthfully computes the source-gate/evaluator
    evidence for the registered comparison task (score = real accuracy of
    the submitted prediction snapshot against the trusted validation
    answer), with immutable prediction snapshots under the dispatcher
    storage root."""
    import httpx as _httpx

    from research.contracts.source_gate import check_source_bytes, load_policy

    registry = EvaluatorRegistry.load()
    spec = registry.task(f"comparison_{task_id}")
    answer_rows = max(0, len(spec.answer_path.read_text().splitlines()) - 1)
    score = _expected_score(task_id)
    sample = (REPO_ROOT / "tasks" / "comparison" / task_id
              / "data" / "public" / "sample_submission.csv").read_bytes()
    prediction_hash = hashlib.sha256(sample).hexdigest()
    jobs: dict = {}
    counter = job_counter if job_counter is not None else {"n": 0}

    def service(request):
        if request.method == "POST" and request.url.path == "/api/v1/jobs":
            payload = json.loads(request.content)
            submissions.append(payload)
            counter["n"] += 1
            job_id = "job_%016x" % counter["n"]
            code = payload["code"]
            failed_marker = next(
                (m for m in fail_markers if f'"{m}"' in code), None
            )
            storage_dir = (
                base / ".runtime/rsi-trustworthy/storage/mlsandbox/jobs" / job_id
            )
            storage_dir.mkdir(parents=True, exist_ok=True)
            (storage_dir / "prediction_snapshot.csv").write_bytes(sample)
            if failed_marker is not None:
                jobs[job_id] = {
                    "job_id": job_id, "status": "failed",
                    "created_at": "2026-10-11T00:00:00Z",
                    "completed_at": "2026-10-11T00:00:01Z",
                    "result": {
                        "result": "error",
                        "run_log": f"Traceback\nRuntimeError: {failed_marker} boom",
                        "source_identity_verified": True,
                        "worker_cleanup_verified": True,
                    },
                }
            else:
                source = code.encode()
                raw_hash = hashlib.sha256(source).hexdigest()
                tree_hash = hashlib.sha256(
                    b"main.py\0" + raw_hash.encode() + b"\n"
                ).hexdigest()
                gate = check_source_bytes(source, load_policy()).to_dict()
                gate.update(source_sha256=tree_hash, entrypoint="main.py",
                            entrypoint_sha256=raw_hash)
                jobs[job_id] = {
                    "job_id": job_id, "status": "completed",
                    "created_at": "2026-10-11T00:00:00Z",
                    "completed_at": "2026-10-11T00:00:02Z",
                    "result": {
                        "score": score, "result": "success", "source_gate": gate,
                        "source_identity_verified": True,
                        "worker_cleanup_verified": True,
                        "evaluation": {
                            "score": score, "split": "validation",
                            "metric": "accuracy",
                            "task_id": f"comparison_{task_id}",
                            "direction": "maximize",
                            "evaluator": "external_trusted",
                            "row_count": answer_rows,
                            "answer_sha256": spec.answer_sha256,
                            "registry_version": registry.registry_version,
                            "prediction_sha256": prediction_hash,
                            "prediction_snapshot": (
                                f"/mnt/rsi_storage/mlsandbox/jobs/{job_id}/prediction_snapshot.csv"
                            ),
                        },
                    },
                }
            return _httpx.Response(200, json={"job_id": job_id})
        if request.method == "GET":
            return _httpx.Response(200, json=jobs[request.url.path.split("/")[-1]])
        raise AssertionError(request.method)

    def factory(*args, **kwargs):
        kwargs["transport"] = _httpx.MockTransport(service)
        return _REAL_HTTPX_CLIENT(*args, **kwargs)

    return factory


def _run_inner_offline(base: Path, monkeypatch, *, task_id: str,
                       config_index: int = 0, seed: int = 11,
                       theta: tuple = FIXED_THETA,
                       fail_markers: frozenset[str] = frozenset(),
                       completion_usage: dict | None = None,
                       empty_calls: set[int] | None = None,
                       tamper_profile: bool = False,
                       tamper_cleanup: bool = False,
                       ledger_precommit=None,
                       shared_counter: dict | None = None):
    """Drive the PRODUCTION run_inner with fake provider/sandbox HTTP only."""
    import httpx as _httpx

    base.mkdir(parents=True, exist_ok=True)
    evo_link = base / "OpenMLE-Evo"
    if not evo_link.exists():
        evo_link.symlink_to(REPO_ROOT / "OpenMLE-Evo", target_is_directory=True)
    tasks_link = base / "tasks"
    if not tasks_link.exists():
        tasks_link.symlink_to(REPO_ROOT / "tasks", target_is_directory=True)
    monkeypatch.setenv("LOGGING_DIR", str(base / "logs"))
    monkeypatch.setattr(cr, "check_comparison_review", lambda root: None)
    monkeypatch.setattr(cr.ac, "git_head_commit", lambda root: COMMIT)
    auth = base / ac.AUTH_FILE
    auth.parent.mkdir(parents=True, exist_ok=True)
    if not auth.exists():
        auth.write_text("SANDBOX_API_KEYS=ONLY_SYNTHETIC_SANDBOX_KEY\n", encoding="utf-8")
        auth.chmod(0o600)
    credentials_file = base / "fake-credentials.json"
    credentials_file.write_text(
        json.dumps({"access_token": TOKEN, "expires_at": time.time() + 3600}),
        encoding="utf-8",
    )
    provider = ManagedKimiCredentials(credentials_file)
    monkeypatch.setattr(cr, "ManagedKimiCredentials", lambda: provider)

    calls: list = []
    monkeypatch.setattr(
        f"{LITELLM_MODULE}.completion_fn",
        _make_completion(calls, usage=completion_usage, empty_calls=empty_calls),
    )

    submissions: list = []
    monkeypatch.setattr(
        _httpx, "Client",
        _make_sandbox(base, task_id, submissions, fail_markers, shared_counter),
    )

    cells = _cells()
    cell = cells[_cell_index(task_id, seed)]
    run = cell.inner_run(config_index)
    ledger_rel = Path(".runtime/comparison-istratde/comparison-ledger.json")
    ledger_path = base / ledger_rel
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if not ledger_path.exists():
        ledger = ComparisonLedger.open(ledger_path, cells=cells)
        ledger.start_clock()
        if ledger_precommit is not None:
            ledger_precommit(ledger)
        ledger.close()

    if tamper_profile:
        real_builder = cr.build_comparison_solver

        def tampered_builder(spec, *, repo_root):
            solver = real_builder(spec, repo_root=repo_root)
            llm = solver.draft_fn.args[0]
            llm.generation_kwargs["temperature"] = 0.5
            return solver

        monkeypatch.setattr(cr, "build_comparison_solver", tampered_builder)

    if tamper_cleanup:
        import research.adapters.sandbox_eval_client as sec

        real_client = sec.SandboxEvalClient

        class UncleanClient(real_client):
            def cancel_all_live(self):
                return [{"cleanup_verified": False}]

        monkeypatch.setattr(sec, "SandboxEvalClient", UncleanClient)

    spec = ComparisonRunSpec(
        run=run,
        decoded=decode_theta(theta),
        run_dir=str(Path(".runtime/comparison-istratde/runs") / run_dir_name(run)),
        ledger_path=str(ledger_rel),
        runner_commit=COMMIT,
    )
    spec_path = base / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    code = cr.run_inner(spec_path, repo_root=base)
    run_dir = base / spec.run_dir
    result = None
    result_path = run_dir / "run-result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
    return {
        "exit": code, "result": result, "run_dir": run_dir, "calls": calls,
        "submissions": submissions, "ledger_path": ledger_path, "run": run,
        "spec": spec,
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _ledger_view(path: Path) -> dict:
    ledger = ComparisonLedger.open(path, cells=_cells())
    try:
        return ledger.snapshot()
    finally:
        ledger.close()


# ------------------------------------------------------------------ assembly
def test_build_solver_real_assembly_matches_comparison_pins():
    spec = ComparisonRunSpec(
        run=_cells()[0].inner_run(0),
        decoded=decode_theta(FIXED_THETA),
        run_dir=".runtime/comparison-istratde/runs/banknote-11-FIXED-0",
        ledger_path=".runtime/comparison-istratde/comparison-ledger.json",
        runner_commit=COMMIT,
    )
    solver = cr.build_comparison_solver(spec, repo_root=REPO_ROOT)
    assert isinstance(solver, Evolutionary)
    cfg = solver.cfg
    assert cfg.num_generations == 3
    assert cfg.individuals_per_generation == 2
    assert cfg.step_limit == 12
    assert cfg.num_islands == 1
    assert cfg.execution_mode == "generation"
    assert cfg.max_llm_call_retries == 1
    assert cfg.crossover_prob == pytest.approx(0.5)
    assert cfg.max_debug_depth == 1  # FIXED theta depth
    weights = cfg.experience["parent_selection"]["weights"]
    assert weights["score"] == pytest.approx(1.0)
    assert weights["delta"] == pytest.approx(0.4)
    assert weights["novelty"] == pytest.approx(0.25)
    assert cfg.experience["enabled"] is True
    assert cfg.experience["prompt_memory"]["enabled"] is False
    for name in ("draft", "improve", "debug", "crossover", "analyze"):
        llm = getattr(solver, name + "_fn").args[0]
        assert llm.client.api_key == ac.API_KEY_PLACEHOLDER
        kwargs = llm.generation_kwargs
        assert kwargs["stream"] is True
        assert kwargs["max_tokens"] == 8192
        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["temperature"] == 1.0
        assert kwargs["top_p"] == 0.95
        assert kwargs["allowed_openai_params"] == ["reasoning_effort"]
    # Generic train.csv/test.csv program contract with the task feature count.
    assert "train.csv" in solver.task_info["data_description"]
    assert "test.csv" in solver.task_info["data_description"]
    assert "f4" in solver.task_info["data_description"]  # banknote: 4 features
    assert "f5,label" not in solver.task_info["data_description"]
    assert "Banknote Authentication" in solver.task_info["task_description"]


def test_task_info_feature_counts_all_tasks():
    for task_id, n in (("banknote", 4), ("wdbc", 30), ("spambase", 57)):
        binding = load_task_bindings()[task_id]
        info = cr.task_info_for(binding)
        assert f"f{n}" in info["data_description"]
        assert f"f{n + 1}" not in info["data_description"]
        assert "submission.csv" in info["public_user_prompt"]


# ---------------------------------------------------- happy path, all tasks
@pytest.mark.parametrize("task_id", ["banknote", "wdbc", "spambase"])
def test_run_completes_exact_counts_per_task(tmp_path, monkeypatch, task_id):
    config_logger(None)
    outcome = _run_inner_offline(tmp_path / task_id, monkeypatch, task_id=task_id)
    assert outcome["exit"] == 0
    result = outcome["result"]
    assert result["status"] == "completed"
    assert result["termination"] == "completed"
    # Exactly 6 main requests, no debug (all candidates scored), 3 generations.
    assert result["requests"] == {"main": 6, "debug": 0, "charged_tokens": 180}
    assert result["main_candidates"] == 6
    assert result["generations_completed"] == 3
    assert result["incumbent"]["validation_score"] == pytest.approx(
        _expected_score(task_id)
    )
    assert len(result["incumbent"]["code_sha256"]) == 64
    assert result["worker_cleanup_verified"] is True
    # Dataset/spec/quota identities bound to the run.
    binding = load_task_bindings()[task_id]
    assert result["dataset"]["task_binding_identity"] == binding.identity()
    assert result["dataset"]["registry_task_id"] == f"comparison_{task_id}"
    assert result["run_identity"] == outcome["run"].identity()
    assert result["inner_seed"] == outcome["run"].inner_seed
    assert result["model_profile_identity"] == ModelProfile().identity()
    # Every job carries trusted evaluator evidence on the validation split.
    for job in result["jobs"]:
        assert job["evidence_verified"] is True
        assert job["evaluation"]["task_id"] == f"comparison_{task_id}"
        assert job["evaluation"]["split"] == "validation"
        assert job["evaluation"]["row_count"] > 0
    # Prediction snapshots bound byte-for-byte into the run dir.
    assert len(result["prediction_artifacts"]) == 6
    for job_id, digest in result["prediction_artifacts"].items():
        stored = outcome["run_dir"] / "predictions" / f"{job_id}.csv"
        assert hashlib.sha256(stored.read_bytes()).hexdigest() == digest
    # The sandbox submissions used the pinned task identity and grant path.
    assert {s["task_id"] for s in outcome["submissions"]} == {f"comparison_{task_id}"}
    assert {s["data_dir"] for s in outcome["submissions"]} == {
        f"/mnt/rsi_data/comparison/{task_id}/data/public"
    }
    # Provider attempts: exactly 6, pinned profile, ledger-linked.
    assert len(outcome["calls"]) == 6
    trace = _read_jsonl(outcome["run_dir"] / "request-trace.jsonl")
    assert len(trace) == 6
    assert {t["kind"] for t in trace} == {"main"}
    for record in trace:
        assert record["request"]["stream"] is True
        assert record["request"]["max_tokens"] == 8192
        assert record["request"]["reasoning_effort"] == "low"
        assert "api_key" not in json.dumps(record)
    # Credential hygiene: token appears in no run artifact.
    for artifact in outcome["run_dir"].rglob("*"):
        if artifact.is_file():
            assert TOKEN not in artifact.read_bytes().decode("utf-8", "replace")
    # Ledger: cell committed exactly this run's 6 requests; nothing stopped.
    snapshot = _ledger_view(outcome["ledger_path"])
    assert snapshot["stopped"] is None
    cell_view = snapshot["cells"][f"{task_id}:11:FIXED"]
    assert cell_view["committed"] == {"requests": 6, "tokens": 180}


def test_paired_inner_seed_reproduces_operator_routing(tmp_path, monkeypatch):
    """Same paired inner seed -> identical operator sequence and code hashes;
    the seed is method-independent (same position in another method matches)."""
    config_logger(None)
    a = _run_inner_offline(tmp_path / "a", monkeypatch, task_id="banknote")
    b = _run_inner_offline(tmp_path / "b", monkeypatch, task_id="banknote",
                           seed=22)
    assert a["run"].inner_seed != b["run"].inner_seed
    c = _run_inner_offline(tmp_path / "c", monkeypatch, task_id="banknote")

    def trace_ops(run_dir):
        return [
            (event["operator"], event["child_code_sha256"])
            for event in _read_jsonl(run_dir / "operator-trace.jsonl")
        ]

    assert trace_ops(a["run_dir"]) == trace_ops(c["run_dir"])
    assert len(trace_ops(a["run_dir"])) == 6
    # Method-independent paired seed: FIXED and ISTRATDE at the same
    # (task, seed, config) position share the derived inner seed.
    cells = _cells()
    fixed = cells[_cell_index("banknote", 11, "FIXED")].inner_run(0)
    istratde = cells[_cell_index("banknote", 11, "ISTRATDE")].inner_run(0)
    assert fixed.inner_seed == istratde.inner_seed


def test_fresh_state_across_configs(tmp_path, monkeypatch):
    """Two configs of one cell: fresh memory-off searches, no shared state."""
    config_logger(None)
    first = _run_inner_offline(tmp_path / "cell", monkeypatch, task_id="wdbc",
                               config_index=0)
    second = _run_inner_offline(tmp_path / "cell", monkeypatch, task_id="wdbc",
                                config_index=1, shared_counter={"n": 100})
    assert first["result"]["status"] == second["result"]["status"] == "completed"
    assert first["run"].inner_seed != second["run"].inner_seed
    snapshot = _ledger_view(first["ledger_path"])
    cell_view = snapshot["cells"]["wdbc:11:FIXED"]
    assert cell_view["committed"]["requests"] == 12
    # Independent journals per run dir.
    j1 = (first["run_dir"] / "journal.jsonl").read_bytes()
    j2 = (second["run_dir"] / "journal.jsonl").read_bytes()
    assert j1 != j2


# --------------------------------------------------------------------- debug
def test_depth1_repairs_two_distinct_candidates_capped_at_two_total(
    tmp_path, monkeypatch
):
    """FIXED theta depth 1: two different failed mains repaired once each;
    a third failure is SKIPPED (total 2), never repaired twice per chain."""
    config_logger(None)
    outcome = _run_inner_offline(
        tmp_path / "d1", monkeypatch, task_id="banknote",
        fail_markers=frozenset({"call-1", "call-3", "call-5"}),
    )
    result = outcome["result"]
    assert result["status"] == "completed"
    assert result["requests"]["main"] == 6
    assert result["requests"]["debug"] == 2
    events = _read_jsonl(outcome["run_dir"] / "operator-trace.jsonl")
    debug_events = [e for e in events if e["operator"] == "debug"]
    assert len(debug_events) == 2
    # One repair per chain (depth 1), two DISTINCT failed parents.
    parents = {tuple(e["parent_node_ids"]) for e in debug_events}
    assert len(parents) == 2
    # The third failed candidate was skipped with the recorded reason.
    control_log = _read_jsonl(outcome["run_dir"] / "budget-control.jsonl")
    skips = [e for e in control_log if e["event"] == "debug_skipped"]
    assert skips and "inner_debug_budget_exhausted" in skips[-1]["reason"]
    # Incumbent comes from a repaired candidate; failures stayed separate.
    assert result["incumbent"] is not None
    assert result["incumbent"]["operators_used"] == ["debug"]
    failed_jobs = [j for j in result["jobs"] if j["status"] == "failed"]
    assert len(failed_jobs) == 3


def test_depth2_chain_uses_two_slots_then_denies_second_chain(
    tmp_path, monkeypatch
):
    """Theta depth 2: one chain may spend both slots (two repairs of the SAME
    candidate); the next buggy candidate's chain is denied at the total cap."""
    config_logger(None)
    theta = (0.5, 1.0, 0.4, 0.25, 2.0)
    outcome = _run_inner_offline(
        tmp_path / "d2", monkeypatch, task_id="banknote", theta=theta,
        fail_markers=frozenset({"call-1", "call-2", "call-4"}),
    )
    result = outcome["result"]
    assert result["status"] == "completed"
    assert result["requests"]["main"] == 6
    assert result["requests"]["debug"] == 2
    events = _read_jsonl(outcome["run_dir"] / "operator-trace.jsonl")
    debug_events = [e for e in events if e["operator"] == "debug"]
    # Both debugs belong to ONE chain (same original parent chain: the
    # second repair's parent is the first repair node).
    assert len(debug_events) == 2
    first_parent = debug_events[0]["parent_node_ids"][0]
    assert debug_events[1]["parent_node_ids"][0] == debug_events[0]["child_node_id"]
    assert first_parent != debug_events[0]["child_node_id"]
    control_log = _read_jsonl(outcome["run_dir"] / "budget-control.jsonl")
    skips = [e for e in control_log if e["event"] == "debug_skipped"]
    assert skips and "inner_debug_budget_exhausted" in skips[-1]["reason"]
    assert result["incumbent"]["operators_used"] == ["debug"]


def test_theta_depth_zero_disables_debug(tmp_path, monkeypatch):
    config_logger(None)
    theta = (0.5, 1.0, 0.4, 0.25, 0.0)
    outcome = _run_inner_offline(
        tmp_path / "d0", monkeypatch, task_id="banknote", theta=theta,
        fail_markers=frozenset({"call-1"}),
    )
    result = outcome["result"]
    assert result["requests"]["debug"] == 0
    assert result["debug_budget"] == 0
    control_log = _read_jsonl(outcome["run_dir"] / "budget-control.jsonl")
    skips = [e for e in control_log if e["event"] == "debug_skipped"]
    assert skips and skips[0]["reason"] == "debug_disabled:theta_max_debug_depth_0"


# ---------------------------------------------------- failure classification
def test_malformed_generation_never_fabricates_code(tmp_path, monkeypatch):
    config_logger(None)
    outcome = _run_inner_offline(
        tmp_path / "bad", monkeypatch, task_id="banknote",
        empty_calls={1, 2, 3, 4, 5, 6, 7, 8},
    )
    result = outcome["result"]
    assert result["status"] == "completed"
    assert result["requests"]["main"] == 6
    assert result["requests"]["debug"] == 2  # optional debug also extracted empty
    assert result["incumbent"] is None
    assert result["jobs"] == []  # no empty candidate ever reached the sandbox
    outcomes = _read_jsonl(outcome["run_dir"] / "generation-outcomes.jsonl")
    assert len(outcomes) == 8
    assert all(o["classification"] != "ok" for o in outcomes)
    assert not list((outcome["run_dir"] / "generated").glob("*.py"))


def test_unknown_provider_usage_stops_globally_keeps_incumbent(
    tmp_path, monkeypatch
):
    """The third attempt streams WITHOUT provider usage: unknown usage must
    retain the full reservation, stop the ledger globally, and the two
    already-scored candidates remain the truthful incumbent evidence."""
    config_logger(None)

    class _UsageMissingStream(_Stream):
        def __iter__(self):
            midpoint = max(1, len(self._content) // 2)
            for part in (self._content[:midpoint], self._content[midpoint:]):
                yield {"choices": [{"delta": {"content": part}}]}
            if self._usage is None:  # no usage chunk at all
                return
            yield {"usage": dict(self._usage), "choices": []}

    calls: list = []

    def completion(**kwargs):
        calls.append(kwargs)
        marker = len(calls)
        code = f"import pandas as pd\nMARKER = \"call-{marker}\"\nprint('ok')\n"
        usage = None if marker == 3 else {"prompt_tokens": 10, "completion_tokens": 20}
        return _UsageMissingStream(f"Plan {marker}.\n```python\n{code}```\n", usage)

    outcome = _run_inner_offline_with_completion(
        tmp_path / "unknown", monkeypatch, task_id="banknote",
        completion=completion, calls=calls,
    )
    result = outcome["result"]
    assert result["status"] == "failed"
    assert "usage" in result["termination"]
    assert result["incumbent"] is not None  # retained despite the later stop
    assert result["incumbent"]["validation_score"] == pytest.approx(
        _expected_score("banknote")
    )
    snapshot = _ledger_view(outcome["ledger_path"])
    assert snapshot["stopped"]["reason"].startswith("unknown_usage:")
    assert snapshot["global"]["committed"]["requests"] == 3
    charged = [r for r in snapshot["requests"] if not r["usage_known"]]
    assert len(charged) == 1
    assert charged[0]["charged_tokens"] == charged[0]["reserved_tokens"]
    # A stopped ledger refuses any further run (fail closed before
    # activity); a rerun of the SAME config is refused as not-fresh first.
    with pytest.raises(cr.ComparisonRunnerError, match="run_dir_not_fresh"):
        _run_inner_offline_with_completion(
            tmp_path / "unknown", monkeypatch, task_id="banknote",
            completion=completion, calls=calls, reuse_base=True,
        )
    with pytest.raises(cr.ComparisonRunnerError, match="comparison_ledger_stopped"):
        _run_inner_offline_with_completion(
            tmp_path / "unknown", monkeypatch, task_id="banknote",
            completion=completion, calls=calls, reuse_base=True, config_index=1,
        )


def _run_inner_offline_with_completion(base, monkeypatch, *, task_id, completion,
                                       calls, reuse_base=False,
                                       config_index=0, **kwargs):
    """Variant of _run_inner_offline with a caller-supplied completion_fn."""
    import httpx as _httpx

    if not reuse_base:
        base.mkdir(parents=True, exist_ok=True)
        evo_link = base / "OpenMLE-Evo"
        if not evo_link.exists():
            evo_link.symlink_to(REPO_ROOT / "OpenMLE-Evo", target_is_directory=True)
        tasks_link = base / "tasks"
        if not tasks_link.exists():
            tasks_link.symlink_to(REPO_ROOT / "tasks", target_is_directory=True)
    monkeypatch.setenv("LOGGING_DIR", str(base / "logs"))
    monkeypatch.setattr(cr, "check_comparison_review", lambda root: None)
    monkeypatch.setattr(cr.ac, "git_head_commit", lambda root: COMMIT)
    auth = base / ac.AUTH_FILE
    auth.parent.mkdir(parents=True, exist_ok=True)
    if not auth.exists():
        auth.write_text("SANDBOX_API_KEYS=ONLY_SYNTHETIC_SANDBOX_KEY\n", encoding="utf-8")
        auth.chmod(0o600)
    credentials_file = base / "fake-credentials.json"
    credentials_file.write_text(
        json.dumps({"access_token": TOKEN, "expires_at": time.time() + 3600}),
        encoding="utf-8",
    )
    provider = ManagedKimiCredentials(credentials_file)
    monkeypatch.setattr(cr, "ManagedKimiCredentials", lambda: provider)
    monkeypatch.setattr(f"{LITELLM_MODULE}.completion_fn", completion)
    submissions: list = []
    monkeypatch.setattr(
        _httpx, "Client", _make_sandbox(base, task_id, submissions)
    )
    cells = _cells()
    cell = cells[_cell_index(task_id, 11)]
    run = cell.inner_run(config_index)
    ledger_rel = Path(".runtime/comparison-istratde/comparison-ledger.json")
    ledger_path = base / ledger_rel
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if not ledger_path.exists():
        ledger = ComparisonLedger.open(ledger_path, cells=cells)
        ledger.start_clock()
        ledger.close()
    spec = ComparisonRunSpec(
        run=run,
        decoded=decode_theta(FIXED_THETA),
        run_dir=str(Path(".runtime/comparison-istratde/runs") / run_dir_name(run)),
        ledger_path=str(ledger_rel),
        runner_commit=COMMIT,
    )
    spec_path = base / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    code = cr.run_inner(spec_path, repo_root=base)
    run_dir = base / spec.run_dir
    result_path = run_dir / "run-result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    return {"exit": code, "result": result, "run_dir": run_dir,
            "ledger_path": ledger_path, "submissions": submissions}


def test_profile_drift_refused_before_any_reservation(tmp_path, monkeypatch):
    config_logger(None)
    outcome = _run_inner_offline(
        tmp_path / "drift", monkeypatch, task_id="banknote", tamper_profile=True,
    )
    result = outcome["result"]
    assert result["status"] == "failed"
    assert "request_profile_mismatch:temperature" in result["termination"]
    snapshot = _ledger_view(outcome["ledger_path"])
    assert snapshot["global"]["committed"]["requests"] == 0
    assert snapshot["stopped"] is not None  # off-profile attempt stops globally
    assert outcome["submissions"] == []  # nothing reached the sandbox


def test_cell_token_exhaustion_terminates_cell_other_cell_continues(
    tmp_path, monkeypatch
):
    """Cell cap: truthful cell termination, no global stop, no incumbent
    fabrication; another cell still runs (shared ledger, real reopen)."""
    config_logger(None)

    def drain(ledger):
        # Committed cell tokens reach 190960 (actual usage, each exactly at
        # its reservation); the next reservation crosses 200000 -> truthful
        # cell termination, no global stop.
        cell = _cells()[_cell_index("banknote", 11)]
        other = cell.inner_run(1)
        for _ in range(5):
            r = ledger.reserve(other, "main", estimated_input_tokens=30000)
            ledger.commit(r.request_id, ProviderUsage(30000, 8192))

    outcome = _run_inner_offline(
        tmp_path / "cap", monkeypatch, task_id="banknote",
        ledger_precommit=drain,
    )
    result = outcome["result"]
    assert result["status"] == "cell_terminated"
    assert "cell_tokens_exhausted" in result["termination"]
    assert result["incumbent"] is None
    snapshot = _ledger_view(outcome["ledger_path"])
    assert snapshot["stopped"] is None
    assert snapshot["cells"]["banknote:11:FIXED"]["terminated"]["reason"] == (
        "cell_tokens_exhausted"
    )
    # Another cell on the SAME ledger continues normally.
    outcome_b = _run_inner_offline(
        tmp_path / "cap", monkeypatch, task_id="wdbc", shared_counter={"n": 500},
    )
    assert outcome_b["exit"] == 0
    assert outcome_b["result"]["status"] == "completed"
    assert outcome_b["result"]["requests"]["main"] == 6


def test_cleanup_failure_stops_globally_before_result(tmp_path, monkeypatch):
    config_logger(None)
    with pytest.raises(ComparisonBudgetError, match="sandbox_cleanup_unproven"):
        _run_inner_offline(
            tmp_path / "cleanup", monkeypatch, task_id="banknote",
            tamper_cleanup=True,
        )
    run_dir = (
        tmp_path / "cleanup" / ".runtime/comparison-istratde/runs/banknote-11-FIXED-0"
    )
    checkpoint = json.loads(
        (run_dir / "search-control-checkpoint.json").read_text()
    )
    assert checkpoint["cleanup_error"] == "sandbox_cleanup_unproven"
    ledger_path = tmp_path / "cleanup" / ".runtime/comparison-istratde/comparison-ledger.json"
    snapshot = _ledger_view(ledger_path)
    assert snapshot["stopped"]["reason"] == "sandbox_cleanup_unproven"
    # Live-job evidence persisted for emergency reconciliation.
    live = json.loads((run_dir / "live-jobs.json").read_text())
    assert live["schema"] == "istratde-comparison-live-jobs.v1"
