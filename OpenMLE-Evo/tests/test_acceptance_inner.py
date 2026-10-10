"""US-009 inner acceptance assembly tests (offline; litellm streaming faked).

Exercises the PRODUCTION assembly in research.search.acceptance_inner with
the REAL Evolutionary solver, REAL GenericLLM operators built from the
vendored dojo YAML prompt templates and REAL LiteLLMClient objects (only the
module-level completion_fn is monkeypatched with a deterministic streaming
fake carrying provider usage). No model, sandbox or network calls.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

import operator_trace_harness as h
from dojo.core.solvers.llm_helpers.backends.lite_llm import LiteLLMClient
from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.solvers.evo.evo import Evolutionary
from dojo.solvers.evo.operator_trace import OperatorTraceWriter
from dojo.utils.logger import config_logger
from research.contracts.budget import (
    BudgetLedger,
    BudgetLimits,
    CancellationToken,
)
from research.search import acceptance as ap
from research.search import acceptance_config as ac
from research.search import acceptance_inner as ai_module
from research.search.acceptance_inner import (
    AcceptanceError,
    InnerRunSpec,
    build_acceptance_solver,
    make_prepare_operator,
    operator_clients,
)
from research.search.budget_transport import TransportGuardConfig
from research.search.credentials import ManagedKimiCredentials
from research.search.run_control import run_guarded_search

REPO_ROOT = Path(__file__).resolve().parents[2]
LITELLM_MODULE = "dojo.core.solvers.llm_helpers.backends.lite_llm"
TOKEN = "offline-" + "fixture-" + "token-789"

DECODED = {
    "crossover_prob": 0.7870834469795227,
    "score_weight": 0.3256833553314209,
    "delta_weight": 0.23071765899658203,
    "novelty_weight": 0.9376811981201172,
    "max_debug_depth": 1,
}


def _spec(**overrides) -> InnerRunSpec:
    base = dict(
        run_id="evotest",
        run_dir=".runtime/acceptance-us009/runs/evotest",
        decoded=dict(DECODED),
        ledger_path=".runtime/acceptance-us009/budget-ledger.json",
        runner_commit="c" * 40,
    )
    base.update(overrides)
    return InnerRunSpec(**base)


# ------------------------------------------------------------------- spec
def test_spec_roundtrip_and_validation(tmp_path):
    spec = _spec()
    spec.validate()
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    assert InnerRunSpec.from_json_file(path) == spec


@pytest.mark.parametrize(
    "overrides,rule",
    [
        ({"run_id": "bad id!"}, "spec_run_id_invalid"),
        ({"decoded": {"crossover_prob": 0.5}}, "spec_decoded_dims_mismatch"),
        ({"run_dir": "/abs/path"}, "spec_path_not_relative"),
        ({"run_dir": ".runtime/acceptance-us009/runs/../x"}, "spec_path_not_relative"),
        ({"run_dir": ".runtime/acceptance-us009/other/evotest"}, "spec_run_dir_outside_runtime"),
        ({"ledger_path": ".runtime/other.json"}, "spec_ledger_path_mismatch"),
        ({"runner_commit": ""}, "spec_runner_commit_missing"),
        ({"resource_type": "tpu"}, "spec_resource_type_invalid"),
        ({"job_timeout_seconds": 10}, "spec_job_timeout_invalid"),
    ],
)
def test_spec_fail_closed(overrides, rule):
    with pytest.raises(AcceptanceError, match=rule):
        _spec(**overrides).validate()


def test_spec_zero_weights_rejected_by_safety_gate():
    decoded = dict(DECODED, score_weight=0.0, delta_weight=0.0, novelty_weight=0.0)
    with pytest.raises(ValueError, match="empty_parent_selection_weights"):
        _spec(decoded=decoded).validate()


def test_spec_schema_mismatch():
    with pytest.raises(AcceptanceError, match="spec_schema_mismatch"):
        InnerRunSpec.from_dict({"schema": "other"})


# ---------------------------------------------------------------- assembly
def test_build_solver_real_assembly_matches_pins_and_decoded():
    solver = build_acceptance_solver(_spec(), repo_root=REPO_ROOT)
    assert isinstance(solver, Evolutionary)
    cfg = solver.cfg
    assert cfg.num_generations == 3
    assert cfg.individuals_per_generation == 2
    assert cfg.step_limit == 12
    assert cfg.num_islands == 1
    assert cfg.num_generations_till_crossover == 1
    assert cfg.crossover_prob == pytest.approx(DECODED["crossover_prob"])
    assert cfg.max_debug_depth == 1
    assert cfg.execution_mode == "generation"
    assert cfg.max_llm_call_retries == 1  # fresh managed auth for every generation
    assert cfg.use_test_score is False
    assert cfg.data_preview is False
    weights = cfg.experience["parent_selection"]["weights"]
    assert weights["score"] == pytest.approx(DECODED["score_weight"])
    assert weights["delta"] == pytest.approx(DECODED["delta_weight"])
    assert weights["novelty"] == pytest.approx(DECODED["novelty_weight"])
    assert cfg.experience["enabled"] is True
    assert cfg.experience["prompt_memory"]["enabled"] is False
    # Real operators with real templates and the placeholder credential.
    for name in ("draft", "improve", "debug", "crossover", "analyze"):
        llm = getattr(solver, name + "_fn").args[0]
        assert isinstance(llm, GenericLLM)
        assert isinstance(llm.client, LiteLLMClient)
        assert llm.client.api_key == ac.API_KEY_PLACEHOLDER
        operator_cfg = cfg.operators[name]
        assert operator_cfg.llm.generation_kwargs["stream"] is True
        assert operator_cfg.llm.generation_kwargs["max_tokens"] == 8192
        assert operator_cfg.llm.generation_kwargs["reasoning_effort"] == "low"
        assert operator_cfg.system_message_prompt_template.template.strip()
    clients = operator_clients(solver)
    assert len(clients) == 5


def test_composed_config_files_match_production_contract():
    """The committed Hydra YAMLs satisfy the same pins the builder enforces."""
    evidence = ap.check_validation_configs(REPO_ROOT)
    assert evidence["model"] == "openai/k3"
    assert evidence["stream"] is True


# ------------------------------------------------------- guarded real search
class _Stream:
    """Deterministic streaming fake: content chunks + provider usage chunk."""

    def __init__(self, content: str, usage: dict, calls: list):
        self._content = content
        self._usage = usage
        self._calls = calls

    def __iter__(self):
        midpoint = max(1, len(self._content) // 2)
        for part in (self._content[:midpoint], self._content[midpoint:]):
            yield {"choices": [{"delta": {"content": part}}]}
        yield {"usage": dict(self._usage), "choices": []}

    def close(self):
        pass


def test_guarded_search_with_runtime_credentials_and_hygiene(tmp_path, monkeypatch):
    config_logger(None)
    random.seed(1234)
    solver = build_acceptance_solver(_spec(), repo_root=REPO_ROOT)
    solver.cfg.checkpoint_path = str(tmp_path / "checkpoint")

    calls: list = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        marker = f"call-{len(calls)}"
        code = "import pandas as pd\nSCORE = 0.5\n" f"MARKER = \"{marker}\"\n"
        content = f"Plan {marker}.\n```python\n{code}```\n"
        assert kwargs.get("stream") is True
        return _Stream(content, {"prompt_tokens": 10, "completion_tokens": 20}, calls)

    monkeypatch.setattr(f"{LITELLM_MODULE}.completion_fn", fake_completion)

    credential_file = tmp_path / "kimi-code.json"
    credential_file.write_text(
        json.dumps({"access_token": TOKEN, "refresh_token": "rt",
                    "expires_at": time.time() + 3600}),
        encoding="utf-8",
    )
    credentials = ManagedKimiCredentials(credential_file)
    secrets: list[str] = []
    clients = operator_clients(solver)
    prepare = make_prepare_operator(credentials, clients, secrets)

    ledger = BudgetLedger.open(
        tmp_path / "ledger.json",
        limits=BudgetLimits(max_tokens=1_000_000, max_requests=10_000, max_elapsed_seconds=86400.0),
    )
    token = CancellationToken(deadline_epoch=time.time() + 600)
    trace_path = tmp_path / "operator-trace.jsonl"
    solver.operator_tracer = OperatorTraceWriter(trace_path)
    run_guarded_search(
        solver,
        h.FakeTask(),
        {},
        ledger=ledger,
        token=token,
        transport_config=TransportGuardConfig(
            estimated_input_tokens=100, max_output_tokens=4096,
            request_deadline_seconds=120.0,
        ),
        cancel_live_jobs=lambda: [],
        checkpoint_path=tmp_path / "search-control-checkpoint.json",
        prepare_operator=prepare,
    )

    # Every provider attempt went through the ledger with provider usage.
    snapshot = ledger.snapshot()
    assert snapshot["committed"]["requests"] == len(calls) >= 6
    assert snapshot["committed"]["tokens"] == 30 * len(calls)
    assert snapshot["stopped"] is None
    ledger.close()

    # The placeholder was replaced in memory only, fresh from the file.
    assert all(client.api_key == TOKEN for client in clients)
    assert secrets and all(s == TOKEN for s in secrets)

    # Credential hygiene: the token appears in NO run artifact on disk
    # (checkpoint journal/state, operator trace, search-control checkpoint).
    artifacts = [trace_path, tmp_path / "search-control-checkpoint.json"]
    artifacts.extend((tmp_path / "checkpoint").rglob("*"))
    for artifact in artifacts:
        if artifact.is_file():
            assert TOKEN not in artifact.read_text(encoding="utf-8", errors="replace"), artifact

    # Real trace events for every created node; control checkpoint persisted.
    events = h.read_trace(trace_path)
    non_root = [n for n in solver.journal.nodes if not solver.journal.is_root_node(n)]
    assert len(events) == len(non_root) >= 6
    control = json.loads((tmp_path / "search-control-checkpoint.json").read_text())
    assert control["schema"] == "search-control.v1"
    assert control["termination"] == "completed"
    # The persisted config snapshot hash is computed from credential-free
    # fields only (operator_trace.config_snapshot never reads api_key).
    assert len({e["config"]["config_sha256"] for e in events}) == 1


# ----------------------------------------------------- production run_inner
def _run_inner_offline(base: Path, monkeypatch, run_id: str = "cfg0"):
    """Drive the PRODUCTION run_inner with fake provider/sandbox HTTP only.

    Mirrors the supervisor entry test, kept here so repeatability and
    evidence-field regressions live next to the assembly tests.
    """
    import httpx as _httpx

    base.mkdir(parents=True, exist_ok=True)
    (base / "OpenMLE-Evo").symlink_to(REPO_ROOT / "OpenMLE-Evo", target_is_directory=True)
    monkeypatch.setenv("LOGGING_DIR", str(base / "logs"))
    monkeypatch.setattr(ai_module, "check_runner_review", lambda root: None)
    monkeypatch.setattr(ac, "git_head_commit", lambda root: "c" * 40)
    auth = base / ac.AUTH_FILE
    auth.parent.mkdir(parents=True, exist_ok=True)
    auth.write_text("SANDBOX_API_KEYS=ONLY_SYNTHETIC_SANDBOX_KEY\n", encoding="utf-8")
    auth.chmod(0o600)
    credentials_file = base / "fake-credentials.json"
    credentials_file.write_text(
        json.dumps({"access_token": "offline-" + "repeat-" + "token", "expires_at": time.time() + 3600}),
        encoding="utf-8",
    )
    provider = ManagedKimiCredentials(credentials_file)
    monkeypatch.setattr(ai_module, "ManagedKimiCredentials", lambda: provider)

    calls: list = []

    def completion(**kwargs):
        calls.append(kwargs)
        marker = len(calls)
        content = f"Plan {marker}.\n```python\nprint('candidate {marker}')\n```\n"
        return _Stream(content, {"prompt_tokens": 10, "completion_tokens": 20}, calls)

    monkeypatch.setattr(f"{LITELLM_MODULE}.completion_fn", completion)

    prediction = (REPO_ROOT / "tasks/hello_synth/data/public/sample_submission.csv").read_bytes()
    import hashlib as _hashlib

    jobs: dict = {}

    def service(request):
        if request.method == "POST" and request.url.path == "/api/v1/jobs":
            payload = json.loads(request.content)
            job_id = "job_%016x" % (len(jobs) + 1)
            source = payload["code"].encode()
            raw_hash = _hashlib.sha256(source).hexdigest()
            tree_hash = _hashlib.sha256(b"main.py\0" + raw_hash.encode() + b"\n").hexdigest()
            from research.contracts.source_gate import check_source_bytes, load_policy

            gate = check_source_bytes(source, load_policy()).to_dict()
            gate.update(source_sha256=tree_hash, entrypoint="main.py", entrypoint_sha256=raw_hash)
            jobs[job_id] = {
                "job_id": job_id, "status": "completed",
                "created_at": "2026-10-08T14:45:38Z",
                "completed_at": "2026-10-08T14:45:40Z",
                "result": {
                    "score": 0.55, "result": "success", "source_gate": gate,
                    "source_identity_verified": True, "worker_cleanup_verified": True,
                    "evaluation": {
                        "score": 0.55, "split": "test", "metric": "accuracy",
                        "task_id": "hello_synth", "direction": "maximize",
                        "evaluator": "external_trusted", "row_count": 40,
                        "answer_sha256": "f25ab07cfc53a420662e3ef899df728de124136c70277b66236ca06f08638579",
                        "registry_version": "evaluator-registry.v1",
                        "prediction_sha256": _hashlib.sha256(prediction).hexdigest(),
                    },
                },
            }
            return _httpx.Response(200, json={"job_id": job_id})
        if request.method == "GET":
            return _httpx.Response(200, json=jobs[request.url.path.split("/")[-1]])
        raise AssertionError(request.method)

    original_client = _httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = _httpx.MockTransport(service)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(_httpx, "Client", factory)

    spec = ai_module.InnerRunSpec(
        run_id, str(ac.RUNTIME_DIR / "runs" / run_id), dict(DECODED),
        str(ac.RUNTIME_DIR / ac.LEDGER_FILENAME), "c" * 40,
    )
    spec_path = base / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    ledger_path = base / spec.ledger_path
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with BudgetLedger.open(ledger_path, limits=ac.acceptance_budget_limits()):
        pass
    code = ai_module.run_inner(spec_path, repo_root=base)
    assert code == 0
    run_dir = base / spec.run_dir
    result = json.loads((run_dir / "run-result.json").read_text())
    return result, run_dir, calls, jobs


def test_run_inner_repeatable_branch_selection_and_full_evidence(tmp_path, monkeypatch):
    """Seeded inner RNGs + identical fake provider responses must reproduce
    the exact operator branch selection; the run result carries the full
    completion evidence set."""
    result_a, run_dir_a, calls_a, jobs_a = _run_inner_offline(tmp_path / "a", monkeypatch)
    result_b, run_dir_b, calls_b, jobs_b = _run_inner_offline(tmp_path / "b", monkeypatch, run_id="cfg1")

    # Repeatable branch selection: identical operator sequence and identical
    # generated candidate code across two seeded runs.
    def trace_ops(run_dir):
        return [
            (event["operator"], event["child_code_sha256"])
            for event in (
                json.loads(line)
                for line in (run_dir / "operator-trace.jsonl").read_text().splitlines()
                if line.strip()
            )
        ]

    assert trace_ops(run_dir_a) == trace_ops(run_dir_b)
    assert len(trace_ops(run_dir_a)) == 6

    # Full completion evidence on the PRODUCTION path.
    assert result_a["status"] == "completed"
    assert result_a["main_candidates"] == 6
    assert result_a["generations_completed"] == 3
    assert result_a["best_job_id"] and result_a["best_job_id"] in result_a["job_ids"]
    assert result_a["best_score"] == 0.55
    assert result_a["worker_cleanup_verified"] is True
    assert result_a["budget"]["stopped"] is None
    assert result_a["budget"]["open_reservations"] == []
    assert result_a["budget"]["deadline_epoch"] == pytest.approx(
        result_a["started_at"] + 5400.0
    )
    assert result_a["rng"]["python_random_seed"] == 42
    assert result_a["rng"]["numpy_random_seed"] == 42
    for field in ("operator_trace_sha256", "request_trace_sha256", "journal_sha256"):
        assert len(result_a[field]) == 64
    # Every job carries trusted evaluator evidence bindings.
    for job in result_a["jobs"]:
        assert job["evidence_verified"] is True
        assert len(job["evaluation"]["prediction_sha256"]) == 64
        assert len(job["evaluation"]["answer_sha256"]) == 64
    # Request trace: one record per actual provider attempt, each linked to
    # its reservation and operator, with allowlisted public parameters only.
    records = [
        json.loads(line)
        for line in (run_dir_a / "request-trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == len(calls_a) == 6
    assert [r["attempt"] for r in records] == [1, 2, 3, 4, 5, 6]
    assert all(r["run_id"] == "cfg0" for r in records)
    assert {r["operator"] for r in records} <= {"draft", "improve", "debug", "crossover"}
    for record in records:
        assert set(record["request"]) <= {
            "model", "max_tokens", "max_completion_tokens", "stream",
            "temperature", "top_p", "reasoning_effort", "allowed_openai_params",
            "request_timeout", "stream_options", "num_retries", "max_retries",
        }
        assert record["request"]["stream"] is True
        assert record["request"]["max_tokens"] == 8192
        assert record["request"]["reasoning_effort"] == "low"
        assert "api_key" not in json.dumps(record)
    # Live-jobs registry persisted and fully reconciled at the end.
    live = json.loads((run_dir_a / "live-jobs.json").read_text())
    assert live["live_jobs"] == [] and live["submission_intents"] == []
    assert live["uncertain_submissions"] == []
