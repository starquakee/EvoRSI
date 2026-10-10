"""US-009 offline repair regressions: real-seam replay, cooperative stop,
provider rejection and the child budget lower bound.

Everything runs against the PRODUCTION assembly with fake provider/sandbox
HTTP only — no model request, no real sandbox job, no credential, no review
or authorization file. Retained runtime evidence is only READ (skipped when
absent on a fresh checkout).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx
import pytest

from research.contracts.budget import BudgetLedger, BudgetLimits
from research.search import acceptance_config as ac
from research.search import acceptance_inner as ai
from research.search.credentials import ManagedKimiCredentials
from research.search.generation_outcome import (
    classify_completion,
    extract_candidate_code,
    strip_thinking,
)
from test_acceptance_inner import DECODED

ROOT = Path(__file__).resolve().parents[2]
LITELLM_MODULE = "dojo.core.solvers.llm_helpers.backends.lite_llm"
SENTINEL = "ONLY_SYNTHETIC_MODEL_TOKEN_FOR_OFFLINE_REPAIR"


# ---------------------------------------------------------------- replay
JOURNALS = [
    ROOT / ".runtime/acceptance-us009/runs/cfg0/journal.jsonl",
    ROOT / ".runtime/acceptance-us009/runs/cfg1/journal.jsonl",
]


@pytest.mark.skipif(not all(p.exists() for p in JOURNALS), reason="retained runtime evidence absent")
def test_retained_truncation_records_extract_empty_at_real_seam():
    """Replay the retained response text through the ACTUAL dojo extract_code
    (no candidate code is executed): the 9 truncated/malformed records must
    extract empty, and the acceptance classifier must agree on every node."""
    from dojo.core.solvers.utils.response import extract_code as real_extract_code

    empty = 0
    total = 0
    for path in JOURNALS:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            node = json.loads(line)
            if node.get("step") == 0:
                continue  # root node
            total += 1
            stored = node.get("code") or ""
            real = real_extract_code(stored)
            ours = extract_candidate_code(stored)
            # The offline mirror matches the real seam on every record.
            assert bool(real.strip()) == bool(ours.strip())
            if not real.strip():
                empty += 1
                classification, _ = classify_completion(stored)
                assert classification in (
                    "empty_visible_content",
                    "unclosed_fence",
                    "invalid_python",
                    "no_complete_code_block",
                )
    assert total == 16
    assert empty == 9  # the confirmed diagnosis: 9 of 16 extract empty


def test_extraction_mirror_matches_real_seam_on_synthetic_cases():
    from dojo.core.solvers.utils.response import extract_code as real_extract_code

    cases = [
        "",
        "plain prose — nothing to extract",
        "Plan.\n```python\nprint('ok')\n```\n",
        "```python\nunclosed_block = True\n",
        "```python\nnot — python\n```\n",
        "print('plain code fallback')",
        "<think>hidden</think>```python\nx = 1\n```",
    ]
    for text in cases:
        assert bool(real_extract_code(text).strip()) == bool(
            extract_candidate_code(text).strip()
        ), text


def test_strip_thinking_matches_dojo_parse():
    from dojo.core.solvers.utils.response import parse_thinking_tags

    for text in ("<think>a</think>code", "b</think>code", "plain"):
        assert strip_thinking(text) == parse_thinking_tags(text)[1]


# ------------------------------------------------------- offline run harness
def _offline_env(base: Path, monkeypatch):
    base.mkdir(parents=True, exist_ok=True)
    (base / "OpenMLE-Evo").symlink_to(ROOT / "OpenMLE-Evo", target_is_directory=True)
    (base / "research").symlink_to(ROOT / "research", target_is_directory=True)
    (base / "tasks").symlink_to(ROOT / "tasks", target_is_directory=True)
    monkeypatch.setenv("LOGGING_DIR", str(base / "logs"))
    monkeypatch.setattr(ai, "check_runner_review", lambda root: None)
    monkeypatch.setattr(ac, "git_head_commit", lambda root: "c" * 40)
    auth = base / ac.AUTH_FILE
    auth.parent.mkdir(parents=True, exist_ok=True)
    auth.write_text("SANDBOX_API_KEYS=ONLY_SYNTHETIC_SANDBOX_KEY\n", encoding="utf-8")
    auth.chmod(0o600)
    credentials_file = base / "fake-credentials.json"
    credentials_file.write_text(
        json.dumps({"access_token": SENTINEL, "expires_at": time.time() + 3600}),
        encoding="utf-8",
    )
    provider = ManagedKimiCredentials(credentials_file)
    monkeypatch.setattr(ai, "ManagedKimiCredentials", lambda **kwargs: provider)


def _fake_sandbox(monkeypatch):
    posts: list = []

    def service(request):
        if request.method == "POST" and request.url.path == "/api/v1/jobs":
            posts.append(json.loads(request.content))
            return httpx.Response(200, json={"job_id": "job_%016x" % len(posts)})
        if request.method == "GET":
            return httpx.Response(200, json={
                "job_id": request.url.path.split("/")[-1], "status": "completed",
                "created_at": "2026-10-10T00:00:00Z",
                "started_at": "2026-10-10T00:00:01Z",
                "completed_at": "2026-10-10T00:00:02Z",
                "result": {
                    "result": "success", "score": None,
                    "source_identity_verified": True,
                    "worker_cleanup_verified": True,
                },
            })
        raise AssertionError("unexpected offline request: " + request.method)

    original = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(service)
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    return posts


def _spec_and_ledger(base: Path, *, max_requests: int = 30):
    spec = ai.InnerRunSpec(
        "cfg0", str(ac.RUNTIME_DIR / "runs/cfg0"), dict(DECODED),
        str(ac.RUNTIME_DIR / ac.LEDGER_FILENAME), "c" * 40,
    )
    spec_path = base / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    ledger_path = base / spec.ledger_path
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with BudgetLedger.open(
        ledger_path,
        limits=BudgetLimits(
            max_tokens=200_000, max_requests=max_requests, max_elapsed_seconds=5400.0
        ),
    ):
        pass
    return spec, spec_path, ledger_path


class _Stream:
    def __init__(self, content, usage):
        self._content = content
        self._usage = usage

    def __iter__(self):
        yield {"choices": [{"delta": {"content": self._content}}]}
        yield {"usage": dict(self._usage), "choices": []}

    def close(self):
        pass


# ------------------------------------------------------- cooperative stop
def test_cooperative_stop_between_generation_and_scoring(tmp_path, monkeypatch):
    """Drive the PRODUCTION call sequence and stop precisely between
    generation and scoring: exactly one budgeted call, the candidate source
    bytes + outcome artifact survive, nothing is submitted or scored, and no
    dangling success/cache state remains."""
    _offline_env(tmp_path, monkeypatch)
    posts = _fake_sandbox(monkeypatch)
    spec, spec_path, ledger_path = _spec_and_ledger(tmp_path)
    run_dir = tmp_path / spec.run_dir

    calls: list = []

    def completion(**kwargs):
        calls.append(kwargs)
        # Externally request the cooperative stop DURING the first settled
        # provider call (file scoped to this run only).
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "stop-request.json").write_text('{"reason": "offline stop test"}')
        content = "Plan.\n```python\nprint('offline candidate')\n```\n"
        return _Stream(content, {"prompt_tokens": 10, "completion_tokens": 20})

    monkeypatch.setattr(f"{LITELLM_MODULE}.completion_fn", completion)

    assert ai.run_inner(spec_path, repo_root=tmp_path) == 0

    # Exactly one budgeted call, settled with provider usage.
    assert len(calls) == 1
    ledger = json.loads(ledger_path.read_text())
    assert ledger["committed"]["requests"] == 1
    assert ledger["committed"]["tokens"] == 30
    # Generation artifacts survived the stop: exact source bytes + outcome.
    code_path = run_dir / "generated" / "001-draft.py"
    assert code_path.is_file()
    code_text = code_path.read_text()
    outcomes = [
        json.loads(line)
        for line in (run_dir / "generation-outcomes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome["classification"] == "ok"
    assert outcome["extracted_sha256"] == hashlib.sha256(code_text.encode()).hexdigest()
    assert outcome["node_code_sha256"] == outcome["extracted_sha256"]
    # The stop landed BEFORE the operator trace and any evaluation.
    trace_path = run_dir / "operator-trace.jsonl"
    assert not trace_path.exists() or not trace_path.read_text().strip()
    assert posts == []  # no candidate ever reached the sandbox
    result = json.loads((run_dir / "run-result.json").read_text())
    assert result["status"] == "failed"
    assert result["termination"] == "Cancelled:cooperative_stop:stop_requested_file"
    assert result["best_score"] is None
    assert result["jobs"] == []
    assert result["job_counts"]["submitted"] == 0
    # The journal holds only the root node — no fabricated success node.
    journal_src = run_dir / "checkpoint" / "journal.jsonl"
    if journal_src.exists():
        nodes = [json.loads(l) for l in journal_src.read_text().splitlines() if l.strip()]
        assert all(node.get("step") == 0 for node in nodes)
    control = json.loads((run_dir / "search-control-checkpoint.json").read_text())
    assert control["termination"] == "Cancelled:cooperative_stop:stop_requested_file"
    # No dangling live/unknown submissions.
    live = json.loads((run_dir / "live-jobs.json").read_text())
    assert live["live_jobs"] == [] and live["submission_intents"] == []
    # Credential hygiene across every artifact.
    combined = "\n".join(
        p.read_text(errors="replace") for p in run_dir.rglob("*") if p.is_file()
    )
    assert SENTINEL not in combined


# ------------------------------------------------------- provider rejection
def test_provider_parameter_rejection_stops_run_without_fallback(tmp_path, monkeypatch):
    """A provider 400 (e.g. an unsupported parameter) must stop the run after
    exactly ONE budgeted attempt — no hidden retry or function fallback."""
    import litellm

    _offline_env(tmp_path, monkeypatch)
    posts = _fake_sandbox(monkeypatch)
    spec, spec_path, ledger_path = _spec_and_ledger(tmp_path)

    calls: list = []

    def completion(**kwargs):
        calls.append(kwargs)
        raise litellm.BadRequestError(
            "Unsupported parameter: reasoning_effort",
            model="openai/k3",
            llm_provider="openai",
        )

    monkeypatch.setattr(f"{LITELLM_MODULE}.completion_fn", completion)

    assert ai.run_inner(spec_path, repo_root=tmp_path) == 0
    assert len(calls) == 1
    assert posts == []
    ledger = json.loads(ledger_path.read_text())
    assert ledger["stopped"] is not None
    assert ledger["committed"]["requests"] == 1  # retained unknown-usage charge
    run_dir = tmp_path / spec.run_dir
    result = json.loads((run_dir / "run-result.json").read_text())
    assert result["status"] == "failed"
    assert result["jobs"] == []


# ------------------------------------------------------- child lower bound
def test_child_refuses_when_remaining_requests_below_main_minimum(tmp_path, monkeypatch):
    """The child needs at least 3 generations x 2 candidates = 6 main calls;
    with fewer remaining it refuses BEFORE any provider call."""
    _offline_env(tmp_path, monkeypatch)
    _fake_sandbox(monkeypatch)
    spec, spec_path, ledger_path = _spec_and_ledger(tmp_path, max_requests=5)

    calls: list = []
    monkeypatch.setattr(
        f"{LITELLM_MODULE}.completion_fn",
        lambda **kwargs: calls.append(kwargs),
    )
    with pytest.raises(ai.AcceptanceError, match="remaining_budget_insufficient"):
        ai.run_inner(spec_path, repo_root=tmp_path)
    assert calls == []
    ledger = json.loads(ledger_path.read_text())
    assert ledger["committed"]["requests"] == 0
    # Not even a reservation was made for the refused run.
    assert not [e for e in ledger["events"] if e["type"] == "reserve"]


def test_final_main_call_allowed_exactly_at_budget_equality(tmp_path, monkeypatch):
    """Positive boundary: with exactly 3x2 requests remaining (and no future
    outer configs), ALL six main calls must proceed — main is allowed at
    equality; only optional debug needs extra slack."""
    _offline_env(tmp_path, monkeypatch)
    posts = _fake_sandbox(monkeypatch)
    spec, spec_path, ledger_path = _spec_and_ledger(tmp_path, max_requests=6)

    calls: list = []

    def completion(**kwargs):
        calls.append(kwargs)
        content = "Plan %d.\n```python\nprint('candidate %d')\n```\n" % (len(calls), len(calls))
        return _Stream(content, {"prompt_tokens": 10, "completion_tokens": 20})

    monkeypatch.setattr(f"{LITELLM_MODULE}.completion_fn", completion)

    assert ai.run_inner(spec_path, repo_root=tmp_path) == 0
    assert len(calls) == 6
    assert len(posts) == 6
    ledger = json.loads(ledger_path.read_text())
    assert ledger["committed"]["requests"] == 6
    assert ledger["stopped"] is None  # budget fit exactly; no stop was needed
    run_dir = tmp_path / spec.run_dir
    attempts = [
        json.loads(line)
        for line in (run_dir / "request-trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(attempts) == 6
    assert {entry["operator"] for entry in attempts} <= {"draft", "improve", "crossover"}
    result = json.loads((run_dir / "run-result.json").read_text())
    assert result["main_candidates"] == 6
    assert result["generations_completed"] == 3


# ------------------------------------------------------- truncation feedback
class _FinishStream:
    def __init__(self, content, usage, finish_reason=None):
        self._content = content
        self._usage = usage
        self._finish = finish_reason

    def __iter__(self):
        yield {"choices": [{"delta": {"content": self._content}}]}
        if self._finish is not None:
            yield {"choices": [{"delta": {}, "finish_reason": self._finish}]}
        yield {"usage": dict(self._usage), "choices": []}

    def close(self):
        pass


def test_truncated_generation_produces_precise_debug_feedback(tmp_path, monkeypatch):
    """A length-truncated draft (unclosed fence) must NOT collapse into an
    unexplained empty_candidate_code: the outcome record classifies it, and
    the debug operator's actual prompt carries the precise cause."""
    _offline_env(tmp_path, monkeypatch)
    posts = _fake_sandbox(monkeypatch)
    spec, spec_path, ledger_path = _spec_and_ledger(tmp_path)
    run_dir = tmp_path / spec.run_dir

    truncated = "Plan.\n```python\nimport pandas as pd\ndef train():\n"
    calls: list = []

    def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _FinishStream(
                truncated,
                {"prompt_tokens": 530, "completion_tokens": 8192,
                 "completion_tokens_details": {"reasoning_tokens": 4000}},
                finish_reason="length",
            )
        content = "Plan %d.\n```python\nprint('candidate %d')\n```\n" % (len(calls), len(calls))
        return _FinishStream(content, {"prompt_tokens": 10, "completion_tokens": 20})

    monkeypatch.setattr(f"{LITELLM_MODULE}.completion_fn", completion)

    assert ai.run_inner(spec_path, repo_root=tmp_path) == 0

    outcomes = [
        json.loads(line)
        for line in (run_dir / "generation-outcomes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    first = outcomes[0]
    assert first["classification"] == "unclosed_fence"
    assert first["truncated"] is True
    assert first["usage"]["finish_reason"] == "length"
    assert first["usage"]["reasoning_tokens"] == 4000
    # The truncated raw response is preserved for offline diagnosis (the
    # seam strips thinking tags/outer whitespace, as does the journal).
    assert (run_dir / "generated" / "001-draft.response.txt").read_text() == truncated.strip()
    # The next (debug) provider call carries the precise bounded cause.
    debug_messages = json.dumps(calls[1]["messages"])
    assert "unclosed_fence" in debug_messages
    assert "finish_reason=length" in debug_messages
    # The failed draft node keeps the classified feedback (no bare collapse).
    journal_src = run_dir / "checkpoint" / "journal.jsonl"
    nodes = [json.loads(l) for l in journal_src.read_text().splitlines() if l.strip()]
    draft_nodes = [n for n in nodes if n.get("step") == 1]
    assert draft_nodes
    assert "empty_candidate_code:unclosed_fence" in json.dumps(draft_nodes[0])
