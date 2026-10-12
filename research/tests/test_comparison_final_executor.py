"""Real SandboxFinalExecutor failure classification (no network, fake client).

The ordinary per-job timeout and the ORIGINAL experiment deadline are TWO
DISTINCT limits: a clean ordinary job timeout (cancellation + cleanup fully
proven) is a candidate-level failure (raw null / utility 0), while global
deadline expiry, unproven cleanup and contradictions raise so the caller
stops the original ledger. All clients/registry are synthetic; no sandbox,
network, model or live ledger is used.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from research.comparison import final
from research.comparison.budget import ComparisonLedger
from research.comparison.final import FinalPhaseError, SandboxFinalExecutor
from research.contracts.budget import Cancelled
from research.tests.test_comparison_orchestration_supervisor import make_controller


def _fake_registry() -> Any:
    spec = SimpleNamespace(
        split="final",
        metric="accuracy",
        answer_sha256="a" * 64,
        registry_version="evaluator-registry.v1+final.synthetic",
        worker_data_public_path="/mnt/rsi_data/comparison/banknote/data/final",
    )
    return SimpleNamespace(
        task=lambda task_id: spec,
        registry_version="evaluator-registry.v1+final.synthetic",
    )


class FakeClient:
    """Scripted client: ordinary TimeoutError, deadline Cancelled, or a
    completed-looking result; cleanup state is controllable."""

    def __init__(self, behavior: str, *, live: bool = False) -> None:
        self.behavior = behavior
        self.live = live
        self.cancel_calls = 0
        self.submissions: list[dict[str, Any]] = []

    def submit_and_wait(self, source: str, **kwargs: Any) -> Any:
        self.submissions.append(kwargs)
        token = kwargs.get("cancellation_token")
        if (
            token is not None
            and token.deadline_epoch is not None
            and token.deadline_epoch <= time.time()
        ):
            raise Cancelled("original_deadline_expired")
        if self.behavior == "ordinary_timeout":
            raise TimeoutError(
                "sandbox job did not finish; cancel_evidence={'cleanup_verified': True}"
            )
        raise AssertionError("unexpected submission")

    def cancel_all_live(self) -> list[dict[str, Any]]:
        self.cancel_calls += 1
        self.live = False
        return [{"job_id": "job-synthetic", "cleanup_verified": True}]

    def live_job_ids(self) -> frozenset[str]:
        return frozenset({"job-synthetic"}) if self.live else frozenset()

    def uncertain_submission_ids(self) -> frozenset[str]:
        return frozenset()


def _executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: FakeClient,
    *,
    deadline_epoch: float | None,
) -> SandboxFinalExecutor:
    monkeypatch.setattr(
        "research.adapters.sandbox_eval_client.SandboxEvalClient",
        lambda **kwargs: client,
    )
    monkeypatch.setattr(
        "research.search.acceptance_inner.load_sandbox_api_key",
        lambda path: "synthetic-key",
    )
    return SandboxFinalExecutor(
        tmp_path,
        registry=_fake_registry(),
        state_dir=tmp_path / "final-state",
        deadline_fn=(lambda: deadline_epoch) if deadline_epoch is not None else None,
    )


def _translate_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execution: Any
) -> tuple[dict[str, Any], ComparisonLedger]:
    orch = make_controller(tmp_path / "ctrl", monkeypatch)
    cell = orch.cells[0]
    ledger = ComparisonLedger.open(
        tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: 1000.0
    )
    ledger.start_clock()
    winner_bytes = b"print('synthetic winner')\n"
    result = final._translate_execution(
        orch,
        ledger,
        cell,
        {"source_sha256": hashlib.sha256(winner_bytes).hexdigest()},
        execution,
        lambda: 1000.0,
        tmp_path / "sealed",
        intent_sha256="c" * 64,
        barrier_identity="d" * 64,
        answer_sha256="e" * 64,
        registry_version="evaluator-registry.v1+final.synthetic",
        winner_bytes=winner_bytes,
    )
    return result, ledger


def test_ordinary_clean_job_timeout_is_candidate_failure_not_global_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient("ordinary_timeout")
    # Ample ORIGINAL global time remains: the ordinary job limit fires first.
    executor = _executor(
        tmp_path, monkeypatch, client, deadline_epoch=time.time() + 3600.0
    )
    execution = executor.execute(
        task_id="comparison_banknote_final",
        source_bytes=b"print('synthetic')\n",
        cell_key="banknote:11:FIXED",
        timeout_seconds=30.0,
    )
    assert execution.status == "failed"
    assert execution.score is None
    assert execution.cleanup_verified is True
    assert client.cancel_calls >= 1, "cancellation must be attempted"
    # Durable live-job tracking was written (never silently disabled).
    assert (tmp_path / "final-state" / "banknote-11-FIXED" / "live-jobs.json").exists()
    result, ledger = _translate_failed(tmp_path, monkeypatch, execution)
    try:
        assert result["status"] == "failed"
        assert result["raw_score"] is None
        assert result["utility"] == 0.0
        assert not ledger.stopped, "ordinary clean timeout must not stop the ledger"
    finally:
        ledger.close()


def test_global_deadline_expiry_raises_for_global_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient("ordinary_timeout")
    # The ORIGINAL experiment deadline is already past: this is a global
    # stop, never an ordinary candidate timeout.
    executor = _executor(
        tmp_path, monkeypatch, client, deadline_epoch=time.time() - 1.0
    )
    with pytest.raises(Cancelled):
        executor.execute(
            task_id="comparison_banknote_final",
            source_bytes=b"print('synthetic')\n",
            cell_key="banknote:11:FIXED",
            timeout_seconds=30.0,
        )
    assert client.cancel_calls >= 1, "cleanup must be attempted on every path"


def test_timeout_with_unproven_cleanup_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient("ordinary_timeout", live=True)

    class LeakyClient(FakeClient):
        def cancel_all_live(self) -> list[dict[str, Any]]:
            self.cancel_calls += 1
            return []  # cancellation fails: the job stays live

    leaky = LeakyClient("ordinary_timeout", live=True)
    executor = _executor(
        tmp_path, monkeypatch, leaky, deadline_epoch=time.time() + 3600.0
    )
    with pytest.raises(FinalPhaseError, match="final_cleanup_unproven"):
        executor.execute(
            task_id="comparison_banknote_final",
            source_bytes=b"print('synthetic')\n",
            cell_key="banknote:11:FIXED",
            timeout_seconds=30.0,
        )


def test_state_dir_is_required(tmp_path: Path) -> None:
    with pytest.raises(FinalPhaseError, match="final_state_dir_required"):
        SandboxFinalExecutor(tmp_path, registry=_fake_registry(), state_dir=None)
