"""Offline tests for research/comparison/audit.py (EXP-008).

The audit revalidates ACTUAL persisted artifacts (never copied result
fields) over a genuine completed+published+final-scored synthetic
experiment built by the production machinery. Tampered predictions,
tampered winner bytes, duplicate final intents, orphan ledger charges,
missing ledgers and partial matrices must stay VISIBLE as failed or
incomplete checks. The audit never mutates or locks the original ledger.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from research.comparison import deploy, review
from research.comparison.audit import AUDIT_SCHEMA, audit_experiment
from research.comparison.budget import ComparisonLedger
from research.comparison.orchestrate import Orchestrator
from research.comparison.specs import iter_cells
from research.tests.test_comparison_final import (
    SyntheticExecutor,
    _publish_kwargs,
    _repo_bindings,
    _run_final,
    _synthetic_repo,
)
from research.tests.test_comparison_orchestrate import (
    ScriptedNative,
    default_scores,
    fake_child,
    step_positions,
)
from research.tests.test_comparison_report import _fresh_orch, _write_manifest
from research.tests.test_comparison_specs import COMMIT


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def completed(tmp_path_factory) -> dict:
    """ONE fully completed + published + final-scored synthetic experiment."""
    base = tmp_path_factory.mktemp("audit-complete")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(deploy, "git_head_commit", lambda root: COMMIT)
        orch = Orchestrator.initialize(
            runtime_dir=base / "state",
            repo_root=base,
            cells=iter_cells(_repo_bindings(), COMMIT),
            provider=ScriptedNative(),
            now_fn=lambda: 1000.0,
        )
        ledger_path = orch.runtime_dir / review.LEDGER_FILENAME
        with ComparisonLedger.open(
            ledger_path, cells=orch.cells, now_fn=lambda: 1000.0
        ) as ledger:
            ledger.start_clock()
            orch.run_schedule(
                ledger, fake_child(orch, default_scores(orch), [], ledger_path)
            )
        with ComparisonLedger.open(
            ledger_path, cells=orch.cells, now_fn=lambda: 1000.0
        ) as ledger:
            barrier = orch.verify_final_barrier(ledger)
            from research.comparison.final import publish_final_data

            repo = _synthetic_repo(base)
            publication = publish_final_data(
                orch, ledger, barrier, now_fn=lambda: 2000.0, **_publish_kwargs(repo)
            )
            executor = SyntheticExecutor(
                repo["sealed"], registry_version=publication["registry_version"]
            )
            _run_final(orch, ledger, barrier, executor, now_fn=lambda: 2000.0, repo=repo)
        _write_manifest(orch)
    return {"base": base, "runtime_dir": base / "state"}


def _audit(base: Path):
    return audit_experiment(
        base,
        bindings=_repo_bindings(),
        provider=ScriptedNative(),
        runtime_dir=base / "state",
        sealed_root=base / "research" / "comparison" / "sealed",
        now_fn=lambda: 3000.0,
    )


def _copy(completed: dict, tmp_path: Path) -> Path:
    base = tmp_path / "work"
    shutil.copytree(completed["base"], base)
    return base


def _check(result: dict, name: str) -> dict:
    return next(c for c in result["checks"] if c["name"] == name)


class TestPass:
    def test_full_experiment_audits_pass(self, completed):
        result = _audit(completed["base"])
        assert result["schema"] == AUDIT_SCHEMA
        assert result["verdict"] == "PASS", result["summary"]
        statuses = {c["name"]: c["status"] for c in result["checks"]}
        assert statuses == {
            "orchestration_state": "pass",
            "manifest_binding": "pass",
            "ledger_accounting": "pass",
            "ledger_reconciliation": "pass",
            "barrier_recomputation": "pass",
            "final_publication": "pass",
            "final_results_recomputed": "pass",
            "no_duplicate_final_execution": "pass",
            "unknown_usage_visible": "pass",
            "matrix_completeness": "pass",
        }
        assert result["summary"]["terminal_cells"] == 27
        assert result["summary"]["final_results"] == 27
        roles = [a["role"] for a in result["artifacts"]]
        assert "budget_ledger" in roles
        assert "experiment_manifest" in roles
        assert "frozen_schedule" in roles
        assert roles.count("cell_state") == 27
        assert roles.count("winner_source") == 27
        assert roles.count("final_result") == 27
        assert roles.count("final_prediction") == 27
        assert roles.count("final_intent") == 27
        assert "publication:registry.final.v1.json" in roles
        assert all(len(a["sha256"]) == 64 for a in result["artifacts"])

    def test_audit_is_read_only(self, completed, tmp_path):
        base = _copy(completed, tmp_path)
        ledger_path = base / "state" / review.LEDGER_FILENAME
        lock = ledger_path.with_name(ledger_path.name + ".lock")
        lock.unlink(missing_ok=True)
        watched = [
            ledger_path,
            base / "state" / "schedule.json",
            base / "state" / "cells" / "banknote-11-FIXED" / "cell.json",
            base / "state" / "cells" / "banknote-11-FIXED" / "winner.py",
            base / "state" / "final" / "banknote-11-FIXED" / "final.json",
            base / "state" / "final" / "banknote-11-FIXED" / "prediction.csv",
            base / "state" / "final" / "publication.json",
        ]
        before = {path: _sha256(path) for path in watched}
        result = _audit(base)
        assert result["verdict"] == "PASS"
        assert {path: _sha256(path) for path in watched} == before
        assert not lock.exists()


class TestTampering:
    def test_tampered_final_prediction_fails(self, completed, tmp_path):
        base = _copy(completed, tmp_path)
        prediction = (
            base / "state" / "final" / "banknote-11-FIXED" / "prediction.csv"
        )
        prediction.write_bytes(prediction.read_bytes() + b"\n")
        result = _audit(base)
        assert result["verdict"] == "FAIL"
        check = _check(result, "final_results_recomputed")
        assert check["status"] == "fail"
        assert "final_prediction_mismatch" in check["detail"]

    def test_tampered_final_score_fails(self, completed, tmp_path):
        base = _copy(completed, tmp_path)
        result_path = base / "state" / "final" / "wdbc-22-RANDOM" / "final.json"
        payload = json.loads(result_path.read_text())
        payload["raw_score"] = 0.99
        result_path.write_text(json.dumps(payload))
        result = _audit(base)
        assert result["verdict"] == "FAIL"
        check = _check(result, "final_results_recomputed")
        assert check["status"] == "fail"

    def test_tampered_winner_source_fails(self, completed, tmp_path):
        base = _copy(completed, tmp_path)
        winner = base / "state" / "cells" / "spambase-33-ISTRATDE" / "winner.py"
        winner.write_text(winner.read_text() + "# tampered\n")
        result = _audit(base)
        assert result["verdict"] == "FAIL"
        assert _check(result, "orchestration_state")["status"] == "fail"
        # Dependent checks are explicit skips, not silent passes.
        assert _check(result, "ledger_reconciliation")["status"] == "skip"

    def test_duplicate_final_intent_fails(self, completed, tmp_path):
        base = _copy(completed, tmp_path)
        cell_final = base / "state" / "final" / "banknote-11-FIXED"
        shutil.copy(cell_final / "intent.json", cell_final / "intent-copy.json")
        result = _audit(base)
        assert result["verdict"] == "FAIL"
        check = _check(result, "no_duplicate_final_execution")
        assert check["status"] == "fail"
        assert "final_intent_duplicate" in check["detail"]

    def test_missing_final_result_is_incomplete(self, completed, tmp_path):
        base = _copy(completed, tmp_path)
        (base / "state" / "final" / "wdbc-11-FIXED" / "final.json").unlink()
        result = _audit(base)
        # The artifact chain still validates, but the missing final result
        # keeps the matrix visibly incomplete.
        assert _check(result, "matrix_completeness")["status"] == "fail"
        assert result["verdict"] == "INCOMPLETE"


class TestIncompleteStates:
    def test_not_started_is_incomplete_not_corrupt(self, tmp_path, monkeypatch):
        orch = _fresh_orch(tmp_path, monkeypatch)
        result = _audit(tmp_path)
        assert result["verdict"] == "INCOMPLETE"
        assert _check(result, "orchestration_state")["status"] == "pass"
        ledger = _check(result, "ledger_accounting")
        assert ledger["status"] == "skip"
        assert "not_started" in ledger["detail"]
        assert _check(result, "matrix_completeness")["status"] == "fail"

    def test_terminal_cells_without_ledger_are_corrupt(self, tmp_path, monkeypatch):
        orch = _fresh_orch(tmp_path, monkeypatch)
        ledger_path = tmp_path / "ledger.json"
        cell_key = "banknote:11:FIXED"
        positions = [p for p in orch.positions if p.cell_key == cell_key]
        positions.sort(key=lambda p: p.position)
        with ComparisonLedger.open(
            ledger_path, cells=orch.cells, now_fn=lambda: 1000.0
        ) as ledger:
            ledger.start_clock()
            step_positions(
                orch,
                ledger,
                positions,
                fake_child(orch, default_scores(orch), [], ledger_path),
            )
        ledger_path.unlink()  # terminal evidence but the ledger is GONE
        result = _audit(tmp_path)
        assert result["verdict"] == "FAIL"
        check = _check(result, "ledger_accounting")
        assert check["status"] == "fail"
        assert "ledger_missing" in check["detail"]

    def test_orphan_charge_is_an_integrity_failure(self, tmp_path, monkeypatch):
        """A settled request without any completed boundary is visible."""
        orch = _fresh_orch(tmp_path, monkeypatch)
        ledger_path = tmp_path / "state" / review.LEDGER_FILENAME
        run = orch.cells[0].inner_run(0)
        with ComparisonLedger.open(
            ledger_path, cells=orch.cells, now_fn=lambda: 1000.0
        ) as ledger:
            ledger.start_clock()
            reservation = ledger.reserve(run, "main", estimated_input_tokens=100)
            ledger.commit(reservation.request_id, None)  # unknown -> stopped
        result = _audit(tmp_path)
        # Unknown usage stays visible and the ledger is stopped as required.
        unknown = _check(result, "unknown_usage_visible")
        assert unknown["status"] == "pass"
        assert "1 unknown-usage" in unknown["detail"]
        # ...but a charge with no completed boundary is an integrity failure.
        reconciliation = _check(result, "ledger_reconciliation")
        assert reconciliation["status"] == "fail"
        assert "ledger_request_orphan" in reconciliation["detail"]
        assert result["verdict"] == "FAIL"
