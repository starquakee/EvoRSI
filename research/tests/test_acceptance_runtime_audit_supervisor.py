"""The reusable historical audit must reject unresolved usage and false completion."""
import hashlib
import json
import pytest
from research.tools import audit_acceptance_runtime as audit

def run_fixture(tmp_path, monkeypatch, *, reservations=None, stopped=None):
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    path = tmp_path / ".runtime/acceptance-us009-round2/budget-ledger.json"
    path.parent.mkdir(parents=True)
    ledger = {"schema_version":2, "started_at":100, "limits":{"max_tokens":200000,"max_requests":30,"max_elapsed_seconds":5400}, "committed":{"requests":0,"tokens":0}, "reservations":reservations or [], "stopped":stopped, "events":[]}
    path.write_text(json.dumps(ledger))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"verdict":"COMPLETE", "elapsed_seconds":1.0, "runtime_evidence_sha256":{path.relative_to(tmp_path).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()}}))
    issues = []
    audit.audit_round("round2", report, issues)
    return issues

def test_clean_finished_round_is_accepted(tmp_path, monkeypatch):
    assert run_fixture(tmp_path, monkeypatch) == []

def test_v2_outstanding_reservation_is_rejected(tmp_path, monkeypatch):
    issues = run_fixture(tmp_path, monkeypatch, reservations=[{"reservation_id":"unknown-provider-usage","tokens":100,"created_at":101}])
    assert any("reservation" in issue for issue in issues), issues

def test_stopped_round_cannot_match_complete_report(tmp_path, monkeypatch):
    issues = run_fixture(tmp_path, monkeypatch, stopped={"at":101,"reason":"provider_usage_unknown"})
    assert issues, "A stopped round2 was accepted as matching a COMPLETE report"
