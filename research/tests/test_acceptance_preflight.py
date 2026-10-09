"""US-009 preflight check unit tests (offline; tmp fixtures, loopback HTTP)."""
from __future__ import annotations

import hashlib
import http.server
import json
import threading

import pytest

from research.search import acceptance as ap
from research.search import acceptance_config as ac
from research.search.acceptance_inner import AcceptanceError


def test_ledger_absent_ok_on_fresh_root(tmp_path):
    evidence = ap.check_ledger_absent(tmp_path)
    assert evidence["exists"] is False


def test_ledger_present_fails(tmp_path):
    ledger = tmp_path / ac.RUNTIME_DIR / ac.LEDGER_FILENAME
    ledger.parent.mkdir(parents=True)
    ledger.write_text("{}", encoding="utf-8")
    with pytest.raises(AcceptanceError, match="acceptance_ledger_already_exists"):
        ap.check_ledger_absent(tmp_path)


def test_public_data_hash_match(tmp_path):
    content = b"id,f1,f2,label\n200,0.1,0.2,0\n"
    digest = hashlib.sha256(content).hexdigest()
    relative = "tasks/hello_synth/data/public/train.csv"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    original = ac.PUBLIC_DATA_HASHES
    try:
        ac.PUBLIC_DATA_HASHES = {relative: digest}
        assert ap.check_public_data(tmp_path)[relative] == digest
    finally:
        ac.PUBLIC_DATA_HASHES = original


def test_public_data_hash_mismatch_fails(tmp_path):
    relative = "tasks/hello_synth/data/public/train.csv"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"tampered")
    original = ac.PUBLIC_DATA_HASHES
    try:
        ac.PUBLIC_DATA_HASHES = {relative: "0" * 64}
        with pytest.raises(AcceptanceError, match="public_data_hash_mismatch"):
            ap.check_public_data(tmp_path)
    finally:
        ac.PUBLIC_DATA_HASHES = original


def test_public_data_missing_fails(tmp_path):
    original = ac.PUBLIC_DATA_HASHES
    try:
        ac.PUBLIC_DATA_HASHES = {"tasks/x/train.csv": "0" * 64}
        with pytest.raises(AcceptanceError, match="public_data_missing"):
            ap.check_public_data(tmp_path)
    finally:
        ac.PUBLIC_DATA_HASHES = original


def test_auth_file_ok(tmp_path):
    auth = tmp_path / ac.AUTH_FILE
    auth.parent.mkdir(parents=True)
    auth.write_text("SANDBOX_API_KEYS=fixture\n", encoding="utf-8")
    auth.chmod(0o600)
    evidence = ap.check_auth_file(tmp_path)
    assert evidence["mode"] == "0o600"
    assert "fixture" not in json.dumps(evidence)


def test_auth_file_too_open_fails(tmp_path):
    auth = tmp_path / ac.AUTH_FILE
    auth.parent.mkdir(parents=True)
    auth.write_text("SANDBOX_API_KEYS=fixture\n", encoding="utf-8")
    auth.chmod(0o644)
    with pytest.raises(AcceptanceError, match="sandbox_auth_permissions_too_open"):
        ap.check_auth_file(tmp_path)


def test_auth_file_missing_fails(tmp_path):
    with pytest.raises(AcceptanceError, match="sandbox_auth_unavailable"):
        ap.check_auth_file(tmp_path)


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(401)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_gateway_reachable_with_auth_challenge():
    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        evidence = ap.check_gateway(f"http://127.0.0.1:{port}")
        assert evidence["status"] == 401
    finally:
        server.shutdown()
        server.server_close()


def test_gateway_unreachable_fails():
    with pytest.raises(AcceptanceError, match="gateway_unreachable"):
        ap.check_gateway("http://127.0.0.1:9")  # discard port: refused


def test_evaluator_registry_real_repo():
    """The committed synthetic registry loads and allowlists hello_synth."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    evidence = ap.check_evaluator_registry(repo)
    assert evidence["task_id"] == "hello_synth"
    assert evidence["metric"] == "accuracy"


def test_preflight_report_schema_and_no_side_effects(tmp_path, capsys):
    """preflight never creates the ledger or reads credentials."""
    import research.search.credentials as creds

    def forbidden(self):  # pragma: no cover
        raise AssertionError("preflight must not read credentials")

    original = creds.ManagedKimiCredentials.access_token
    creds.ManagedKimiCredentials.access_token = forbidden
    try:
        code = ap.preflight(tmp_path)
    finally:
        creds.ManagedKimiCredentials.access_token = original
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "us009-preflight.v1"
    assert report["offline_preflight_only"] is True
    assert report["model_calls"] == 0
    assert report["ledger_created"] is False
    assert report["model_credentials_read"] is False
    assert report["sandbox_auth_checked"] is True
    assert code == 1  # tmp root lacks configs/data -> checks fail closed
    assert not (tmp_path / ac.RUNTIME_DIR / ac.LEDGER_FILENAME).exists()
    assert report["runner_review"]["bound"] is False
    failed = {name for name, c in report["checks"].items() if not c["ok"]}
    assert failed  # every failure carries a machine-readable reason
    assert all(report["checks"][name]["reason"] for name in failed)
