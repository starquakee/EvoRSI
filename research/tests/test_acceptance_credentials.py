"""US-009 credential provider tests (offline; fixture files in tmp only)."""
from __future__ import annotations

import json
import time

import pytest

from research.search.credentials import (
    CredentialError,
    ManagedKimiCredentials,
    assert_redacted,
    default_credentials_path,
)

NOW = 1_800_000_000.0
TOKEN = "fixture-" + "token-" + "abc123"


def _write(path, *, token=TOKEN, expires_at=NOW + 900):
    path.write_text(
        json.dumps({"access_token": token, "refresh_token": "rt", "expires_at": expires_at}),
        encoding="utf-8",
    )


def _provider(path, margin=120.0):
    return ManagedKimiCredentials(path, refresh_margin_seconds=margin, now_fn=lambda: NOW)


def test_valid_token_returned(tmp_path):
    path = tmp_path / "kimi-code.json"
    _write(path)
    assert _provider(path).access_token() == TOKEN


def test_fresh_read_per_call(tmp_path):
    """A refresh that rewrites the file between calls is picked up."""
    path = tmp_path / "kimi-code.json"
    _write(path)
    provider = _provider(path)
    assert provider.access_token() == TOKEN
    _write(path, token="rotated-" + "token-456")
    assert provider.access_token() == "rotated-" + "token-456"


def test_expired_token_fails_closed(tmp_path):
    path = tmp_path / "kimi-code.json"
    _write(path, expires_at=NOW - 1)
    with pytest.raises(CredentialError, match="token_expired"):
        _provider(path).access_token()


def test_expiring_within_margin_fails_closed(tmp_path):
    path = tmp_path / "kimi-code.json"
    _write(path, expires_at=NOW + 60)
    with pytest.raises(CredentialError, match="token_expiring_soon"):
        _provider(path).access_token()


def test_missing_file_fails_closed(tmp_path):
    with pytest.raises(CredentialError, match="credentials_unavailable"):
        _provider(tmp_path / "absent.json").access_token()


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps(["not", "an", "object"]),
        json.dumps({"access_token": "", "expires_at": NOW + 900}),
        json.dumps({"access_token": TOKEN, "expires_at": "soon"}),
        json.dumps({"access_token": TOKEN, "expires_at": True}),
        json.dumps({"expires_at": NOW + 900}),
    ],
)
def test_malformed_credentials_fail_closed(tmp_path, payload):
    path = tmp_path / "kimi-code.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(CredentialError, match="credentials_malformed"):
        _provider(path).access_token()


def test_repr_never_contains_token(tmp_path):
    path = tmp_path / "kimi-code.json"
    _write(path)
    provider = _provider(path)
    provider.access_token()
    assert TOKEN not in repr(provider)
    assert TOKEN not in str(provider)
    assert TOKEN not in json.dumps(provider.__dict__, default=str)


def test_env_override_path(tmp_path, monkeypatch):
    path = tmp_path / "elsewhere.json"
    _write(path)
    monkeypatch.setenv("KIMI_CREDENTIALS_FILE", str(path))
    assert default_credentials_path() == path


def test_assert_redacted():
    assert_redacted({"a": 1}, "secret")
    with pytest.raises(CredentialError, match="secret_in_payload"):
        assert_redacted({"nested": ["x", f"prefix-{TOKEN}-suffix"]}, TOKEN)
    # empty secrets are ignored (would trivially match everything)
    assert_redacted({"anything": True}, "")


def test_negative_margin_rejected(tmp_path):
    with pytest.raises(ValueError, match="refresh_margin_must_be_nonnegative"):
        ManagedKimiCredentials(tmp_path / "x", refresh_margin_seconds=-1)


# ---------------------------------------------------------------------------
# Official-helper OAuth refresh (offline: stdlib loopback HTTP server fakes
# the official instance; no model calls, no real credentials).
# ---------------------------------------------------------------------------

import http.server
import os
import threading
from pathlib import Path

from research.search.credentials import KimiAuthHelper

SERVER_TOKEN = "synthetic-" + "server-" + "token"


class _OfficialHandler(http.server.BaseHTTPRequestHandler):
    """Fake official loopback instance: healthz + authenticated usage query
    that 'refreshes' by rewriting the credential file (as the official client
    does with its own locking)."""

    def log_message(self, *args):
        pass

    def _envelope(self, code):
        body = json.dumps({"code": code, "msg": "success" if code == 0 else "err", "data": {}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        state = self.server.state
        if self.path == "/api/v1/healthz":
            return self._envelope(0)
        if self.path.startswith("/api/v1/oauth/usage"):
            state["usage_queries"] += 1
            if self.headers.get("Authorization") != f"Bearer {SERVER_TOKEN}":
                self.send_response(401)
                self.end_headers()
                return
            # official refresh: write a fresh credential file
            state["write_credentials"]()
            return self._envelope(0)
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/v1/shutdown":
            self.server.state["shutdowns"] += 1
            return self._envelope(0)
        self.send_response(404)
        self.end_headers()


def _server(state, port=0):
    server = http.server.HTTPServer(("127.0.0.1", port), _OfficialHandler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _helper(tmp_path, port=None, *, spawn_fn=None):
    instances = tmp_path / "instances"
    instances.mkdir()
    if port is not None:
        (instances / "inst.json").write_text(json.dumps({
            "server_id": "x", "pid": os.getpid(), "host": "127.0.0.1", "port": port,
        }))
    token_file = tmp_path / "server.token"
    token_file.write_text(SERVER_TOKEN)
    return KimiAuthHelper(
        instances_dir=instances,
        server_token_path=token_file,
        kimi_binary="/bin/false",
        spawn_fn=spawn_fn,
    )


def test_refresh_through_discovered_official_instance(tmp_path):
    creds = tmp_path / "kimi-code.json"
    _write(creds, expires_at=NOW - 5)  # expired

    def write_fresh():
        _write(creds, token="refreshed-" + "token", expires_at=NOW + 900)

    state = {"usage_queries": 0, "shutdowns": 0, "write_credentials": write_fresh}
    server = _server(state)
    try:
        helper = _helper(tmp_path, port=server.server_address[1])
        provider = ManagedKimiCredentials(creds, now_fn=lambda: NOW, helper=helper)
        assert provider.access_token() == "refreshed-token"
        assert state["usage_queries"] == 1
        # reused (not owned) instance: close() must NOT shut it down
        helper.close()
        assert state["shutdowns"] == 0
    finally:
        server.shutdown()
        server.server_close()


def test_refresh_rejected_fails_closed_and_sanitized(tmp_path, capsys):
    creds = tmp_path / "kimi-code.json"
    _write(creds, expires_at=NOW - 5)
    token_file = tmp_path / "server.token"
    token_file.write_text("wrong-token")  # bearer mismatch -> 401
    instances = tmp_path / "instances"
    instances.mkdir()
    state = {"usage_queries": 0, "shutdowns": 0, "write_credentials": lambda: None}
    server = _server(state)
    try:
        (instances / "inst.json").write_text(json.dumps({
            "server_id": "x", "pid": os.getpid(), "host": "127.0.0.1",
            "port": server.server_address[1],
        }))
        helper = KimiAuthHelper(instances_dir=instances, server_token_path=token_file,
                                kimi_binary="/bin/false")
        provider = ManagedKimiCredentials(creds, now_fn=lambda: NOW, helper=helper)
        with pytest.raises(CredentialError, match="auth_refresh_http_401"):
            provider.access_token()
        out = capsys.readouterr().out
        assert TOKEN not in out and SERVER_TOKEN not in out
    finally:
        server.shutdown()
        server.server_close()


def test_owned_helper_started_when_none_discovered_and_closed(tmp_path):
    creds = tmp_path / "kimi-code.json"
    _write(creds, expires_at=NOW + 30)  # expiring within margin
    state = {"usage_queries": 0, "shutdowns": 0,
             "write_credentials": lambda: _write(creds, token="owned-" + "refresh", expires_at=NOW + 900)}
    servers = []

    class FakeProcess:
        def __init__(self, port):
            self.pid = 987654
            self._port = port

        def poll(self):
            return None

        def kill(self):
            pass

        def terminate(self):
            pass

    def spawn(command, **kwargs):
        port = int(command[command.index("--port") + 1])
        server = _server(state, port=port)
        servers.append(server)
        return FakeProcess(port)

    try:
        helper = _helper(tmp_path, port=None, spawn_fn=spawn)
        provider = ManagedKimiCredentials(creds, now_fn=lambda: NOW, helper=helper)
        assert provider.access_token() == "owned-refresh"
        helper.close()
        assert state["shutdowns"] == 1  # ONLY the owned helper is shut down
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def test_no_helper_expiry_fails_closed_without_refresh(tmp_path):
    creds = tmp_path / "kimi-code.json"
    _write(creds, expires_at=NOW - 5)
    provider = ManagedKimiCredentials(creds, now_fn=lambda: NOW)
    with pytest.raises(CredentialError, match="token_expired"):
        provider.access_token()


def test_malformed_credentials_never_trigger_refresh(tmp_path):
    creds = tmp_path / "kimi-code.json"
    creds.write_text("{broken", encoding="utf-8")

    class ExplodingHelper:
        def refresh(self):
            raise AssertionError("refresh must not run for malformed credentials")

    provider = ManagedKimiCredentials(creds, now_fn=lambda: NOW, helper=ExplodingHelper())
    with pytest.raises(CredentialError, match="credentials_malformed"):
        provider.access_token()


def test_helper_unavailable_when_no_instance_and_no_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))  # no home binary either
    helper = _helper(tmp_path, port=None)
    helper._kimi_binary = None  # resolve via PATH: use a name that cannot exist
    monkeypatch.setattr("research.search.credentials.shutil.which", lambda name: None)
    with pytest.raises(CredentialError, match="auth_helper_unavailable"):
        helper.refresh()
