"""Offline tests for the US-010 endpoint switch (default 6581, explicit 6580 rollback).

Pins the routing contract: the verified isolated rsi-trustworthy stack is the
default for every local research client; the legacy Windows-mounted stack is
reachable ONLY via an explicit argument or SANDBOX_ENDPOINT; malformed
endpoints fail closed; the API key is never defaulted. No network.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from research.adapters.endpoints import (
    DEFAULT_ENDPOINT,
    ENDPOINT_ENV_VAR,
    LEGACY_ROLLBACK_ENDPOINT,
    EndpointError,
    describe_endpoint,
    resolve_endpoint,
    validate_endpoint,
)
from research.adapters.sandbox_eval_client import SandboxEvalClient

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestEndpointConstants:
    def test_default_is_isolated_stack(self):
        assert DEFAULT_ENDPOINT == "http://127.0.0.1:6581"

    def test_rollback_is_legacy_stack(self):
        assert LEGACY_ROLLBACK_ENDPOINT == "http://127.0.0.1:6580"

    def test_default_and_rollback_are_distinct_loopback_http(self):
        assert DEFAULT_ENDPOINT != LEGACY_ROLLBACK_ENDPOINT
        for endpoint in (DEFAULT_ENDPOINT, LEGACY_ROLLBACK_ENDPOINT):
            assert endpoint.startswith("http://127.0.0.1:")


class TestResolveEndpoint:
    def test_default_when_no_explicit_and_no_env(self):
        assert resolve_endpoint(environ={}) == DEFAULT_ENDPOINT

    def test_env_override_selects_explicit_rollback(self):
        env = {ENDPOINT_ENV_VAR: LEGACY_ROLLBACK_ENDPOINT}
        assert resolve_endpoint(environ=env) == LEGACY_ROLLBACK_ENDPOINT

    def test_explicit_argument_wins_over_env(self):
        env = {ENDPOINT_ENV_VAR: LEGACY_ROLLBACK_ENDPOINT}
        assert (
            resolve_endpoint("http://127.0.0.1:9999", environ=env)
            == "http://127.0.0.1:9999"
        )

    def test_no_silent_fallback_to_legacy(self):
        # An empty env value must NOT silently route to the legacy stack;
        # it falls back to the verified default instead.
        assert resolve_endpoint(environ={ENDPOINT_ENV_VAR: ""}) == DEFAULT_ENDPOINT

    def test_trailing_slash_normalized(self):
        assert resolve_endpoint("http://127.0.0.1:6581/", environ={}) == DEFAULT_ENDPOINT


class TestValidateEndpoint:
    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "   ",
            "6581",
            "127.0.0.1:6581",
            "ftp://127.0.0.1:6581",
            "http://",
            "http://127.0.0.1:6581/api/v1",
            "http://127.0.0.1:6581?x=1",
        ],
    )
    def test_malformed_endpoints_fail_closed(self, bad):
        with pytest.raises(EndpointError) as excinfo:
            validate_endpoint(bad)
        assert str(excinfo.value).startswith("sandbox_endpoint_invalid:")

    def test_resolve_uses_validation_for_env(self):
        with pytest.raises(EndpointError):
            resolve_endpoint(environ={ENDPOINT_ENV_VAR: "not-a-url"})


class TestDescribeEndpoint:
    def test_labels(self):
        assert describe_endpoint(DEFAULT_ENDPOINT) == "rsi-trustworthy(isolated,default)"
        assert describe_endpoint(LEGACY_ROLLBACK_ENDPOINT) == (
            "legacy-windows-mounted(rollback,explicit)"
        )
        assert describe_endpoint("http://127.0.0.1:9999") == "explicit-override"


class TestClientRouting:
    def test_client_defaults_to_isolated_stack(self, monkeypatch):
        monkeypatch.delenv("SANDBOX_ENDPOINT", raising=False)
        monkeypatch.setenv("SANDBOX_API_KEY", "test-key")
        client = SandboxEvalClient()
        endpoint, api_key = client._resolve_auth(None, None)
        assert endpoint == DEFAULT_ENDPOINT
        assert api_key == "test-key"

    def test_client_explicit_rollback_via_env(self, monkeypatch):
        monkeypatch.setenv("SANDBOX_ENDPOINT", LEGACY_ROLLBACK_ENDPOINT)
        monkeypatch.setenv("SANDBOX_API_KEY", "test-key")
        client = SandboxEvalClient()
        endpoint, _ = client._resolve_auth(None, None)
        assert endpoint == LEGACY_ROLLBACK_ENDPOINT

    def test_client_explicit_argument_wins(self, monkeypatch):
        monkeypatch.setenv("SANDBOX_ENDPOINT", LEGACY_ROLLBACK_ENDPOINT)
        monkeypatch.setenv("SANDBOX_API_KEY", "test-key")
        client = SandboxEvalClient(endpoint="http://127.0.0.1:9999")
        endpoint, _ = client._resolve_auth(None, None)
        assert endpoint == "http://127.0.0.1:9999"

    def test_api_key_never_defaulted(self, monkeypatch):
        # The endpoint default must not weaken the credential requirement:
        # without SANDBOX_API_KEY the client still fails closed.
        monkeypatch.delenv("SANDBOX_ENDPOINT", raising=False)
        monkeypatch.delenv("SANDBOX_API_KEY", raising=False)
        client = SandboxEvalClient()
        endpoint, api_key = client._resolve_auth(None, None)
        assert endpoint == DEFAULT_ENDPOINT
        assert api_key is None
        evidence = client.cancel_job("job-never-submitted")
        assert evidence["cancel_error"] == "sandbox_endpoint_or_api_key_unavailable"

    def test_malformed_env_endpoint_fails_closed(self, monkeypatch):
        monkeypatch.setenv("SANDBOX_ENDPOINT", "not-a-url")
        monkeypatch.setenv("SANDBOX_API_KEY", "test-key")
        client = SandboxEvalClient()
        with pytest.raises(EndpointError):
            client._resolve_auth(None, None)


# The legacy formal-baseline entrypoint imports vendored istratde/jax, which
# only the pre-existing legacy GPU venv provides (the research venv's newer
# CPU jax lacks PositionalSharding). The interpreter path lives in a JSON
# data file so research/**/*.py stays free of host absolute paths.
LEGACY_VENV_PYTHON = Path(
    json.loads(
        (Path(__file__).resolve().parent / "legacy_venv.json").read_text(
            encoding="utf-8"
        )
    )["python"]
)


@pytest.mark.skipif(
    not LEGACY_VENV_PYTHON.exists(),
    reason="legacy GPU venv not present (integration test only)",
)
class TestLegacyBaselineRollbackGuard:
    """de_formal_baseline targets titanic-extended@1, which exists only on the
    legacy 6580 stack; on the isolated 6581 default the evaluator registry
    allowlists hello_synth only, so the script must refuse BEFORE any Evo/LLM
    spend and require an explicit rollback endpoint."""

    def _run_load_env(self, extra_env: dict[str, str]) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env.pop("SANDBOX_URL", None)
        env.update(extra_env)
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["SANDBOX_API_KEY"] = "test-only"
        env["JAX_PLATFORM_NAME"] = "cpu"
        return subprocess.run(
            [
                str(LEGACY_VENV_PYTHON),
                "-c",
                "from research.experiments.de_formal_baseline import load_env; "
                "print('SANDBOX_URL=' + load_env()['SANDBOX_URL'])",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_default_stack_refused_before_any_spend(self):
        result = self._run_load_env({})
        assert result.returncode != 0
        assert "legacy_baseline_requires_rollback_stack" in result.stderr

    def test_explicit_rollback_accepted(self):
        result = self._run_load_env({"SANDBOX_URL": LEGACY_ROLLBACK_ENDPOINT})
        assert result.returncode == 0, result.stderr
        assert f"SANDBOX_URL={LEGACY_ROLLBACK_ENDPOINT}" in result.stdout
