"""Sandbox endpoint defaults and rollback routing (US-010).

The verified isolated rsi-trustworthy stack (http://127.0.0.1:6581,
US-006/US-008 acceptance evidence in reports/) is the DEFAULT endpoint for
all local research clients. The legacy Windows-mounted stack
(http://127.0.0.1:6580) is preserved untouched as an explicit rollback ONLY:
it is selected by an explicit endpoint argument or the SANDBOX_ENDPOINT
environment variable, never by silent fallback.

Resolution order (first wins): explicit argument > SANDBOX_ENDPOINT >
DEFAULT_ENDPOINT. Malformed endpoints fail closed with a machine-readable
error. Credentials are never defaulted: SANDBOX_API_KEY stays mandatory
from the environment (or an explicit argument).
"""
from __future__ import annotations

import os
from urllib.parse import urlsplit

# Verified isolated rsi-trustworthy gateway (loopback only, US-006/008).
DEFAULT_ENDPOINT = "http://127.0.0.1:6581"
# Legacy Windows-mounted stack (rollback only; unchanged, still running).
LEGACY_ROLLBACK_ENDPOINT = "http://127.0.0.1:6580"

ENDPOINT_ENV_VAR = "SANDBOX_ENDPOINT"


class EndpointError(ValueError):
    """Machine-readable endpoint configuration failure (fail closed)."""


def validate_endpoint(endpoint: str) -> str:
    """Validate and normalize an endpoint URL; fail closed on malformed input."""
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise EndpointError("sandbox_endpoint_invalid:empty")
    candidate = endpoint.strip().rstrip("/")
    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        raise EndpointError(f"sandbox_endpoint_invalid:unparseable:{exc}") from exc
    if parts.scheme not in ("http", "https"):
        raise EndpointError("sandbox_endpoint_invalid:scheme_must_be_http_or_https")
    if not parts.hostname:
        raise EndpointError("sandbox_endpoint_invalid:missing_host")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise EndpointError("sandbox_endpoint_invalid:unexpected_path_or_query")
    return candidate


def resolve_endpoint(
    explicit: str | None = None,
    environ: dict[str, str] | None = None,
) -> str:
    """Resolve the sandbox endpoint: explicit arg > env > verified default.

    The legacy 6580 stack is reachable only by passing it explicitly here or
    via SANDBOX_ENDPOINT; there is no silent fallback to it.
    """
    env = os.environ if environ is None else environ
    value = explicit or env.get(ENDPOINT_ENV_VAR) or DEFAULT_ENDPOINT
    return validate_endpoint(value)


def describe_endpoint(endpoint: str) -> str:
    """Human label for an already-validated endpoint (for logs/evidence)."""
    normalized = validate_endpoint(endpoint)
    if normalized == DEFAULT_ENDPOINT:
        return "rsi-trustworthy(isolated,default)"
    if normalized == LEGACY_ROLLBACK_ENDPOINT:
        return "legacy-windows-mounted(rollback,explicit)"
    return "explicit-override"
