"""Runtime-only managed Kimi credential access for the acceptance run.

The controller side reads the CURRENT credential file fresh before every
model request and keeps the token only in process memory: it is never
written to configs, journals, logs, evidence or Git, and never forwarded to
the candidate worker.

Managed OAuth access tokens expire in ~900s while the acceptance window is
5400s, so refresh must be reproducible WITHOUT human intervention: when the
token is expired or expiring, the provider refreshes through an OFFICIAL
loopback Kimi web instance (its own credential refresh and file locking),
discovered from the registered instance metadata or started as an owned
helper on a dynamically chosen port. Only the authenticated non-model
`GET /api/v1/oauth/usage?provider=managed%3Akimi-code` endpoint is used —
never login/logout, message or model endpoints. Helpers started by this
invocation are shut down by it; pre-existing (reused) instances are left
running. No tokens, bearer headers or account details are ever printed.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

#: Environment override for the credential file location (tests/ops).
ENV_CREDENTIALS_PATH = "KIMI_CREDENTIALS_FILE"

#: Default locations relative to the user's home directory (no host paths in
#: source; the audit forbids them under research/).
DEFAULT_CREDENTIALS_RELPATH = Path(".kimi-code") / "credentials" / "kimi-code.json"
DEFAULT_SERVER_TOKEN_RELPATH = Path(".kimi-code") / "server.token"
DEFAULT_INSTANCES_RELPATH = Path(".kimi-code") / "server" / "instances"

DEFAULT_REFRESH_MARGIN_SECONDS = 120.0

#: The only official endpoint used: non-model managed-auth usage query, which
#: drives the official credential refresh/locking path.
USAGE_PATH = "/api/v1/oauth/usage?provider=managed%3Akimi-code"
HEALTH_PATH = "/api/v1/healthz"
SHUTDOWN_PATH = "/api/v1/shutdown"


class CredentialError(RuntimeError):
    """Credential access failed closed; the message is machine-readable."""


def default_credentials_path() -> Path:
    override = os.environ.get(ENV_CREDENTIALS_PATH)
    if override:
        return Path(override)
    return Path.home() / DEFAULT_CREDENTIALS_RELPATH


class KimiAuthHelper:
    """Discovery/reuse/owned-start of official loopback Kimi web instances.

    All IO goes through an injectable opener/spawn so tests can use fakes;
    production uses stdlib urllib with proxies disabled (loopback only).
    """

    def __init__(
        self,
        *,
        instances_dir: Path | None = None,
        server_token_path: Path | None = None,
        kimi_binary: str | None = None,
        opener: Any = None,
        spawn_fn: Callable[..., Any] | None = None,
        startup_timeout_seconds: float = 30.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self._instances_dir = instances_dir or (Path.home() / DEFAULT_INSTANCES_RELPATH)
        self._server_token_path = server_token_path or (Path.home() / DEFAULT_SERVER_TOKEN_RELPATH)
        self._kimi_binary = kimi_binary
        self._opener = opener
        self._spawn_fn = spawn_fn or subprocess.Popen
        self._startup_timeout = startup_timeout_seconds
        self._sleep = sleep_fn
        self._owned: list[dict[str, Any]] = []

    # ------------------------------------------------------------- plumbing
    def _build_opener(self) -> Any:
        if self._opener is not None:
            return self._opener
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _server_token(self) -> str:
        try:
            token = self._server_token_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise CredentialError("auth_helper_token_unavailable") from exc
        if not token:
            raise CredentialError("auth_helper_token_unavailable")
        return token

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except (OSError, OverflowError):
            return False
        return True

    def _get(self, port: int, path: str, *, authenticated: bool) -> tuple[int, Any]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", method="GET"
        )
        if authenticated:
            request.add_header("Authorization", f"Bearer {self._server_token()}")
        try:
            with self._build_opener().open(request, timeout=15) as response:
                body = response.read()
                return response.status, body
        except urllib.error.HTTPError as exc:
            # Sanitized: the envelope body/headers may carry account details.
            return exc.code, None
        except (OSError, ValueError) as exc:
            raise CredentialError("auth_helper_unreachable") from exc

    # ------------------------------------------------------------ discovery
    def discover(self) -> list[dict[str, Any]]:
        """Registered loopback instances with a live process."""
        entries: list[dict[str, Any]] = []
        try:
            files = sorted(self._instances_dir.glob("*.json"))
        except OSError:
            return entries
        for path in files:
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(meta, dict):
                continue
            host, port, pid = meta.get("host"), meta.get("port"), meta.get("pid")
            if host != "127.0.0.1" or not isinstance(port, int) or not isinstance(pid, int):
                continue
            if not self._pid_alive(pid):
                continue
            entries.append({"host": host, "port": port, "pid": pid, "owned": False})
        return entries

    def _healthy(self, port: int) -> bool:
        try:
            status, _ = self._get(port, HEALTH_PATH, authenticated=False)
        except CredentialError:
            return False
        return status == 200

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _start_owned(self) -> dict[str, Any]:
        binary = self._kimi_binary or shutil.which("kimi")
        if not binary:
            # The deployment's configured home binary (PATH may be clean).
            home_binary = Path.home() / ".kimi-code" / "bin" / "kimi"
            if home_binary.exists():
                binary = str(home_binary)
        if not binary:
            raise CredentialError("auth_helper_unavailable")
        port = self._free_port()
        process = self._spawn_fn(
            [binary, "web", "--port", str(port), "--no-open"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        entry = {"host": "127.0.0.1", "port": port, "pid": process.pid,
                 "owned": True, "process": process}
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise CredentialError("auth_helper_start_failed")
            if self._healthy(port):
                self._owned.append(entry)
                return entry
            self._sleep(0.2)
        try:
            process.kill()
        except OSError:
            pass
        raise CredentialError("auth_helper_start_failed")

    def ensure(self) -> dict[str, Any]:
        """A healthy official instance: reuse a live one, else start owned."""
        for entry in self.discover():
            if self._healthy(entry["port"]):
                return entry
        return self._start_owned()

    def refresh(self) -> None:
        """Drive the OFFICIAL managed-OAuth refresh (usage query only)."""
        entry = self.ensure()
        status, body = self._get(entry["port"], USAGE_PATH, authenticated=True)
        if status != 200:
            raise CredentialError(f"auth_refresh_http_{status}")
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (ValueError, UnicodeDecodeError) as exc:
            raise CredentialError("auth_refresh_malformed") from exc
        if not isinstance(payload, dict) or payload.get("code") != 0:
            raise CredentialError("auth_refresh_rejected")

    def close(self) -> None:
        """Shut down ONLY helpers started by this invocation; reused
        pre-existing instances are left running."""
        for entry in self._owned:
            process = entry.get("process")
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{entry['port']}{SHUTDOWN_PATH}", method="POST"
                )
                request.add_header("Authorization", f"Bearer {self._server_token()}")
                self._build_opener().open(request, timeout=10).close()
            except Exception:
                if process is not None:
                    try:
                        process.terminate()
                    except OSError:
                        pass
        self._owned.clear()


class ManagedKimiCredentials:
    """Fresh-read provider for the managed Kimi OAuth access token.

    The token is returned to the caller and intentionally NOT cached on the
    instance, so ``repr``/attribute dumps can never leak it and every call
    observes the latest on-disk refresh. When the on-disk token is expired or
    inside the refresh margin, an injected official :class:`KimiAuthHelper`
    performs the refresh (reproducible across the 90-minute window); without
    a helper, expiry fails closed.
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        refresh_margin_seconds: float = DEFAULT_REFRESH_MARGIN_SECONDS,
        now_fn: Callable[[], float] = time.time,
        helper: KimiAuthHelper | None = None,
    ):
        if refresh_margin_seconds < 0:
            raise ValueError("refresh_margin_must_be_nonnegative")
        self._path = path
        self._margin = float(refresh_margin_seconds)
        self._now_fn = now_fn
        self._helper = helper

    @property
    def path(self) -> Path:
        return self._path if self._path is not None else default_credentials_path()

    def attach_helper(self, helper: KimiAuthHelper) -> None:
        """Attach the official refresh helper (post-construction so tests can
        substitute the provider wholesale)."""
        self._helper = helper

    def _read_token(self) -> str:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise CredentialError("credentials_unavailable") from exc
        except ValueError as exc:
            raise CredentialError("credentials_malformed") from exc
        if not isinstance(raw, dict):
            raise CredentialError("credentials_malformed")
        token = raw.get("access_token")
        expires_at = raw.get("expires_at")
        if (
            not isinstance(token, str)
            or not token
            or isinstance(expires_at, bool)
            or not isinstance(expires_at, (int, float))
            or not math.isfinite(float(expires_at))
        ):
            raise CredentialError("credentials_malformed")
        remaining = float(expires_at) - self._now_fn()
        if remaining <= 0:
            raise CredentialError("token_expired")
        if remaining < self._margin:
            raise CredentialError("token_expiring_soon")
        return token

    def access_token(self) -> str:
        """Return the current access token, refreshing via the official
        helper when expired/expiring; fail closed otherwise."""
        try:
            return self._read_token()
        except CredentialError as exc:
            reason = str(exc)
            if self._helper is None or reason not in ("token_expired", "token_expiring_soon"):
                raise
        self._helper.refresh()  # official refresh + file locking
        return self._read_token()  # re-read; still stale -> fail closed

    def __repr__(self) -> str:  # never includes token material
        return f"ManagedKimiCredentials(path={self.path!s})"


def assert_redacted(payload: Any, *secrets: str) -> None:
    """Fail closed if any secret string appears in a JSON-serializable payload.

    Used before persisting acceptance evidence. Empty secrets are ignored
    (they would trivially match everything).
    """
    text = json.dumps(payload, sort_keys=True, default=str)
    for secret in secrets:
        if secret and secret in text:
            raise CredentialError("secret_in_payload")
