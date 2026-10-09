"""Reusable versioned pre-execution source policy gate (US-004).

One policy engine shared by every submission path (Evo draft/debug/
improve/crossover clients and the sandbox API admission). It checks the
exact immutable bytes that will be executed, hashes them, binds the policy
version/hash to the verdict and returns machine-readable denials.

This is a static admission filter, NOT a security boundary: keyword
matching can always be evaded. The verdict therefore carries
``security_boundary=False``; real isolation is the worker container
contract (US-006/US-008). Denied submissions must neither be written,
enqueued nor executed; a missing or corrupt policy fails closed (deny).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

POLICY_SCHEMA_VERSION = 1
POLICY_PATH_ENV = "SOURCE_GATE_POLICY_PATH"
DEFAULT_POLICY_PATH = Path(__file__).resolve().with_name("source_policy.v1.json")

CHECK_KIND = "static_admission_filter"

# Environment keys whose VALUES are sandbox paths; the value is checked for
# traversal and protected components just like data_dir/data_paths.
PATH_VALUED_ENV_KEYS = frozenset({"DATA_DIR", "SANDBOX_DATA_DIR"})


class SourcePolicyError(Exception):
    """Raised when the policy artifact is missing or invalid (fail closed)."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class GateDenial:
    """One machine-readable denial: stable rule id plus human reason."""

    rule: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "reason": self.reason}


@dataclass(frozen=True)
class SourcePolicy:
    """Versioned, immutable source policy parsed from the JSON artifact."""

    policy_version: str
    policy_sha256: str
    max_source_bytes: int
    denied_source_markers: tuple[str, ...]
    denied_env_keys: frozenset[str]
    denied_env_prefixes: tuple[str, ...]
    allow_execution_command: bool
    requirement_pattern: re.Pattern[str]
    protected_path_components: frozenset[str]
    security_boundary: bool = False


@dataclass(frozen=True)
class SourceVerdict:
    """Machine-readable outcome of gating the exact submitted bytes."""

    allowed: bool
    policy_version: str
    policy_sha256: str
    source_sha256: str
    denials: tuple[GateDenial, ...] = field(default_factory=tuple)
    security_boundary: bool = False
    check_kind: str = CHECK_KIND

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "source_sha256": self.source_sha256,
            "denials": [d.to_dict() for d in self.denials],
            "security_boundary": self.security_boundary,
            "check_kind": self.check_kind,
        }


def _require_str_list(raw: Any, field_name: str) -> list[str]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise SourcePolicyError(f"policy_field_invalid:{field_name}")
    return list(raw)


def parse_policy(raw: Mapping[str, Any], policy_sha256: str) -> SourcePolicy:
    """Validate a decoded policy document; any deviation fails closed."""
    if raw.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise SourcePolicyError("policy_schema_version_unsupported")
    policy_version = raw.get("policy_version")
    if not isinstance(policy_version, str) or not policy_version:
        raise SourcePolicyError("policy_field_invalid:policy_version")
    max_source_bytes = raw.get("max_source_bytes")
    if not isinstance(max_source_bytes, int) or isinstance(max_source_bytes, bool) or max_source_bytes <= 0:
        raise SourcePolicyError("policy_field_invalid:max_source_bytes")
    markers = _require_str_list(raw.get("denied_source_markers"), "denied_source_markers")
    env_keys = _require_str_list(raw.get("denied_env_keys"), "denied_env_keys")
    env_prefixes = _require_str_list(raw.get("denied_env_prefixes"), "denied_env_prefixes")
    components = _require_str_list(
        raw.get("protected_path_components"), "protected_path_components"
    )
    allow_execution_command = raw.get("allow_execution_command")
    if not isinstance(allow_execution_command, bool):
        raise SourcePolicyError("policy_field_invalid:allow_execution_command")
    pattern_raw = raw.get("requirement_pattern")
    if not isinstance(pattern_raw, str) or not pattern_raw:
        raise SourcePolicyError("policy_field_invalid:requirement_pattern")
    try:
        requirement_pattern = re.compile(pattern_raw)
    except re.error as exc:
        raise SourcePolicyError("policy_field_invalid:requirement_pattern") from exc
    return SourcePolicy(
        policy_version=policy_version,
        policy_sha256=policy_sha256,
        max_source_bytes=max_source_bytes,
        denied_source_markers=tuple(markers),
        denied_env_keys=frozenset(key.upper() for key in env_keys),
        denied_env_prefixes=tuple(prefix.upper() for prefix in env_prefixes),
        allow_execution_command=allow_execution_command,
        requirement_pattern=requirement_pattern,
        protected_path_components=frozenset(comp.lower() for comp in components),
        security_boundary=False,
    )


def load_policy(path: str | os.PathLike[str] | None = None) -> SourcePolicy:
    """Load and validate the policy artifact; missing/corrupt fails closed.

    The path comes from the server-side environment only
    (``SOURCE_GATE_POLICY_PATH``) or defaults to the bundled versioned
    artifact next to this module. Candidate submissions can never reach it:
    the API rejects job environment keys with the ``SOURCE_GATE_`` prefix.
    """
    if path is None:
        path = os.environ.get(POLICY_PATH_ENV) or DEFAULT_POLICY_PATH
    policy_path = Path(path)
    try:
        raw_bytes = policy_path.read_bytes()
    except OSError as exc:
        raise SourcePolicyError(f"policy_unavailable:{type(exc).__name__}") from exc
    try:
        decoded = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourcePolicyError("policy_corrupt") from exc
    if not isinstance(decoded, dict):
        raise SourcePolicyError("policy_corrupt")
    return parse_policy(decoded, _sha256(raw_bytes))


def policy_unavailable_verdict(reason: str, source: bytes = b"") -> SourceVerdict:
    """Fail-closed verdict used when the policy artifact cannot be loaded."""
    return SourceVerdict(
        allowed=False,
        policy_version="unavailable",
        policy_sha256="",
        source_sha256=_sha256(source),
        denials=(GateDenial(rule="policy_unavailable", reason=reason),),
    )


def check_source_bytes(source: bytes, policy: SourcePolicy) -> SourceVerdict:
    """Gate the exact immutable submitted bytes against the policy.

    Marker matching is case-insensitive substring filtering — an advisory
    admission filter, never represented as a security boundary.
    """
    source_sha256 = _sha256(source)
    denials: list[GateDenial] = []
    if not source.strip():
        denials.append(GateDenial(rule="empty_source", reason="submitted source is empty"))
    if len(source) > policy.max_source_bytes:
        denials.append(
            GateDenial(
                rule="source_too_large",
                reason=f"{len(source)} bytes exceeds limit {policy.max_source_bytes}",
            )
        )
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError:
        denials.append(GateDenial(rule="source_not_utf8", reason="source is not valid UTF-8"))
        text = ""
    lowered = text.lower()
    for marker in policy.denied_source_markers:
        if marker.lower() in lowered:
            denials.append(
                GateDenial(
                    rule="denied_marker",
                    reason=f"forbidden_marker:{marker}",
                )
            )
    return SourceVerdict(
        allowed=not denials,
        policy_version=policy.policy_version,
        policy_sha256=policy.policy_sha256,
        source_sha256=source_sha256,
        denials=tuple(denials),
    )


def check_source_tree(path: str | os.PathLike[str], policy: SourcePolicy) -> SourceVerdict:
    """Check a bounded immutable capture; all assets contribute to identity."""
    from .source_snapshot import CaptureError, capture_source

    try:
        return capture_source(path, policy).verdict
    except CaptureError as exc:
        return SourceVerdict(
            allowed=False, policy_version=policy.policy_version,
            policy_sha256=policy.policy_sha256, source_sha256="",
            denials=(GateDenial(exc.rule, exc.reason),),
        )


def _path_denials(raw_path: str, field_name: str, policy: SourcePolicy) -> list[GateDenial]:
    denials: list[GateDenial] = []
    parts = PurePath(raw_path).parts
    if any(part == ".." for part in parts):
        denials.append(
            GateDenial(
                rule="path_traversal",
                reason=f"{field_name} contains '..' traversal: {raw_path!r}",
            )
        )
    lowered = {part.lower() for part in parts}
    for component in sorted(policy.protected_path_components):
        if component in lowered:
            denials.append(
                GateDenial(
                    rule="path_protected_component",
                    reason=f"{field_name} may not reference protected component {component!r}: {raw_path!r}",
                )
            )
    return denials


def validate_job_fields(
    policy: SourcePolicy,
    *,
    execution_command: str | None = None,
    requirements: Sequence[str] = (),
    environment: Mapping[str, str] | None = None,
    working_dir: str | None = None,
    code_file_path: str | None = None,
    data_paths: Sequence[str] = (),
    data_dir: str | None = None,
) -> tuple[GateDenial, ...]:
    """Validate the non-code submission fields before admission.

    These fields are alternate execution paths that would bypass a
    code-only check: ``execution_command`` runs arbitrary shell,
    ``requirements`` can pull attacker code (URLs/git/local paths),
    ``environment`` can hijack the loader (LD_PRELOAD, PYTHONPATH) or
    override server-assigned execution fields and gate/evaluator/budget
    configuration, and path fields can point at evaluator/answer trees.
    Every rejection carries a machine-readable rule id.
    """
    denials: list[GateDenial] = []

    if execution_command is not None and not policy.allow_execution_command:
        denials.append(
            GateDenial(
                rule="execution_command_denied",
                reason="custom execution_command is an alternate execution path and is not allowed by policy",
            )
        )

    for entry in requirements:
        if not isinstance(entry, str) or not policy.requirement_pattern.match(entry):
            denials.append(
                GateDenial(
                    rule="requirement_not_allowlisted",
                    reason=f"requirement must be a pinned PyPI-style spec, got: {entry!r}",
                )
            )

    for key, value in (environment or {}).items():
        if not isinstance(key, str):
            denials.append(
                GateDenial(rule="environment_key_invalid", reason=f"non-string key: {key!r}")
            )
            continue
        upper = key.upper()
        if upper in policy.denied_env_keys:
            denials.append(
                GateDenial(
                    rule="environment_key_denied",
                    reason=f"environment key {key!r} is denied by policy",
                )
            )
        elif any(upper.startswith(prefix) for prefix in policy.denied_env_prefixes):
            denials.append(
                GateDenial(
                    rule="environment_key_denied",
                    reason=f"environment key {key!r} matches a denied prefix",
                )
            )
        if upper in PATH_VALUED_ENV_KEYS and isinstance(value, str) and value:
            denials.extend(_path_denials(value, f"environment[{key}]", policy))

    if data_dir:
        denials.extend(_path_denials(data_dir, "data_dir", policy))

    if working_dir:
        denials.extend(_path_denials(working_dir, "working_dir", policy))
    if code_file_path:
        denials.extend(_path_denials(code_file_path, "code_file_path", policy))
    for raw in data_paths:
        if not isinstance(raw, str):
            denials.append(
                GateDenial(rule="path_invalid", reason=f"data_paths entry not a string: {raw!r}")
            )
            continue
        denials.extend(_path_denials(raw, "data_paths", policy))

    return tuple(denials)
