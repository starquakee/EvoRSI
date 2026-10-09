"""Operator-call trace events for search-parameter effectiveness evidence (US-007).

Stdlib only. One JSONL line per Evo operator call (draft/improve/debug/
crossover) carrying the child code hash, parent linkage and a hashed config
snapshot, so downstream analysis can attribute search-trajectory outcomes to
concrete search-parameter and operator choices without trusting the journal
alone.

Events fail closed: :func:`validate_operator_event` raises ``ValueError``
naming the first missing/bad field, and :class:`OperatorTraceWriter` validates
before writing, so a malformed event can never enter the trace silently.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Union

OPERATOR_TRACE_SCHEMA = "operator-trace.v1"

TRACE_OPERATORS = ("draft", "improve", "debug", "crossover")

REQUIRED_EVENT_FIELDS = (
    "event",
    "schema",
    "operator",
    "generation_id",
    "mode",
    "child_node_id",
    "child_code_sha256",
    "parent_node_ids",
    "parent_steps",
    "parent_code_sha256",
    "selection",
    "config",
)

_HEX64 = "0123456789abcdef"


def sha256_hex(text_or_bytes: Union[str, bytes]) -> str:
    """Return the lowercase hex SHA-256 of a str (utf-8) or bytes payload."""
    if isinstance(text_or_bytes, str):
        data = text_or_bytes.encode("utf-8")
    else:
        data = bytes(text_or_bytes)
    return hashlib.sha256(data).hexdigest()


def _is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in _HEX64 for ch in value)
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _plain(value: Any) -> Any:
    """Convert mapping/sequence-like config values (incl. OmegaConf nodes)
    into plain JSON-serializable Python containers without importing any
    third-party package."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except (TypeError, AttributeError):
            pass
    return getattr(config, key, default)


def config_snapshot(cfg: Any) -> Dict[str, Any]:
    """Extract the search-parameter snapshot of an EvolutionarySolverConfig.

    All fields are read via getattr-with-default so partially populated
    configs still snapshot. ``config_sha256`` is the SHA-256 of the canonical
    JSON (sort_keys, compact separators) of the snapshot WITHOUT the hash
    field itself.
    """
    experience = _config_get(cfg, "experience", {}) or {}
    parent_selection = _config_get(experience, "parent_selection", {}) or {}
    prompt_memory = _config_get(experience, "prompt_memory", {}) or {}
    snapshot: Dict[str, Any] = {
        "num_islands": _config_get(cfg, "num_islands"),
        "max_island_size": _config_get(cfg, "max_island_size"),
        "crossover_prob": _config_get(cfg, "crossover_prob"),
        "migration_prob": _config_get(cfg, "migration_prob"),
        "initial_temp": _config_get(cfg, "initial_temp"),
        "final_temp": _config_get(cfg, "final_temp"),
        "num_generations_till_migration": _config_get(
            cfg, "num_generations_till_migration"
        ),
        "num_generations_till_crossover": _config_get(
            cfg, "num_generations_till_crossover"
        ),
        "num_generations": _config_get(cfg, "num_generations"),
        "individuals_per_generation": _config_get(
            cfg, "individuals_per_generation"
        ),
        "max_debug_depth": _config_get(cfg, "max_debug_depth"),
        "fresh_draft_prob": _config_get(cfg, "fresh_draft_prob", 0.0),
        "few_shot": _plain(_config_get(cfg, "few_shot", {})),
        "execution_mode": _config_get(cfg, "execution_mode", "generation"),
        "experience": {
            "enabled": bool(_config_get(experience, "enabled", False)),
            "parent_selection": {
                "weights": _plain(_config_get(parent_selection, "weights", {})),
            },
            "prompt_memory": {
                "enabled": bool(_config_get(prompt_memory, "enabled", False)),
            },
        },
    }
    snapshot = _plain(snapshot)
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    snapshot["config_sha256"] = sha256_hex(canonical)
    return snapshot


def validate_operator_event(ev: Dict[str, Any]) -> None:
    """Fail closed on a malformed trace event.

    Raises ``ValueError`` naming the first missing or invalid field.
    Returns ``None`` for a valid event.
    """
    if not isinstance(ev, dict):
        raise ValueError("event: must be a dict")
    for field in REQUIRED_EVENT_FIELDS:
        if field not in ev:
            raise ValueError(f"{field}: missing required field")
    if ev["event"] != "evo_operator_call":
        raise ValueError("event: must be 'evo_operator_call'")
    if ev["schema"] != OPERATOR_TRACE_SCHEMA:
        raise ValueError(f"schema: must be {OPERATOR_TRACE_SCHEMA!r}")
    if ev["operator"] not in TRACE_OPERATORS:
        raise ValueError(f"operator: must be one of {TRACE_OPERATORS}")
    if not _is_int(ev["generation_id"]) or ev["generation_id"] < 0:
        raise ValueError("generation_id: must be an int >= 0")
    if not isinstance(ev["mode"], str) or not ev["mode"]:
        raise ValueError("mode: must be a non-empty str")
    if not isinstance(ev["child_node_id"], str) or not ev["child_node_id"]:
        raise ValueError("child_node_id: must be a non-empty str")
    if not _is_hex64(ev["child_code_sha256"]):
        raise ValueError("child_code_sha256: must be a 64-char hex str")
    parent_node_ids = ev["parent_node_ids"]
    parent_steps = ev["parent_steps"]
    parent_hashes = ev["parent_code_sha256"]
    if not isinstance(parent_node_ids, list):
        raise ValueError("parent_node_ids: must be a list")
    if not isinstance(parent_steps, list):
        raise ValueError("parent_steps: must be a list")
    if not isinstance(parent_hashes, list):
        raise ValueError("parent_code_sha256: must be a list")
    if not (len(parent_node_ids) == len(parent_steps) == len(parent_hashes)):
        raise ValueError(
            "parent_node_ids: parent lists must have equal length"
        )
    for step in parent_steps:
        if not _is_int(step):
            raise ValueError("parent_steps: every step must be an int")
    for digest in parent_hashes:
        if not _is_hex64(digest):
            raise ValueError("parent_code_sha256: every hash must be 64-hex")
    config = ev["config"]
    if not isinstance(config, dict):
        raise ValueError("config: must be a dict")
    if not _is_hex64(config.get("config_sha256")):
        raise ValueError("config: config_sha256 must be a 64-char hex str")


class OperatorTraceWriter:
    """Append-only JSONL writer usable directly as the tracer callable.

    Every event is validated (fail closed) before one flushed line is
    appended, so the trace on disk only ever contains well-formed events.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: Dict[str, Any]) -> None:
        validate_operator_event(event)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
