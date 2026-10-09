"""Atomic JSON persistence helpers shared by cache and budget ledger (US-003).

All writes go through a temporary file + os.replace so a crash mid-write
never leaves a truncated ledger/cache behind. Reads are strict: malformed
JSON or wrong top-level type fails closed instead of being silently
treated as empty state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class PersistError(RuntimeError):
    """Corrupt or unreadable persisted state; callers must fail closed."""


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write payload as JSON atomically (tmp file in same dir + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object; fail closed on missing/invalid/non-object data."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PersistError(f"unreadable:{path.name}:{type(exc).__name__}") from exc
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PersistError(f"malformed_json:{path.name}") from exc
    if not isinstance(obj, dict):
        raise PersistError(f"unexpected_json_type:{path.name}:{type(obj).__name__}")
    return obj
