"""Deterministic result cache (US-003).

Cache keys bind every identity that may change an evaluation outcome:
seed, model, task, data version, metric, policy version, config hash and
the submitted source hash. Only fresh *scored* results are stored —
failed/rejected outcomes are never cached, so fail-closed re-checks always
re-run instead of being hidden behind a cache hit.

A hit returns the original record marked cache_origin="cache" with zero
incremental calls/tokens/time, while original provenance (created_at,
recorded_cost, source_hash, policy_version) is retained. A malformed
cache file fails closed (CacheCorruptError) rather than being silently
treated as empty.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .persist import PersistError, atomic_write_json, read_json_object
from .results import (
    ORIGIN_FRESH,
    STATUS_SCORED,
    EvaluationResult,
    hash_bytes,
)


class CacheCorruptError(RuntimeError):
    """The cache file or a cached record is malformed; fail closed."""


@dataclass(frozen=True)
class CacheIdentity:
    """Everything that identifies a unique evaluation."""

    seed: int
    model: str
    task_id: str
    data_version: str
    metric: str
    policy_version: str
    config_hash: str
    source_hash: str

    def canonical(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "model": self.model,
                "task_id": self.task_id,
                "data_version": self.data_version,
                "metric": self.metric,
                "policy_version": self.policy_version,
                "config_hash": self.config_hash,
                "source_hash": self.source_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def key(self) -> str:
        return hash_bytes(self.canonical().encode("utf-8"))


class ResultCache:
    """File-backed scored-result cache keyed by CacheIdentity."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path):
        self._path = Path(path)
        self._records: dict[str, dict[str, Any]] = {}
        if self._path.exists():
            try:
                payload = read_json_object(self._path)
            except PersistError as exc:
                raise CacheCorruptError(str(exc)) from exc
            if payload.get("schema_version") != self.SCHEMA_VERSION:
                raise CacheCorruptError("unsupported_schema_version")
            records = payload.get("records")
            if not isinstance(records, dict):
                raise CacheCorruptError("malformed_records")
            for key, record in records.items():
                self._records[key] = self._validate_record(key, record)

    @staticmethod
    def _validate_record(key: str, record: Any) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise CacheCorruptError(f"malformed_record:{key}")
        try:
            EvaluationResult.from_dict(record)
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheCorruptError(f"invalid_record:{key}") from exc
        return record

    def __len__(self) -> int:
        return len(self._records)

    def get(self, identity: CacheIdentity) -> EvaluationResult | None:
        """Return the cached hit (zero incremental cost) or None on a miss."""
        record = self._records.get(identity.key())
        if record is None:
            return None
        result = EvaluationResult.from_dict(record)
        if (result.source_hash != identity.source_hash
            or result.policy_version != identity.policy_version
            or result.metric != identity.metric
            or result.status != STATUS_SCORED):
            raise CacheCorruptError("cache_identity_mismatch")
        return result.mark_cached()

    def put(self, identity: CacheIdentity, result: EvaluationResult) -> None:
        """Store a fresh scored result; anything else is refused (fail closed)."""
        if result.status != STATUS_SCORED or result.cache_origin != ORIGIN_FRESH:
            raise ValueError("only_fresh_scored_results_are_cached")
        if result.source_hash != identity.source_hash:
            raise ValueError("source_hash_mismatch")
        if result.policy_version != identity.policy_version or result.metric != identity.metric:
            raise ValueError("cache_identity_mismatch")
        self._records[identity.key()] = result.to_dict()
        self._save()

    def _save(self) -> None:
        atomic_write_json(
            self._path,
            {"schema_version": self.SCHEMA_VERSION, "records": self._records},
        )
