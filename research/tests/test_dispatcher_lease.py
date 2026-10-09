"""Unified cross-queue worker lease tests (US-006).

The rsi-trustworthy stack registers ONE physical worker in both the cpu and
gpu Redis queues. These tests prove the dispatcher's unified lease
(SET NX keyed by endpoint) serializes execution across the two dispatch
loops: a worker leased by one resource loop cannot be acquired by the
other until the lease is released, leases are released on every
_run_worker_job exit path, stale leases are cleared at dispatcher init,
and completion markers live in the per-job scratch tree in local-scratch
mode (so the worker's job-storage mount can stay read-only).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DISPATCHER_PATH = REPO_ROOT / "sandbox-controller" / "task_dispatcher" / "task_dispatcher.py"
ENDPOINT = "http://worker:8080"


class FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}

    # strings
    def set(self, key, value, nx: bool = False):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def get(self, key):
        return self.kv.get(key)

    def delete(self, key) -> None:
        self.kv.pop(key, None)
        self.lists.pop(key, None)

    def scan_iter(self, match: str = "*"):
        prefix = match.rstrip("*")
        for key in list(self.kv):
            if key.startswith(prefix):
                yield key

    # lists
    def lpop(self, key):
        items = self.lists.get(key) or []
        if not items:
            return None
        value = items.pop(0)
        return value

    def rpush(self, key, value) -> None:
        self.lists.setdefault(key, []).append(value)

    def lrem(self, key, count, value) -> None:
        items = self.lists.get(key) or []
        self.lists[key] = [item for item in items if item != value]

    def llen(self, key) -> int:
        return len(self.lists.get(key) or [])


class FakeConn:
    def cursor(self):
        raise AssertionError("DB should not be touched in these tests")

    def close(self) -> None:
        return None


@pytest.fixture()
def td(monkeypatch, tmp_path):
    """Load a fresh task_dispatcher instance with one endpoint in BOTH queues."""
    config = tmp_path / "sandbox_config.json"
    config.write_text(
        json.dumps({"cpu": {"endpoints": [ENDPOINT]}, "gpu": {"endpoints": [ENDPOINT]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("SANDBOX_CONFIG_FILE", str(config))
    monkeypatch.setenv("LOCAL_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("EXTERNAL_EVALUATOR_ENABLED", "0")
    monkeypatch.delenv("EVALUATOR_REGISTRY_PATH", raising=False)
    module_name = "task_dispatcher_lease_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, DISPATCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_same_endpoint_registered_in_both_queues(td):
    assert td.SANDBOX_ENDPOINTS_BY_TYPE["cpu"] == [ENDPOINT]
    assert td.SANDBOX_ENDPOINTS_BY_TYPE["gpu"] == [ENDPOINT]


def test_lease_blocks_second_queue_acquisition(td):
    redis_client = FakeRedis()
    td.initialize_worker_queue(redis_client)

    endpoint, worker_type = td.acquire_worker_for_job(redis_client, "cpu")
    assert endpoint == ENDPOINT
    assert worker_type == "cpu"
    assert redis_client.get(td.worker_lease_key(ENDPOINT)) == "cpu"

    # The gpu loop still sees the endpoint in its own queue, but the lease
    # must deny the acquisition and return the endpoint to the gpu queue.
    endpoint2, worker_type2 = td.acquire_worker_for_job(redis_client, "gpu")
    assert (endpoint2, worker_type2) == (None, None)
    assert redis_client.llen("available_workers_gpu") == 1


def test_release_allows_other_queue(td):
    redis_client = FakeRedis()
    td.initialize_worker_queue(redis_client)

    td.acquire_worker_for_job(redis_client, "gpu")
    td.release_worker_lease(redis_client, ENDPOINT)
    endpoint, worker_type = td.acquire_worker_for_job(redis_client, "cpu")
    assert (endpoint, worker_type) == (ENDPOINT, "cpu")


def test_initialize_clears_stale_leases(td):
    redis_client = FakeRedis()
    redis_client.set(td.worker_lease_key(ENDPOINT), "cpu", nx=True)
    td.initialize_worker_queue(redis_client)
    assert redis_client.get(td.worker_lease_key(ENDPOINT)) is None
    endpoint, _ = td.acquire_worker_for_job(redis_client, "cpu")
    assert endpoint == ENDPOINT


def test_run_worker_job_releases_lease_on_exception(td, monkeypatch):
    redis_client = FakeRedis()
    td.initialize_worker_queue(redis_client)
    endpoint, worker_type = td.acquire_worker_for_job(redis_client, "cpu")
    assert endpoint == ENDPOINT

    def boom(redis_conn, db_conn, worker_endpoint, job_data):
        raise RuntimeError("simulated worker failure")

    monkeypatch.setattr(td, "execute_job", boom)
    monkeypatch.setattr(td, "get_redis_connection", lambda: redis_client)
    monkeypatch.setattr(td, "get_db_connection", lambda: FakeConn())

    td._run_worker_job(endpoint, worker_type, {"job_id": "job-lease-1"})

    assert redis_client.get(td.worker_lease_key(ENDPOINT)) is None
    assert redis_client.llen("available_workers_cpu") == 1


def test_run_worker_job_releases_lease_on_quarantine(td, monkeypatch):
    redis_client = FakeRedis()
    td.initialize_worker_queue(redis_client)
    endpoint, worker_type = td.acquire_worker_for_job(redis_client, "gpu")
    assert endpoint == ENDPOINT

    def boom(redis_conn, db_conn, worker_endpoint, job_data):
        raise td.RequeueJobOnWorkerStartupError(job_data, worker_endpoint, "probe failed")

    monkeypatch.setattr(td, "execute_job", boom)
    monkeypatch.setattr(td, "get_redis_connection", lambda: redis_client)
    monkeypatch.setattr(td, "get_db_connection", lambda: FakeConn())
    monkeypatch.setattr(td, "requeue_job_to_original_queue", lambda *a, **k: "job_queue_gpu_p1")

    td._run_worker_job(endpoint, worker_type, {"job_id": "job-lease-2"})

    assert redis_client.get(td.worker_lease_key(ENDPOINT)) is None
    # Quarantined: not returned to the queue until recovery.
    assert redis_client.llen("available_workers_gpu") == 0
    assert ENDPOINT in td.QUARANTINED_WORKERS
    td.QUARANTINED_WORKERS.clear()


def test_markers_live_in_scratch_in_local_scratch_mode(td, tmp_path):
    storage_dir = Path(tempfile.mkdtemp(prefix="us006-storage-")) / "jobs" / "j1"
    scratch_base = td.marker_base_dir("j1", storage_dir, True)
    assert str(scratch_base).startswith(str(td.LOCAL_SCRATCH_ROOT))
    assert td.marker_base_dir("j1", storage_dir, False) == storage_dir
