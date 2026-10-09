"""US-009 run-index tests: reuse only complete, verified, intact runs."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from research.search.run_index import (
    SCHEMA,
    RunIdentity,
    RunIndex,
    RunIndexError,
    RunRecord,
)

POLICY_SHA = "b" * 64
DATA_ID = "d" * 64
REGISTRY_ID = "e" * 64
ANSWER_SHA = "a" * 64
PRED1_BYTES = b"id,label\nprediction-one\n"
PRED2_BYTES = b"id,label\nprediction-two\n"
PREDICTION_SHA = hashlib.sha256(PRED1_BYTES).hexdigest()
PRED2_SHA = hashlib.sha256(PRED2_BYTES).hexdigest()
TRACE_BYTES = b'{"event": "operator", "operator": "draft"}\n'
REQUEST_TRACE_BYTES = b'{"reservation_id": "r1"}\n'
JOURNAL_BYTES = b'{"node": 1, "code": "print(1)"}\n'


def _identity(**overrides: Any) -> RunIdentity:
    base: dict[str, Any] = dict(
        seed=42,
        model="k3",
        task_id="hello_synth",
        metric="accuracy",
        policy_version="source-gate.v2",
        config_hash="c" * 64,
        runner_commit="d" * 40,
        policy_sha256=POLICY_SHA,
        data_identity=DATA_ID,
        registry_identity=REGISTRY_ID,
    )
    base.update(overrides)
    return RunIdentity(**base)


def _result_payload(score: float = 0.55) -> dict[str, Any]:
    return {
        "schema": "acceptance-inner-result.v1",
        "run_id": "cfg0",
        "config_hash": "c" * 64,
        "policy_version": "source-gate.v2",
        "policy_sha256": POLICY_SHA,
        "status": "completed",
        "termination": "completed",
        "best_score": score,
        "best_job_id": "job_1",
        "main_candidates": 6,
        "generations_completed": 3,
        "budget": {"stopped": None, "open_reservations": []},
        "prediction_artifacts": {"job_1": PREDICTION_SHA, "job_2": PRED2_SHA},
        "jobs": [
            {
                "job_id": "job_1",
                "status": "completed",
                "score": score,
                "evidence_verified": True,
                "evaluation": {
                    "prediction_sha256": PREDICTION_SHA,
                    "answer_sha256": ANSWER_SHA,
                },
            },
            {
                "job_id": "job_2",
                "status": "completed",
                "score": 0.4,
                "evidence_verified": True,
                "evaluation": {
                    "prediction_sha256": PRED2_SHA,
                    "answer_sha256": ANSWER_SHA,
                },
            },
        ],
    }


def _materialize_run_dir(root, run_dir: str = "runs/cfg0", score: float = 0.55) -> dict[str, str]:
    """Write the bound artifacts (traces, journal, run result) under
    root/run_dir and return their sha256 digests."""
    directory = root / run_dir
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "operator-trace.jsonl").write_bytes(TRACE_BYTES)
    (directory / "request-trace.jsonl").write_bytes(REQUEST_TRACE_BYTES)
    (directory / "journal.jsonl").write_bytes(JOURNAL_BYTES)
    predictions = directory / "predictions"
    predictions.mkdir(exist_ok=True)
    (predictions / "job_1.csv").write_bytes(PRED1_BYTES)
    (predictions / "job_2.csv").write_bytes(PRED2_BYTES)
    result_bytes = json.dumps(_result_payload(score), sort_keys=True).encode("utf-8")
    (directory / "run-result.json").write_bytes(result_bytes)
    return {
        "operator-trace.jsonl": hashlib.sha256(TRACE_BYTES).hexdigest(),
        "request-trace.jsonl": hashlib.sha256(REQUEST_TRACE_BYTES).hexdigest(),
        "journal.jsonl": hashlib.sha256(JOURNAL_BYTES).hexdigest(),
        "run-result.json": hashlib.sha256(result_bytes).hexdigest(),
    }


def _record(root=None, **overrides: Any) -> RunRecord:
    score = overrides.get("score", 0.55)
    hashes = _materialize_run_dir(root, score=score) if root is not None else {
        "operator-trace.jsonl": "a" * 64,
        "request-trace.jsonl": "a" * 64,
        "journal.jsonl": "a" * 64,
        "run-result.json": "a" * 64,
    }
    base: dict[str, Any] = dict(
        identity=_identity(),
        status="completed",
        score=0.55,
        run_dir="runs/cfg0",
        run_result_sha256=hashes["run-result.json"],
        journal_sha256=hashes["journal.jsonl"],
        request_trace_sha256=hashes["request-trace.jsonl"],
        main_candidates=6,
        generations_completed=3,
        best_job_id="job_1",
        job_ids=("job_1", "job_2"),
        prediction_artifacts={"job_1": PREDICTION_SHA, "job_2": PRED2_SHA},
        operator_trace_sha256=hashes["operator-trace.jsonl"],
        worker_cleanup_verified=True,
        operator_counts={"draft": 6, "improve": 1, "debug": 1, "crossover": 1},
        termination="completed",
        created_at=1_800_000_000.0,
    )
    base.update(overrides)
    return RunRecord(**base)


def test_put_get_roundtrip(tmp_path):
    index = RunIndex(tmp_path / "index.json")
    record = _record(tmp_path)
    index.put(record)
    assert len(index) == 1
    assert index.get(record.identity) == record
    reloaded = RunIndex(tmp_path / "index.json")
    assert reloaded.get(record.identity) == record


def test_identity_fields_change_the_key(tmp_path):
    index = RunIndex(tmp_path / "index.json")
    index.put(_record(tmp_path))
    for change in (
        {"seed": 43},
        {"model": "other"},
        {"task_id": "other"},
        {"metric": "logloss"},
        {"policy_version": "source-gate.v1"},
        {"config_hash": "e" * 64},
        {"runner_commit": "f" * 40},
        {"policy_sha256": "0" * 64},
        {"data_identity": "0" * 64},
        {"registry_identity": "0" * 64},
    ):
        assert index.get(_identity(**change)) is None


def test_idempotent_put_same_record(tmp_path):
    index = RunIndex(tmp_path / "index.json")
    record = _record(tmp_path)
    index.put(record)
    index.put(record)
    assert len(index) == 1


def test_conflicting_record_refused(tmp_path):
    index = RunIndex(tmp_path / "index.json")
    index.put(_record(tmp_path))
    # A fully consistent record with a different score (artifacts rebuilt to
    # match) collides with the indexed identity.
    with pytest.raises(RunIndexError, match="run_index_conflict"):
        index.put(_record(tmp_path, score=0.9))


@pytest.mark.parametrize(
    "overrides,rule",
    [
        ({"status": "failed"}, "run_incomplete_not_indexable"),
        ({"score": None}, "run_score_invalid"),
        ({"score": float("nan")}, "run_score_invalid"),
        ({"score": 1.5}, "run_score_invalid"),
        ({"score": -0.1}, "run_score_invalid"),
        ({"job_ids": ()}, "run_job_ids_missing"),
        ({"worker_cleanup_verified": False}, "run_cleanup_unverified"),
        ({"operator_trace_sha256": ""}, "run_artifact_hash_invalid:operator-trace.jsonl"),
        ({"operator_trace_sha256": "z" * 64}, "run_artifact_hash_invalid:operator-trace.jsonl"),
        ({"main_candidates": 5}, "run_main_candidates_missing"),
        ({"generations_completed": 2}, "run_generations_incomplete"),
        ({"operator_counts": {"draft": 4, "debug": 4}}, "run_operator_coverage_insufficient"),
        ({"best_job_id": "job_unknown"}, "run_best_job_unbound"),
    ],
)
def test_unverified_or_incomplete_records_refused(tmp_path, overrides, rule):
    index = RunIndex(tmp_path / "index.json")
    with pytest.raises(RunIndexError, match=rule):
        index.put(_record(tmp_path, **overrides))
    assert len(index) == 0
    assert not (tmp_path / "index.json").exists()


@pytest.mark.parametrize(
    "field", ["policy_sha256", "data_identity", "registry_identity"]
)
def test_identity_content_binding_required(tmp_path, field):
    identity = _identity(**{field: ""})
    index = RunIndex(tmp_path / "index.json")
    with pytest.raises(RunIndexError, match=f"run_identity_unbound:{field}"):
        index.put(_record(tmp_path, identity=identity))


def test_corrupt_file_fails_closed(tmp_path):
    path = tmp_path / "index.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RunIndexError, match="run_index_corrupt"):
        RunIndex(path)


def test_schema_mismatch_fails_closed(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"schema": "other", "records": {}}), encoding="utf-8")
    with pytest.raises(RunIndexError, match="run_index_schema_mismatch"):
        RunIndex(path)


def test_record_under_wrong_key_fails_closed(tmp_path):
    index = RunIndex(tmp_path / "index.json")
    record = _record(tmp_path)
    index.put(record)
    path = tmp_path / "index.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    entry = raw["records"].pop(record.identity.key())
    raw["records"]["x" * 64] = entry
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RunIndexError, match="run_index_identity_mismatch"):
        RunIndex(path)


def test_malformed_persisted_record_fails_closed(tmp_path):
    path = tmp_path / "index.json"
    record = _record(tmp_path)
    payload = {"schema": SCHEMA, "records": {record.identity.key(): {"status": "completed"}}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunIndexError, match="run_record_malformed|run_identity_malformed"):
        RunIndex(path)


def test_nonboolean_cleanup_flag_rejected(tmp_path):
    path = tmp_path / "index.json"
    record = _record(tmp_path)
    data = record.to_dict()
    data["worker_cleanup_verified"] = "true"  # string must not coerce
    payload = {"schema": SCHEMA, "records": {record.identity.key(): data}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunIndexError, match="run_record_malformed"):
        RunIndex(path)


@pytest.mark.parametrize(
    "artifact",
    ["operator-trace.jsonl", "request-trace.jsonl", "journal.jsonl", "run-result.json"],
)
def test_post_index_tamper_invalidates_get(tmp_path, artifact):
    """Revalidation on EVERY get: changing any bound artifact after put must
    invalidate the hit (operator trace, request trace, journal, result)."""
    index = RunIndex(tmp_path / "index.json")
    record = _record(tmp_path)
    index.put(record)
    (tmp_path / record.run_dir / artifact).write_bytes(b"tampered\n")
    with pytest.raises(RunIndexError, match="run_artifact_mismatch"):
        index.get(record.identity)


def test_run_result_content_tampered_semantics(tmp_path):
    """A run-result whose hash matches nothing parseable fails closed."""
    index = RunIndex(tmp_path / "index.json")
    record = _record(tmp_path)
    index.put(record)
    result_path = tmp_path / record.run_dir / "run-result.json"
    payload = _result_payload(score=0.99)  # score no longer matches record
    result_path.write_bytes(json.dumps(payload, sort_keys=True).encode())
    with pytest.raises(RunIndexError, match="run_artifact_mismatch:run-result.json"):
        index.get(record.identity)


def test_run_result_stopped_ledger_rejected(tmp_path):
    from research.search.run_index import validate_run_result_consistency

    record = _record(tmp_path)
    payload = _result_payload()
    payload["budget"]["stopped"] = {"reason": "unknown_usage"}
    with pytest.raises(RunIndexError, match="run_ledger_stopped"):
        validate_run_result_consistency(payload, record)
    payload = _result_payload()
    payload["budget"]["open_reservations"] = [{"reservation_id": "x"}]
    with pytest.raises(RunIndexError, match="run_ledger_unresolved_reservations"):
        validate_run_result_consistency(payload, record)


def test_run_result_job_evidence_required(tmp_path):
    from research.search.run_index import validate_run_result_consistency

    record = _record(tmp_path)
    payload = _result_payload()
    payload["jobs"][0]["evaluation"]["prediction_sha256"] = ""
    with pytest.raises(RunIndexError, match="run_job_prediction_unbound"):
        validate_run_result_consistency(payload, record)
    payload = _result_payload()
    payload["jobs"][0]["evidence_verified"] = False
    payload["jobs"][1]["evidence_verified"] = False
    with pytest.raises(RunIndexError, match="run_best_result_untrusted"):
        validate_run_result_consistency(payload, record)


def test_run_dir_traversal_rejected(tmp_path):
    index = RunIndex(tmp_path / "index.json")
    with pytest.raises(RunIndexError, match="run_dir_invalid"):
        index.put(_record(tmp_path, run_dir="../elsewhere"))
