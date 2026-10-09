"""Unit tests for the trusted independent evaluator (US-005).

Covers: registry loading (fail closed), bounded CSV scoring against the
pinned hello_synth fixture, every machine-readable rejection (unknown
task, oversize, malformed, schema/row/id/label violations), answer
integrity, prediction-file defenses (symlink/traversal/oversize),
worker-mount plan validation, and the structural guarantee that the
evaluator never executes or imports candidate code. Fully offline.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from research.contracts.results import MetricDirection
from research.evaluator import service
from research.evaluator.metrics import METRIC_FUNCTIONS, accuracy
from research.evaluator.service import (
    DEFAULT_REGISTRY_PATH,
    EvaluatorError,
    EvaluatorRegistry,
    read_prediction_file,
    score_submission,
    validate_worker_mount_plan,
)

ANSWER_PATH = (
    Path(__file__).resolve().parents[1] / "evaluator" / "tasks" / "hello_synth" / "test_answer.csv"
)
ANSWER_BYTES = ANSWER_PATH.read_bytes()
ANSWER_ROWS = list(csv.reader(io.StringIO(ANSWER_BYTES.decode("utf-8"))))
ANSWER_HEADER, ANSWER_BODY = ANSWER_ROWS[0], ANSWER_ROWS[1:]


def make_prediction(labels: list[int]) -> bytes:
    lines = ["id,label"] + [f"{row[0]},{label}" for row, label in zip(ANSWER_BODY, labels)]
    return ("\n".join(lines) + "\n").encode("utf-8")


TRUE_LABELS = [int(row[1]) for row in ANSWER_BODY]


@pytest.fixture(scope="module")
def registry() -> EvaluatorRegistry:
    return EvaluatorRegistry.load()


# --- registry loading (fail closed) ---


def test_default_registry_loads_hello_synth(registry: EvaluatorRegistry) -> None:
    spec = registry.task("hello_synth")
    assert spec.metric == "accuracy"
    assert spec.direction is MetricDirection.MAXIMIZE
    assert spec.split == "test"  # validation/test split is explicit
    assert spec.answer_sha256 == hashlib.sha256(ANSWER_BYTES).hexdigest()
    assert registry.registry_version == "evaluator-registry.v1"


def test_registry_missing_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(EvaluatorError) as excinfo:
        EvaluatorRegistry.load(tmp_path / "absent.json")
    assert excinfo.value.reason == "registry_invalid"


def test_registry_corrupt_json_fails_closed(tmp_path: Path) -> None:
    bad = tmp_path / "registry.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(EvaluatorError) as excinfo:
        EvaluatorRegistry.load(bad)
    assert excinfo.value.reason == "registry_invalid"


def test_registry_wrong_schema_fails_closed(tmp_path: Path) -> None:
    bad = tmp_path / "registry.json"
    bad.write_text(json.dumps({"schema_version": 999, "tasks": {"x": {}}}), encoding="utf-8")
    with pytest.raises(EvaluatorError) as excinfo:
        EvaluatorRegistry.load(bad)
    assert excinfo.value.reason == "registry_invalid"


def _write_registry(tmp_path: Path, task_entry: dict) -> Path:
    (tmp_path / "answer.csv").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "answer.csv").write_bytes(ANSWER_BYTES)
    entry = {
        "metric": "accuracy",
        "split": "test",
        "answer_file": "answer.csv",
        "answer_sha256": hashlib.sha256(ANSWER_BYTES).hexdigest(),
        "id_column": "id",
        "target_column": "label",
        "label_values": [0, 1],
        "max_prediction_bytes": 1024,
        "max_rows": 100,
    }
    entry.update(task_entry)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "registry_version": "evaluator-registry.v1",
                "schema_version": 1,
                "tasks": {"t": entry},
            }
        ),
        encoding="utf-8",
    )
    return registry_path


def test_registry_rejects_untrusted_metric_name(tmp_path: Path) -> None:
    # A registry can never point at candidate-supplied metric code: only
    # in-repo trusted metric function names load.
    path = _write_registry(tmp_path, {"metric": "tasks/hello_synth/utils/metric.py"})
    with pytest.raises(EvaluatorError) as excinfo:
        EvaluatorRegistry.load(path)
    assert excinfo.value.reason == "registry_invalid"
    assert "metric.py" not in METRIC_FUNCTIONS


def test_registry_rejects_invalid_split(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, {"split": "hidden"})
    with pytest.raises(EvaluatorError) as excinfo:
        EvaluatorRegistry.load(path)
    assert excinfo.value.reason == "registry_invalid"


def test_registry_accepts_validation_split(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, {"split": "validation"})
    registry = EvaluatorRegistry.load(path)
    assert registry.task("t").split == "validation"


def test_registry_rejects_answer_traversal(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, {"answer_file": "../escape.csv"})
    with pytest.raises(EvaluatorError) as excinfo:
        EvaluatorRegistry.load(path)
    assert excinfo.value.reason == "registry_invalid"


def test_registry_answer_integrity_checked_at_load(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, {"answer_sha256": "0" * 64})
    with pytest.raises(EvaluatorError) as excinfo:
        EvaluatorRegistry.load(path)
    assert excinfo.value.reason == "answer_integrity_failed"


def test_answer_tamper_after_load_fails_closed(tmp_path: Path) -> None:
    path = _write_registry(tmp_path, {})
    registry = EvaluatorRegistry.load(path)
    # Tamper with the answer bytes AFTER the registry was loaded; scoring
    # re-verifies integrity and must fail closed instead of scoring.
    answer_file = tmp_path / "answer.csv"
    answer_file.write_bytes(answer_file.read_bytes().replace(b"200,0", b"200,1", 1))
    with pytest.raises(EvaluatorError) as excinfo:
        score_submission(registry, "t", make_prediction(TRUE_LABELS))
    assert excinfo.value.reason == "answer_integrity_failed"


# --- scoring ---


def test_perfect_prediction_scores_one(registry: EvaluatorRegistry) -> None:
    outcome = score_submission(registry, "hello_synth", make_prediction(TRUE_LABELS))
    assert outcome.score == 1.0
    assert outcome.direction is MetricDirection.MAXIMIZE
    assert outcome.split == "test"
    assert outcome.row_count == len(ANSWER_BODY)
    assert outcome.prediction_sha256 == hashlib.sha256(make_prediction(TRUE_LABELS)).hexdigest()
    assert outcome.answer_sha256 == hashlib.sha256(ANSWER_BYTES).hexdigest()


def test_all_wrong_prediction_scores_zero(registry: EvaluatorRegistry) -> None:
    flipped = [1 - label for label in TRUE_LABELS]
    outcome = score_submission(registry, "hello_synth", make_prediction(flipped))
    assert outcome.score == 0.0


def test_partial_prediction_matches_expected_accuracy(registry: EvaluatorRegistry) -> None:
    labels = list(TRUE_LABELS)
    labels[0] = 1 - labels[0]
    labels[1] = 1 - labels[1]
    outcome = score_submission(registry, "hello_synth", make_prediction(labels))
    expected = round((len(TRUE_LABELS) - 2) / len(TRUE_LABELS), 6)
    assert outcome.score == expected == accuracy(TRUE_LABELS, labels)


def test_unknown_task_rejected(registry: EvaluatorRegistry) -> None:
    with pytest.raises(EvaluatorError) as excinfo:
        score_submission(registry, "not_a_task", make_prediction(TRUE_LABELS))
    assert excinfo.value.reason == "task_not_allowlisted"


# --- malformed / hostile prediction bytes ---


def test_oversized_prediction_rejected(registry: EvaluatorRegistry) -> None:
    spec = registry.task("hello_synth")
    big = b"id,label\n" + b"1,1\n" * spec.max_prediction_bytes
    assert len(big) > spec.max_prediction_bytes
    with pytest.raises(EvaluatorError) as excinfo:
        score_submission(registry, "hello_synth", big)
    assert excinfo.value.reason == "prediction_oversized"


@pytest.mark.parametrize(
    "payload",
    [
        b"",  # empty
        b"\xff\xfe\x00bad",  # not utf-8
        b"id,label\n",  # header only, no rows
        b"label,id\n1,1\n",  # wrong header order
        b"id,label,extra\n200,1,0\n",  # extra column
        b"id\n200\n",  # missing column
        b"id,label\n200,1\n201,0,5\n",  # ragged row
        b"id,label\nabc,1\n",  # non-integer id
        b"id,label\n200,x\n",  # non-integer label
    ],
)
def test_malformed_predictions_rejected(registry: EvaluatorRegistry, payload: bytes) -> None:
    with pytest.raises(EvaluatorError) as excinfo:
        score_submission(registry, "hello_synth", payload)
    assert excinfo.value.reason in {
        "prediction_malformed",
        "submission_schema_mismatch",
        "submission_row_mismatch",
    }


def test_row_count_mismatch_rejected(registry: EvaluatorRegistry) -> None:
    short = b"id,label\n" + b"200,1\n"
    with pytest.raises(EvaluatorError) as excinfo:
        score_submission(registry, "hello_synth", short)
    assert excinfo.value.reason == "submission_row_mismatch"


def test_id_order_mismatch_rejected(registry: EvaluatorRegistry) -> None:
    swapped = list(TRUE_LABELS)
    lines = ["id,label"]
    body = list(ANSWER_BODY)
    body[0], body[1] = body[1], body[0]
    for row, label in zip(body, swapped):
        lines.append(f"{row[0]},{label}")
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    with pytest.raises(EvaluatorError) as excinfo:
        score_submission(registry, "hello_synth", payload)
    assert excinfo.value.reason == "submission_id_mismatch"


def test_out_of_range_label_rejected(registry: EvaluatorRegistry) -> None:
    labels = list(TRUE_LABELS)
    labels[0] = 2
    with pytest.raises(EvaluatorError) as excinfo:
        score_submission(registry, "hello_synth", make_prediction(labels))
    assert excinfo.value.reason == "submission_label_invalid"


def test_prediction_bytes_never_executed(registry: EvaluatorRegistry) -> None:
    # Prediction content that would be valid Python is still just data.
    payload = b"id,label\nprint('pwned')\n"
    with pytest.raises(EvaluatorError):
        score_submission(registry, "hello_synth", payload)


# --- prediction file defenses ---


def test_read_prediction_file_ok(tmp_path: Path) -> None:
    target = tmp_path / "submission.csv"
    target.write_bytes(ANSWER_BYTES)
    assert read_prediction_file(target, allowed_root=tmp_path, max_bytes=1 << 20) == ANSWER_BYTES


def test_read_prediction_file_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.csv"
    real.write_bytes(ANSWER_BYTES)
    link = tmp_path / "link.csv"
    link.symlink_to(real)
    with pytest.raises(EvaluatorError) as excinfo:
        read_prediction_file(link, allowed_root=tmp_path, max_bytes=1 << 20)
    assert excinfo.value.reason == "symlink_rejected"


def test_read_prediction_file_rejects_symlinked_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "answer.csv"
    secret.write_bytes(ANSWER_BYTES)
    root = tmp_path / "root"
    root.mkdir()
    (root / "alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(EvaluatorError) as excinfo:
        read_prediction_file(root / "alias" / "answer.csv", allowed_root=root, max_bytes=1 << 20)
    assert excinfo.value.reason == "symlink_rejected"


def test_read_prediction_file_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(EvaluatorError) as excinfo:
        read_prediction_file(tmp_path / "sub" / ".." / "x.csv", allowed_root=tmp_path, max_bytes=1 << 20)
    assert excinfo.value.reason == "path_traversal"


def test_read_prediction_file_rejects_outside_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside.csv"
    outside.write_bytes(ANSWER_BYTES)
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(EvaluatorError) as excinfo:
        read_prediction_file(outside, allowed_root=root, max_bytes=1 << 20)
    assert excinfo.value.reason == "path_traversal"


def test_read_prediction_file_rejects_oversize_before_read(tmp_path: Path) -> None:
    target = tmp_path / "big.csv"
    target.write_bytes(b"x" * 4096)
    with pytest.raises(EvaluatorError) as excinfo:
        read_prediction_file(target, allowed_root=tmp_path, max_bytes=1024)
    assert excinfo.value.reason == "prediction_oversized"


def test_read_prediction_file_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(EvaluatorError) as excinfo:
        read_prediction_file(tmp_path, allowed_root=tmp_path, max_bytes=1 << 20)
    assert excinfo.value.reason == "not_regular_file"


# --- worker mount plan validation ---


def test_mount_plan_denies_evaluator_root_and_parent_and_subpath(tmp_path: Path) -> None:
    evaluator_root = tmp_path / "evaluator" / "tasks"
    evaluator_root.mkdir(parents=True)
    sibling = tmp_path / "public"
    sibling.mkdir()
    denials = validate_worker_mount_plan(
        [
            str(evaluator_root),  # exact
            str(tmp_path / "evaluator"),  # parent
            str(tmp_path),  # grandparent
            str(evaluator_root / "hello_synth"),  # sub-path / alternate path
            str(sibling),  # unrelated: allowed
        ],
        evaluator_root=evaluator_root,
    )
    reasons = {denial.reason for denial in denials}
    assert len(denials) == 4
    assert reasons == {
        "evaluator_root_mounted",
        "evaluator_ancestor_mounted",
        "evaluator_subpath_mounted",
    }
    assert all(denial.source != str(sibling) for denial in denials)


def test_mount_plan_resolves_symlinked_source(tmp_path: Path) -> None:
    evaluator_root = tmp_path / "evaluator"
    evaluator_root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(evaluator_root, target_is_directory=True)
    denials = validate_worker_mount_plan([str(alias)], evaluator_root=evaluator_root)
    assert [denial.reason for denial in denials] == ["evaluator_root_mounted"]


def test_mount_plan_allows_unrelated_readonly_public_data(tmp_path: Path) -> None:
    evaluator_root = tmp_path / "evaluator"
    evaluator_root.mkdir()
    public = tmp_path / "tasks" / "hello_synth" / "data" / "public"
    public.mkdir(parents=True)
    assert validate_worker_mount_plan([str(public)], evaluator_root=evaluator_root) == []


# --- structural guarantee: evaluator never runs candidate code ---


def test_evaluator_source_has_no_dynamic_execution() -> None:
    banned = ("importlib", "exec(", "eval(", "subprocess", "os.system", "__import__")
    for module in (service.__file__,):
        text = Path(module).read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{token} found in {module}"
    assert set(METRIC_FUNCTIONS) == {"accuracy"}
