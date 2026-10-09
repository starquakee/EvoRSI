"""Prediction reads must remain confined across file-system races."""
import os
import pytest
from research.evaluator.service import read_prediction_file, EvaluatorError, EvaluatorRegistry, score_submission

def test_swap_to_symlink_before_open_rejected(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    prediction = scratch / "submission.csv"
    prediction.write_bytes(b"original")
    outside = tmp_path / "outside.csv"
    outside.write_bytes(b"outside")
    original_open = os.open
    swapped = []
    def swap_open(path, flags, *args, **kwargs):
        if path == "submission.csv" and not swapped:
            swapped.append(True)
            prediction.unlink()
            prediction.symlink_to(outside)
        return original_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", swap_open)
    with pytest.raises(EvaluatorError) as error:
        read_prediction_file(prediction, allowed_root=scratch, max_bytes=1024)
    assert error.value.reason == "symlink_rejected"

def test_parent_swapped_after_open_cannot_redirect(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    prediction = scratch / "submission.csv"
    prediction.write_bytes(b"original")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "submission.csv").write_bytes(b"outside")
    original_open = os.open
    swapped = []
    def swap_open(path, flags, *args, **kwargs):
        if path == "submission.csv" and not swapped:
            swapped.append(True)
            scratch.rename(tmp_path / "moved")
            scratch.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(os, "open", swap_open)
    assert read_prediction_file(prediction, allowed_root=scratch, max_bytes=1024) == b"original"

def test_read_growth_is_bounded(tmp_path, monkeypatch):
    prediction = tmp_path / "submission.csv"
    prediction.write_bytes(b"small")
    original_read = os.read
    read_sizes = []
    def growing_read(fd, size):
        if not read_sizes:
            prediction.write_bytes(b"x" * 4096)
        data = original_read(fd, size)
        read_sizes.append(len(data))
        return data
    monkeypatch.setattr(os, "read", growing_read)
    with pytest.raises(EvaluatorError) as error:
        read_prediction_file(prediction, allowed_root=tmp_path, max_bytes=32)
    assert error.value.reason == "prediction_oversized"
    assert sum(read_sizes) <= 33

def test_fifo_does_not_block(tmp_path):
    prediction = tmp_path / "submission.csv"
    os.mkfifo(prediction)
    with pytest.raises(EvaluatorError) as error:
        read_prediction_file(prediction, allowed_root=tmp_path, max_bytes=1024)
    assert error.value.reason == "not_regular_file"

def test_unclosed_csv_quote_rejected():
    registry = EvaluatorRegistry.load()
    spec = registry.task("hello_synth")
    rows = spec.answer_path.read_text().strip().splitlines()
    identifier, label = rows[-1].split(",")
    rows[-1] = identifier + ',"' + label
    malformed = ("\n".join(rows) + "\n").encode()
    with pytest.raises(EvaluatorError) as error:
        score_submission(registry, "hello_synth", malformed)
    assert error.value.reason == "prediction_malformed"
