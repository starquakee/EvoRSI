"""Offline rendering tests for research/comparison/plots.py (EXP-008).

Render the ACTUAL figures from report-shaped payloads and verify the
rendered PNG bytes (signature, dimensions, non-trivial content), for a
full matrix, a partially observed matrix and an empty (not-started) one.
No network, no display (Agg), no experiment mutation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from research.comparison.plots import render_report_plots
from research.comparison.protocol import canonical_protocol

PROTO = canonical_protocol()


def _row(task_id: str, seed: int, method: str, utility, with_curve: bool) -> dict:
    if with_curve:
        curve = [
            {
                "position": index,
                "config_index": index,
                "cumulative_requests": 6 * (index + 1),
                "cumulative_charged_tokens": 600 * (index + 1),
                "validation_incumbent": round(0.5 + 0.01 * index, 6),
            }
            for index in range(8)
        ]
    else:
        curve = []
    return {
        "cell_key": f"{task_id}:{seed}:{method}",
        "task_id": task_id,
        "method": method,
        "search_seed": seed,
        "utility": utility,
        "incumbent_curve": curve,
    }


def _report(rows: list[dict]) -> dict:
    return {"schema": "test", "matrix": {"rows": rows}}


def _full_rows() -> list[dict]:
    return [
        _row(task.task_id, seed, method, 0.6 + 0.01 * seed, True)
        for task in PROTO.tasks
        for seed in PROTO.search_seeds
        for method in PROTO.methods
    ]


def _assert_png(path: Path) -> None:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 4000  # a real rendered figure, not an empty stub
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(data))
    assert image.width > 400 and image.height > 300


class TestRender:
    def test_full_matrix_renders_three_figures(self, tmp_path):
        written = render_report_plots(_report(_full_rows()), tmp_path)
        assert set(written) == {
            "paired-utility.png",
            "incumbent-vs-requests.png",
            "incumbent-vs-tokens.png",
        }
        for name, digest in written.items():
            path = tmp_path / name
            _assert_png(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

    def test_partial_matrix_renders_with_absent_data(self, tmp_path):
        rows = _full_rows()
        # One seed block entirely unobserved: utilities and curves absent.
        rows = [
            (
                _row(r["task_id"], r["search_seed"], r["method"], None, False)
                if r["search_seed"] == 22
                else r
            )
            for r in rows
        ]
        written = render_report_plots(_report(rows), tmp_path)
        assert len(written) == 3
        for name in written:
            _assert_png(tmp_path / name)

    def test_empty_matrix_renders_annotated_empty_panels(self, tmp_path):
        rows = [
            _row(task.task_id, seed, method, None, False)
            for task in PROTO.tasks
            for seed in PROTO.search_seeds
            for method in PROTO.methods
        ]
        written = render_report_plots(_report(rows), tmp_path)
        assert len(written) == 3
        for name in written:
            _assert_png(tmp_path / name)

    def test_symlinked_output_refused(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        link.symlink_to(target, target_is_directory=True)
        from research.comparison.plots import ComparisonPlotError

        with pytest.raises(ComparisonPlotError, match="plot_output_symlinked"):
            render_report_plots(_report(_full_rows()), link)
