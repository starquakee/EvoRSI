"""Supervisor: freezing must not follow aliased evidence or overwrite unrelated bytes."""
from pathlib import Path

import pytest

from research.comparison.protocol import ComparisonSpecError
from research.comparison.schedule import build_schedule, freeze_schedule, render_schedule
from research.comparison.specs import iter_cells
from research.tests.test_comparison_specs import COMMIT, make_bindings


def test_existing_schedule_symlink_is_refused_even_with_matching_bytes(tmp_path: Path) -> None:
    cells = iter_cells(make_bindings(), COMMIT)
    outside = tmp_path / "external-evidence.json"
    rendered = render_schedule(build_schedule(cells), COMMIT)
    outside.write_text(rendered)
    target = tmp_path / "schedule.json"
    target.symlink_to(outside)
    with pytest.raises((ComparisonSpecError, OSError)):
        freeze_schedule(target, cells)
    assert outside.read_text() == rendered


def test_preexisting_temporary_symlink_never_overwrites_its_target(tmp_path: Path) -> None:
    cells = iter_cells(make_bindings(), COMMIT)
    outside = tmp_path / "unrelated-evidence.json"
    outside.write_text("original evidence")
    target = tmp_path / "schedule.json"
    target.with_name(target.name + ".tmp").symlink_to(outside)
    try:
        freeze_schedule(target, cells)
    except (ComparisonSpecError, OSError):
        pass
    assert outside.read_text() == "original evidence"
    assert not target.is_symlink()
