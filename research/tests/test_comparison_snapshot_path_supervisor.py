"""Real dated producer paths are valid; another job's symlinked snapshot is not."""
from __future__ import annotations

from pathlib import Path

import pytest

from research.comparison import deploy
from research.comparison.protocol import ComparisonSpecError

JOB = 'a' * 16
OTHER = 'b' * 16


@pytest.mark.parametrize('middle', ('', '2026-10-11/final/'))
def test_own_snapshot_path_accepts_flat_and_actual_dated_layout(
    middle: str, tmp_path: Path
) -> None:
    snapshot = f'{deploy.CONTAINER_JOBS_PREFIX}{middle}{JOB}/prediction_snapshot.csv'
    relative = snapshot[len(deploy.CONTAINER_STORAGE_PREFIX):]
    original = tmp_path / relative
    original.parent.mkdir(parents=True)
    original.write_bytes(b'id,label\nx,0\n')
    assert deploy.job_snapshot_host_path(snapshot, JOB, storage_root=tmp_path) == original


@pytest.mark.parametrize('symlink_level', ('file', 'job_directory'))
def test_snapshot_host_path_rejects_symlink_into_another_job(
    symlink_level: str, tmp_path: Path
) -> None:
    snapshot = f'{deploy.CONTAINER_JOBS_PREFIX}2026-10-11/final/{JOB}/prediction_snapshot.csv'
    relative = snapshot[len(deploy.CONTAINER_STORAGE_PREFIX):]
    original = tmp_path / relative
    original.parent.mkdir(parents=True)
    original.write_bytes(b'id,label\nx,0\n')
    # Required positive control: valid same-job dated output was accepted first.
    assert deploy.job_snapshot_host_path(snapshot, JOB, storage_root=tmp_path) == original
    other = original.parent.parent / OTHER / 'prediction_snapshot.csv'
    other.parent.mkdir()
    other.write_bytes(original.read_bytes())
    original.unlink()
    if symlink_level == 'file':
        original.symlink_to(other)
    else:
        original.parent.rmdir()
        original.parent.symlink_to(other.parent, target_is_directory=True)
    assert original.resolve() == other
    with pytest.raises(ComparisonSpecError):
        deploy.job_snapshot_host_path(snapshot, JOB, storage_root=tmp_path)
