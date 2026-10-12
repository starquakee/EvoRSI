"""Regression: prepare must preserve an already-held final data leaf.

The held Linux directory descriptor models the directory object pinned by
an existing bind mount. No Docker mutation, real final data or model call.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from research.comparison import deploy, preflight, review
from research.tests.test_comparison_preflight import COMMIT, FakeNative

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('task', ('banknote', 'wdbc', 'spambase'))
def test_prepare_keeps_existing_final_leaf_visible(
    task: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(deploy, 'git_head_commit', lambda root: COMMIT)
    monkeypatch.setattr(review, '_git_porcelain', lambda root: '')
    base = REPO_ROOT / '.runtime' / 'test-comparison-mount-supervisor'
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as temporary:
        runtime = Path(temporary)
        preflight.create_final_placeholders(runtime)
        leaf = runtime / 'final-publication' / task / 'data' / 'final'
        descriptor = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY)
        try:
            original = os.fstat(descriptor)
            assert original.st_ino == leaf.stat().st_ino
            evidence = preflight.prepare_experiment(
                REPO_ROOT, provider=FakeNative(), runtime_dir=runtime,
                now_fn=lambda: 1000.0,
            )
            assert evidence['cells_initialized'] == 27
            assert not (runtime / review.LEDGER_FILENAME).exists()
            sentinel = 'synthetic-visibility-probe.txt'
            payload = b'synthetic bytes, no real final data\n'
            (leaf / sentinel).write_bytes(payload)
            try:
                visible = os.open(sentinel, os.O_RDONLY, dir_fd=descriptor)
            except FileNotFoundError:
                pytest.fail(
                    'prepare replaced the pre-existing final bind-source leaf; '
                    'a holder of the original directory cannot see new data'
                )
            else:
                try:
                    assert os.read(visible, len(payload) + 1) == payload
                finally:
                    os.close(visible)
        finally:
            os.close(descriptor)
