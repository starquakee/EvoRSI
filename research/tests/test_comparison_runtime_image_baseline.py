"""Frozen image-ID baseline: a same-tag changed image must be refused (EXP-007).

The reviewed manifest freezes the ACTUAL local image IDs. inspect_runtime_stack
must compare the inspected image ID against that immutable baseline: accepting
the same name:tag after the underlying image bytes changed would silently
refresh a drifted baseline. Positive control first, then ONLY the image ID
changes (tag/mounts/env identical) and must refuse.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from research.comparison import preflight, review
from research.comparison.review import ComparisonManifestError

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which('docker') is None, reason='docker CLI unavailable')

FROZEN_IMAGE_IDS = {service: 'sha256:' + 'a' * 64 for service in review.STACK_IMAGE_PINS}


def _snapshots() -> dict[str, dict[str, Any]]:
    config = preflight._compose_config(REPO_ROOT)
    snapshots: dict[str, dict[str, Any]] = {}
    for name, image in preflight.EXPECTED_STACK_IMAGES.items():
        service = config['services'][name.removeprefix('rsi_')]
        snapshots[name] = {
            'Id': name + '-fixture', 'Image': 'sha256:' + 'a' * 64,
            'State': {'Running': True, 'StartedAt': '2026-10-10T00:00:00Z'},
            'Config': {'Image': image, 'Env': ['EVALUATOR_REGISTRY_PATH=/opt/rsi_repo/research/evaluator/registry.v1.json']},
            'Mounts': [{'Source': v['source'], 'Destination': v['target'], 'RW': not v.get('read_only', False)} for v in service.get('volumes', [])],
        }
    return snapshots


def _install(monkeypatch: pytest.MonkeyPatch, snapshots: dict[str, dict[str, Any]]) -> None:
    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert argv[:2] == ['docker', 'inspect']
        return subprocess.CompletedProcess(argv, 0, json.dumps([snapshots[argv[2]]]), '')

    monkeypatch.setattr(preflight.subprocess, 'run', fake_run)


@pytest.mark.parametrize('container', ('rsi_worker', 'rsi_dispatcher'))
def test_same_tag_changed_image_id_refused(
    container: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots = _snapshots()
    _install(monkeypatch, snapshots)
    # Positive control: actual IDs equal to the frozen baseline are accepted.
    assert preflight.inspect_runtime_stack(
        REPO_ROOT, expected_images=FROZEN_IMAGE_IDS
    )['phase'] == 'search'
    # Mutation: ONLY the inspected image ID changes; the name:tag stays.
    changed = 'sha256:' + 'b' * 64
    snapshots[container]['Image'] = changed
    assert snapshots[container]['Config']['Image'] == preflight.EXPECTED_STACK_IMAGES[container]
    with pytest.raises(ComparisonManifestError):
        preflight.inspect_runtime_stack(REPO_ROOT, expected_images=FROZEN_IMAGE_IDS)


def test_baseline_missing_service_entry_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots = _snapshots()
    _install(monkeypatch, snapshots)
    assert preflight.inspect_runtime_stack(
        REPO_ROOT, expected_images=FROZEN_IMAGE_IDS
    )['phase'] == 'search'
    incomplete = {k: v for k, v in FROZEN_IMAGE_IDS.items() if k != 'worker'}
    with pytest.raises(ComparisonManifestError):
        preflight.inspect_runtime_stack(REPO_ROOT, expected_images=incomplete)
