"""Actual mount SOURCES must match the compose contract, not only targets/RW."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from research.comparison import preflight
from research.comparison.review import ComparisonManifestError

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which('docker') is None, reason='docker CLI unavailable')


@pytest.mark.parametrize('container,target', (
    ('rsi_worker', '/mnt/rsi_data/comparison/banknote/data/public'),
    ('rsi_worker', '/usr/local/lib/python3.10/dist-packages/sklearn/datasets/data/breast_cancer.csv'),
    ('rsi_dispatcher', '/opt/rsi_final'),
))
def test_actual_wrong_mount_source_refused(
    container: str, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Render the tracked compose file only; do not contact/mutate running containers.
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

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert argv[:2] == ['docker', 'inspect']
        return subprocess.CompletedProcess(argv, 0, json.dumps([snapshots[argv[2]]]), '')

    monkeypatch.setattr(preflight.subprocess, 'run', fake_run)
    assert preflight.inspect_runtime_stack(REPO_ROOT)['phase'] == 'search'
    mount = next(m for m in snapshots[container]['Mounts'] if m['Destination'] == target)
    original = dict(mount)
    mount['Source'] = str(REPO_ROOT / 'research' / 'evaluator')
    assert mount['Source'] != original['Source']
    assert mount['Destination'] == original['Destination'] and mount['RW'] == original['RW']
    with pytest.raises(ComparisonManifestError):
        preflight.inspect_runtime_stack(REPO_ROOT)
