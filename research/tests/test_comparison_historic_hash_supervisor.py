"""Original ledger identity must survive changes that preserve accounting totals."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.comparison import preflight
from research.comparison.review import ComparisonManifestError

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('changed_path', preflight.HISTORIC_LEDGERS)
def test_historic_audit_rejects_same_totals_changed_bytes(
    changed_path: str, tmp_path: Path
) -> None:
    for relative in (*preflight.HISTORIC_LEDGERS, preflight.HISTORIC_BASELINE_AUDIT):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / relative).read_bytes())
    # The exact ORIGINAL bytes must pass first, proving the intended gate is reached.
    assert preflight._check_historic_ledgers(tmp_path)
    changed = tmp_path / changed_path
    ledger = json.loads(changed.read_text(encoding='utf-8'))
    committed = dict(ledger['committed'])
    ledger['supervisor_same_totals_tamper'] = True
    changed.write_text(json.dumps(ledger), encoding='utf-8')
    assert json.loads(changed.read_text(encoding='utf-8'))['committed'] == committed
    with pytest.raises(ComparisonManifestError):
        preflight._check_historic_ledgers(tmp_path)
