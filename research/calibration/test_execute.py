"""Coordinator lifecycle tests with a fake child, real isolated budget ledger."""
import json
from pathlib import Path

import pytest

from research.calibration import execute as controller
from research.calibration.ledger import CalibrationLedger
from research.calibration.specs import CalibrationError, LEDGER, RUNTIME, trials
from research.comparison.budget import ProviderUsage


def setup_controller(tmp_path, monkeypatch, *, fail_second=False):
    root = Path(__file__).resolve().parents[2]
    roster = trials(root, 'a' * 40)
    manifest = {'identity': 'b' * 64, 'source_commit': 'a' * 40}
    runtime = tmp_path / RUNTIME
    runtime.mkdir(parents=True)
    (runtime / 'preflight.json').write_text(json.dumps({'historical': {}}))
    monkeypatch.setattr(controller, 'check_review', lambda _: manifest)
    monkeypatch.setattr(controller, 'trials', lambda *_: roster)
    monkeypatch.setattr(controller, 'historical_integrity', lambda _: {})
    monkeypatch.setattr(controller.phase, 'verify_idle', lambda *args, **kwargs: None)
    invoked = []
    def fake_child(repo_root, spec_path, trial_id, timeout):
        assert timeout > 0
        invoked.append(trial_id)
        trial = next(item for item in roster if item.trial_id == trial_id)
        with CalibrationLedger.open(repo_root / LEDGER, roster=roster, manifest_identity=manifest['identity']) as ledger:
            ledger.begin_trial(trial)
            for _ in range(6):
                reserved = ledger.reserve(trial, 'main', estimated_input_tokens=100)
                if fail_second and len(invoked) == 2:
                    raise RuntimeError('synthetic interrupted child')
                ledger.commit(reserved.request_id, ProviderUsage(1, 1))
            ledger.finish_trial(trial, 'completed')
        return 0
    monkeypatch.setattr(controller, 'run_child', fake_child)
    def verify(root, manifest, trial, snapshot):
        return {'normalized': {'trial_id': trial.trial_id, 'status': 'completed',
            'validation_accuracy': 0.5 + trial.config_index / 32 + trial.record['repeat'] / 128,
            'requests': 6, 'charged_tokens': 12, 'reason': 'completed'}, 'synthetic': True}
    monkeypatch.setattr(controller, 'validate_trial', verify)
    return roster, invoked


def test_all_trials_execute_once_in_frozen_order_and_second_launch_refuses(tmp_path, monkeypatch):
    roster, invoked = setup_controller(tmp_path, monkeypatch)
    result = controller.execute(tmp_path)
    assert result['status'] == 'completed'
    assert invoked == [trial.trial_id for trial in roster]
    assert result['budget']['global']['committed'] == {'requests': 162, 'tokens': 324}
    assert result['analysis']['verdict'] == 'exploratory_stable_configuration_signal'
    before = (tmp_path / LEDGER).read_bytes()
    with pytest.raises(CalibrationError, match='already_started'):
        controller.execute(tmp_path)
    assert (tmp_path / LEDGER).read_bytes() == before
    assert len(invoked) == 27


def test_interrupted_child_is_not_retried_and_unknown_cost_is_included(tmp_path, monkeypatch):
    _, invoked = setup_controller(tmp_path, monkeypatch, fail_second=True)
    result = controller.execute(tmp_path)
    assert len(invoked) == 2
    assert result['status'] == 'incomplete'
    assert result['analysis']['verdict'] == 'insufficient_complete_evidence'
    assert result['budget']['global']['committed'] == {'requests': 7, 'tokens': 8304}
    assert result['analysis']['recorded_cost'] == {'requests': 7, 'charged_tokens': 8304}
    assert result['emergency_cleanup']['cleanup_verified'] is True
    assert len(result['analysis']['incomplete_trials']) == 26
