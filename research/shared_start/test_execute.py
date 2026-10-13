"""Fake-child lifecycle tests use the real isolated ledger and no live clients."""
import json
from pathlib import Path

import pytest

from research.shared_start import execute as controller
from research.shared_start.ledger import CalibrationLedger
from research.shared_start.specs import CalibrationError, LEDGER, RUNTIME, trials
from research.comparison.budget import ProviderUsage


def setup_controller(tmp_path, monkeypatch, *, interrupt_unit=None, invalid_pool=False, drift_after=False):
    root = Path(__file__).resolve().parents[2]
    roster = trials(root, 'a' * 40)
    manifest = {'identity': 'b' * 64, 'source_commit': 'a' * 40}
    runtime = tmp_path / RUNTIME
    runtime.mkdir(parents=True)
    (runtime / 'preflight.json').write_text(json.dumps({'historical': {}}))
    monkeypatch.setattr(controller, 'check_review', lambda _: manifest)
    monkeypatch.setattr(controller, 'trials', lambda *_: roster)
    history_checks = []
    def historical(_):
        history_checks.append(True)
        if drift_after and len(history_checks) > 1:
            raise RuntimeError('synthetic history mismatch')
        return {}
    monkeypatch.setattr(controller, 'historical_integrity', historical)
    monkeypatch.setattr(controller.phase, 'verify_idle', lambda *args, **kwargs: None)
    invoked = []
    def fake_child(repo_root, spec_path, trial_id, timeout):
        trial = next(item for item in roster if item.trial_id == trial_id)
        assert 0 < timeout <= trial.limits['max_elapsed_seconds']
        invoked.append(trial_id)
        with CalibrationLedger.open(repo_root / LEDGER, roster=roster, manifest_identity=manifest['identity']) as ledger:
            if trial.kind == 'continuation':
                # Coordinator must seal the verified pool before this child exists.
                assert trial.pool_id in ledger.snapshot()['seed_pools']
                assert (runtime / 'runs' / trial.pool_id / 'seed-bundle.json').is_file()
            ledger.begin_trial(trial)
            for _ in range(trial.main_target):
                reserved = ledger.reserve(trial, 'main', estimated_input_tokens=100)
                if interrupt_unit is not None and len(invoked) == interrupt_unit:
                    raise RuntimeError('synthetic interrupted child')
                ledger.commit(reserved.request_id, ProviderUsage(1, 1))
            ledger.finish_trial(trial, 'completed')
        return 0
    monkeypatch.setattr(controller, 'run_child', fake_child)
    def verify(root, manifest, trial, snapshot):
        if invalid_pool and trial.kind == 'fresh_seed_pool':
            raise CalibrationError('pool_terminal_candidate_invalid')
        score = .5 + trial.record['repeat'] / 128
        if trial.kind == 'continuation':
            score += (trial.config_index + 1) / 32
        result = {'normalized': {'trial_id': trial.trial_id, 'status': 'completed',
            'validation_accuracy': score, 'requests': trial.main_target,
            'charged_tokens': 2 * trial.main_target, 'reason': 'completed'}, 'synthetic': True}
        if trial.kind == 'fresh_seed_pool':
            result['pool_bundle'] = {'pool_id': trial.trial_id, 'synthetic': True}
        return result
    monkeypatch.setattr(controller, 'validate_trial', verify)
    return roster, invoked


def test_units_execute_in_frozen_order_pool_charged_once_and_no_second_launch(tmp_path, monkeypatch):
    roster, invoked = setup_controller(tmp_path, monkeypatch)
    result = controller.execute(tmp_path)
    assert result['status'] == 'completed'
    assert invoked == [trial.trial_id for trial in roster]
    assert result['budget']['global']['committed'] == {'requests': 114, 'tokens': 228}
    assert result['analysis']['seed_pool_cost'] == {'requests': 6, 'charged_tokens': 12}
    assert result['analysis']['verdict'] == 'exploratory_stable_configuration_signal'
    before = (tmp_path / LEDGER).read_bytes()
    with pytest.raises(CalibrationError, match='already_started'):
        controller.execute(tmp_path)
    assert (tmp_path / LEDGER).read_bytes() == before
    assert len(invoked) == 30


def test_failed_pool_never_generates_continuation_or_replacement(tmp_path, monkeypatch):
    _, invoked = setup_controller(tmp_path, monkeypatch, invalid_pool=True)
    result = controller.execute(tmp_path)
    assert invoked == ['b00-pool']
    assert result['status'] == 'incomplete'
    assert result['budget']['global']['committed'] == {'requests': 2, 'tokens': 4}
    assert result['analysis']['recorded_cost'] == {'requests': 2, 'charged_tokens': 4}
    assert result['analysis']['verdict'] == 'insufficient_complete_evidence'
    assert not result['budget']['seed_pools']
    assert not (tmp_path / RUNTIME / 'runs/b00-pool/seed-bundle.json').exists()


def test_interrupted_continuation_cost_retained_and_no_retry(tmp_path, monkeypatch):
    _, invoked = setup_controller(tmp_path, monkeypatch, interrupt_unit=2)
    result = controller.execute(tmp_path)
    assert len(invoked) == 2
    assert result['status'] == 'incomplete'
    assert result['budget']['global']['committed'] == {'requests': 3, 'tokens': 8296}
    assert result['analysis']['recorded_cost'] == {'requests': 3, 'charged_tokens': 8296}
    assert result['emergency_cleanup']['cleanup_verified'] is True


def test_integrity_failure_still_emits_incomplete_result(tmp_path, monkeypatch):
    setup_controller(tmp_path, monkeypatch, drift_after=True)
    result = controller.execute(tmp_path)
    assert result['status'] == 'incomplete'
    assert result['historical_check_error'] == 'RuntimeError'
    assert result['analysis']['verdict'] == 'integrity_or_cleanup_unproven'
    assert (tmp_path / RUNTIME / 'outcome.json').is_file()
