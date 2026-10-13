import copy

import pytest

from research.shared_start.analysis import analyze
from research.shared_start.specs import CalibrationError, selected_plan


def complete_outcomes():
    outcomes = []
    for unit in selected_plan()['schedule']:
        score = .5 + unit['repeat'] * .01
        if unit['kind'] == 'continuation':
            score += (int(unit['config_id'][1:]) + 1) * .01
        count = unit['limits']['max_main_requests']
        outcomes.append({'trial_id': unit['trial_id'], 'status': 'completed',
            'validation_accuracy': score, 'requests': count, 'charged_tokens': 100 * count,
            'reason': 'completed'})
    return outcomes


def test_pool_overhead_is_once_per_block_not_once_per_configuration():
    report = analyze(complete_outcomes())
    assert report['recorded_cost'] == {'requests': 114, 'charged_tokens': 11400}
    assert report['seed_pool_cost'] == {'requests': 6, 'charged_tokens': 600}
    assert report['continuation_cost'] == {'requests': 108, 'charged_tokens': 10800}
    assert report['expected_trials'] == 27
    assert report['independent_starting_pools'] == 3
    assert report['verdict'] == 'exploratory_stable_configuration_signal'
    assert report['gain_summaries'][0]['mean_gain_pp'] == pytest.approx(1)
    assert all(row['spearman_two_block_mean_vs_held_out'] == 1 for row in report['leave_one_block_out'])


def test_failed_pool_keeps_cost_and_prevents_stable_verdict():
    first = complete_outcomes()[0]
    first.update(status='failed', validation_accuracy=None, requests=3, charged_tokens=5000)
    report = analyze([first])
    assert report['recorded_cost'] == {'requests': 3, 'charged_tokens': 5000}
    assert report['verdict'] == 'insufficient_complete_evidence'
    assert len(report['incomplete_trials']) == 27


def test_complete_continuations_without_pool_evidence_cannot_pass():
    outcomes = [row for row in complete_outcomes() if not row['trial_id'].endswith('pool')]
    assert analyze(outcomes)['verdict'] == 'insufficient_complete_evidence'


def test_lost_seed_incumbent_is_rejected():
    outcomes = complete_outcomes()
    outcomes[1]['validation_accuracy'] = .1
    with pytest.raises(CalibrationError, match='lost_initial_incumbent'):
        analyze(outcomes)


@pytest.mark.parametrize('mutation', ['duplicate', 'cost', 'nan', 'extra'])
def test_invalid_input_not_silently_ignored(mutation):
    outcomes = copy.deepcopy(complete_outcomes())
    if mutation == 'duplicate':
        outcomes.append(outcomes[0])
    elif mutation == 'cost':
        outcomes[0]['charged_tokens'] = -1
    elif mutation == 'nan':
        outcomes[0]['validation_accuracy'] = float('nan')
    else:
        outcomes[0]['final_score'] = .99
    with pytest.raises(CalibrationError):
        analyze(outcomes)
