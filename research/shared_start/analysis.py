"""Descriptive shared-start analysis with seed-pool overhead counted once."""
from __future__ import annotations

import math
import statistics

from research.calibration.plan import analyze as analyze_balanced, identity, spearman
from research.shared_start.specs import CalibrationError, selected_plan


def continuation_plan():
    selected = selected_plan()
    result = {'schema': 'shared-start-continuation-analysis-plan.v1',
        'parent_plan_identity': selected['plan_identity'],
        'budget': {'repeats': 3, 'trials': 27},
        'configurations': selected['configurations'], 'center_config_id': 'c04',
        'schedule': [item for item in selected['schedule'] if item['kind'] == 'continuation']}
    return {**result, 'plan_identity': identity(result)}


def analyze(outcomes):
    """Outcomes are independently verified elsewhere; failures retain all costs.

    The primary matrix contains 27 continuations. Pool costs are overhead shared
    within each block; they are never duplicated into its nine trial costs.
    """
    selected = selected_plan()
    expected = {unit['trial_id']: unit for unit in selected['schedule']}
    pool_ids = {key for key, unit in expected.items() if unit['kind'] == 'fresh_seed_pool'}
    by_id = {}
    for outcome in outcomes:
        key = outcome.get('trial_id')
        if key not in expected or key in by_id:
            raise CalibrationError('analysis_unknown_or_duplicate_unit')
        if set(outcome) - {'trial_id', 'status', 'validation_accuracy', 'requests', 'charged_tokens', 'reason'}:
            raise CalibrationError('analysis_unexpected_fields')
        if outcome.get('status') not in ('completed', 'failed', 'terminated'):
            raise CalibrationError('analysis_invalid_status')
        value = outcome.get('validation_accuracy')
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
            raise CalibrationError('analysis_invalid_accuracy')
        if any(type(outcome.get(name)) is not int or outcome[name] < 0 for name in ('requests', 'charged_tokens')):
            raise CalibrationError('analysis_invalid_cost')
        by_id[key] = outcome
    continuations = [entry for key, entry in by_id.items() if key not in pool_ids]
    report = analyze_balanced(continuation_plan(), continuations)
    continuation_cost = dict(report['recorded_cost'])
    pool_cost = {'requests': sum(by_id[key]['requests'] for key in pool_ids if key in by_id),
                 'charged_tokens': sum(by_id[key]['charged_tokens'] for key in pool_ids if key in by_id)}
    pool_status = []
    for key in sorted(pool_ids):
        outcome = by_id.get(key)
        pool_status.append({'pool_id': key,
            'status': outcome['status'] if outcome else 'missing',
            'initial_best_validation_accuracy': outcome.get('validation_accuracy') if outcome else None,
            'requests': outcome['requests'] if outcome else 0,
            'charged_tokens': outcome['charged_tokens'] if outcome else 0})
    if any(p['status'] != 'completed' or p['initial_best_validation_accuracy'] is None for p in pool_status):
        report['verdict'] = 'insufficient_complete_evidence'
    report.update(schema='shared-start-analysis.v1', parent_plan_identity=selected['plan_identity'],
        expected_units=30, reported_units=len(by_id), independent_starting_pools=3,
        seed_pool_cost=pool_cost, continuation_cost=continuation_cost,
        recorded_cost={name: continuation_cost[name] + pool_cost[name] for name in pool_cost},
        pools=pool_status,
        limitations='Three independent starting pools, with nine conditional continuations per pool. '
                    'Historical comparison is descriptive, not a randomized causal estimate. '
                    'Validation only; no outer algorithm comparison or unseen-test claim.',
        leave_one_block_out=None, gain_summaries=None)
    if len(by_id) == 30 and all(entry['status'] == 'completed' and entry.get('validation_accuracy') is not None for entry in by_id.values()):
        configs = [c['config_id'] for c in selected['configurations']]
        matrix = {key: [by_id[f'b{block:02d}-{key}']['validation_accuracy'] for block in range(3)] for key in configs}
        report['leave_one_block_out'] = [
            {'held_out_block': block + 1,
             'spearman_two_block_mean_vs_held_out': spearman(
                 [statistics.mean(matrix[key][j] for j in range(3) if j != block) for key in configs],
                 [matrix[key][block] for key in configs])}
            for block in range(3)]
        baseline = [by_id[f'b{block:02d}-pool']['validation_accuracy'] for block in range(3)]
        if any(matrix[key][block] < baseline[block] for key in configs for block in range(3)):
            raise CalibrationError('continuation_lost_initial_incumbent')
        report['gain_summaries'] = [
            {'config_id': key, 'mean_gain_pp': 100 * statistics.mean(matrix[key][b] - baseline[b] for b in range(3)),
             'positive_gain_blocks': sum(matrix[key][b] > baseline[b] for b in range(3))}
            for key in configs]
    return report
