"""Read-only aggregate export AFTER the frozen shared-start supervisor has ended.

No model requests, sandbox calls, ledger classes, or final-answer reads. Raw
programs/messages/predictions/accounting records remain private. Output must be
a fresh directory outside the experiment runtime. Use the frozen source checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

SOURCE_COMMIT = 'b1e3245796122845286a33c41172d9a92b987f6d'


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def digest(path):
    require(path.is_file() and not path.is_symlink(), 'missing_or_symlink_artifact')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    digest(path)
    return json.loads(path.read_text(encoding='utf-8'))


def json_lines(path):
    digest(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def tree_hashes(directory):
    require(directory.is_dir() and not directory.is_symlink(), 'runtime_directory_invalid')
    paths = sorted(directory.rglob('*'))
    require(not any(path.is_symlink() for path in paths), 'runtime_symlink')
    return {path.relative_to(directory).as_posix(): digest(path) for path in paths if path.is_file()}


def independent_matrix_check(analysis, complete):
    """Independent LS projections and SciPy ranks, not frozen mean-square helpers."""
    if not complete:
        return {'complete_matrix': False, 'reason': 'Missing/failed units are not dropped or imputed.'}
    import numpy as np
    from scipy.stats import spearmanr
    values = np.asarray([row['repeat_values'] for row in analysis['configuration_summaries']], dtype=float)
    require(values.shape == (9, 3) and np.isfinite(values).all(), 'full_matrix_shape_invalid')
    y = values.reshape(-1)
    block = np.tile(np.eye(3)[:, 1:], (9, 1))
    configuration = np.repeat(np.eye(9)[:, 1:], 3, axis=0)
    x0 = np.column_stack([np.ones(27), block])
    x1 = np.column_stack([x0, configuration])
    rss0 = float(np.sum((y - x0 @ np.linalg.lstsq(x0, y, rcond=None)[0]) ** 2))
    rss1 = float(np.sum((y - x1 @ np.linalg.lstsq(x1, y, rcond=None)[0]) ** 2))
    mse, msc = rss1 / 16, max(0., rss0 - rss1) / 8
    variance = max(0., (msc - mse) / 3)
    fraction = variance / (variance + mse) if variance + mse > 1e-25 else None
    correlations = [None if len(set(values[:, a])) == 1 or len(set(values[:, b])) == 1
                    else float(spearmanr(values[:, a], values[:, b]).statistic)
                    for a, b in ((0, 1), (0, 2), (1, 2))]
    rank = statistics.median(correlations) if all(v is not None for v in correlations) else None
    for field, value in (('ms_configuration', msc), ('ms_residual', mse),
                         ('estimated_configuration_variance', variance),
                         ('estimated_configuration_variance_fraction', fraction)):
        expected = analysis['variance'][field]
        require((value is None and expected is None) or (value is not None and expected is not None
                and abs(value - expected) < 1e-10), 'independent_variance_mismatch:' + field)
    for actual, expected in zip(correlations, analysis['rank_agreement']['pairwise_spearman']):
        require((actual is None and expected is None) or (actual is not None and expected is not None
                and abs(actual - expected) < 1e-12), 'independent_rank_mismatch')
    expected_rank = analysis['rank_agreement']['median_spearman']
    require((rank is None and expected_rank is None) or (rank is not None and expected_rank is not None
            and abs(rank - expected_rank) < 1e-12), 'independent_median_rank_mismatch')
    stable = fraction is not None and rank is not None and fraction >= .5 and rank >= .5
    return {'complete_matrix': True, 'least_squares_ms_configuration': msc,
            'least_squares_ms_residual': mse, 'configuration_variance_fraction': fraction,
            'scipy_pairwise_spearman': correlations, 'median_spearman': rank,
            'both_prespecified_thresholds_met': stable, 'matches_frozen_analyzer': True}


def _public_historical(report, report_sha):
    analysis = report['analysis']
    return {'report_sha256': report_sha, 'source_commit': report['source_commit'],
            'design': '27 independent searches, each with 2 fresh drafts + 4 subsequent main calls',
            'main_calls': sum(report['operator_counts'][k] for k in ('draft', 'improve', 'crossover')),
            'cost': report['cost'], 'verdict': analysis['verdict'],
            'configuration_variance_fraction': analysis['variance']['estimated_configuration_variance_fraction'],
            'median_spearman': analysis['rank_agreement']['median_spearman'],
            'pairwise_spearman': analysis['rank_agreement']['pairwise_spearman']}


def generate(root, output):
    """Return (public_report, all_30_unit_rows), writing aggregate JSON/CSVs only."""
    root, output = Path(root).resolve(), Path(output).resolve()
    # Disable bytecode before loading frozen modules; evidence reads cause no writes.
    sys.dont_write_bytecode = True
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from research.shared_start.analysis import analyze
    from research.shared_start.collect import validate_trial
    from research.shared_start.prepare import historical_integrity
    from research.shared_start.specs import LEDGER, RUNTIME, selected_plan, trials
    runtime = root / RUNTIME
    require(output != runtime and runtime not in output.parents, 'output_inside_private_runtime')
    require(not output.exists(), 'output_must_be_fresh')
    # outcome.json is written exclusively in the supervisor's final block.
    outcome = read_json(runtime / 'outcome.json')
    require(outcome['status'] in ('completed', 'incomplete') and
            type(outcome.get('finished_at')) in (int, float) and
            outcome['started_at'] <= outcome['finished_at'] <= time.time(), 'supervisor_not_ended')
    before = tree_hashes(runtime)
    manifest = read_json(runtime / 'manifest.json')
    require(outcome['source_commit'] == manifest['source_commit'] == SOURCE_COMMIT, 'frozen_source_mismatch')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    require(head == SOURCE_COMMIT, 'use_frozen_source_checkout')
    require(subprocess.run(['git', 'diff', '--quiet', SOURCE_COMMIT, '--', 'research', 'OpenMLE-Evo'],
                           cwd=root, capture_output=True).returncode == 0, 'frozen_source_modified')
    # Never instantiate/open the ledger class: its context manager can persist state.
    state, snapshot = read_json(root / LEDGER), outcome['budget']
    require(outcome['manifest_identity'] == manifest['identity'] == state['manifest_identity'], 'manifest_binding_mismatch')
    for key in ('requests', 'started_at', 'stopped', 'seed_pools'):
        require(snapshot[key] == state[key], 'ledger_snapshot_mismatch:' + key)
    requests = state['requests']
    require(len({r['request_id'] for r in requests}) == len(requests), 'duplicate_request_id')
    for record in requests:
        require(record['status'] in ('known', 'unknown', 'reserved'), 'request_usage_status_invalid')
        expected = sum(record['usage'].values()) if record['status'] == 'known' else record['reserved_tokens']
        require(record['charged_tokens'] == expected, 'request_usage_charge_mismatch')
    cost = {'requests': len(requests), 'tokens': sum(r['charged_tokens'] for r in requests)}
    require(snapshot['global']['committed'] == cost, 'global_cost_mismatch')
    require(cost['requests'] <= 174 and cost['tokens'] <= 600000, 'approved_budget_exceeded')
    plan, roster = selected_plan(), trials(root, SOURCE_COMMIT)
    by_identity = {trial.identity(): trial for trial in roster}
    require(all(r['run_identity'] in by_identity for r in requests), 'foreign_request')
    for trial in roster:
        stored_cell = state['trials'][trial.identity()]
        snapshot_cell = snapshot['cells'][trial.trial_id]
        require(all(snapshot_cell.get(key) == value for key, value in stored_cell.items()), 'trial_snapshot_state_mismatch')
        own = [r for r in requests if r['run_identity'] == trial.identity()]
        require(snapshot_cell['committed'] == {'requests': len(own), 'tokens': sum(r['charged_tokens'] for r in own)},
                'trial_snapshot_cost_mismatch')
    normalized = {row['trial_id']: row for row in outcome['outcomes']}
    require(len(normalized) == len(outcome['outcomes']), 'duplicate_outcome')
    analysis = analyze(outcome['outcomes'])
    integrity_ok = (outcome['historical_artifacts_unchanged'] and outcome['stack_idle'] and
                    (outcome['emergency_cleanup'] is None or outcome['emergency_cleanup'].get('cleanup_verified') is True))
    if not integrity_ok:
        analysis['verdict'] = 'integrity_or_cleanup_unproven'
    elif outcome['status'] != 'completed':
        analysis['verdict'] = 'insufficient_complete_evidence'
    require(analysis == outcome['analysis'], 'frozen_analysis_or_verdict_mismatch')
    require(analysis['recorded_cost'] == {'requests': cost['requests'], 'charged_tokens': cost['tokens']}, 'analysis_cost_mismatch')
    saved = {entry['normalized']['trial_id']: entry for entry in outcome['verified_units']}
    require(len(saved) == len(outcome['verified_units']), 'duplicate_verified_unit')
    require(set(saved) <= {t.trial_id for t in roster}, 'foreign_verified_unit')
    verified, jobs_seen, import_audits = {}, set(), []
    for trial in roster:
        if trial.trial_id not in saved:
            continue
        current = validate_trial(root, manifest, trial, snapshot)
        require(current == saved[trial.trial_id] == read_json(runtime / 'verified' / (trial.trial_id + '.json')),
                'independent_unit_revalidation_mismatch:' + trial.trial_id)
        require(current['normalized'] == normalized[trial.trial_id], 'normalized_verified_mismatch')
        require(not jobs_seen.intersection(current['rescored_jobs']), 'new_scored_job_duplicate_across_units')
        jobs_seen.update(current['rescored_jobs'])
        verified[trial.trial_id] = current
        if trial.kind == 'continuation':
            bundle = read_json(runtime / 'runs' / trial.pool_id / 'seed-bundle.json')
            observed = current['behavior'].get('initial_parent_candidate_ids')
            has_main = sum(current['behavior']['operator_counts'][name] for name in ('draft', 'improve', 'crossover')) > 0
            both_observed = set(observed or []) == set(bundle['terminal_node_ids']) and len(observed or []) == 2
            require(both_observed if has_main else observed is None, 'initial_population_not_shared_two_seeds')
            import_audits.append({'trial_id': trial.trial_id, 'pool_id': trial.pool_id,
                'bundle_sha256': snapshot['seed_pools'][trial.pool_id]['bundle_sha256'],
                'observed_initial_candidate_count': len(observed or []), 'both_terminal_seeds_observed': both_observed})
    preflight = read_json(runtime / 'preflight.json')
    history = historical_integrity(root)
    require(history == preflight['historical'], 'historical_artifacts_changed')
    require(history['original_comparison_files_verified'] == 6413, 'original_comparison_inventory_mismatch')
    require(history['completed_calibration']['file_count'] == 831, 'old_calibration_inventory_mismatch')
    all_complete = (len(verified) == 30 and all(entry['normalized']['status'] == 'completed' and
                   entry['normalized']['validation_accuracy'] is not None for entry in verified.values()))
    require(outcome['status'] != 'completed' or (all_complete and integrity_ok and state['stopped'] is None), 'false_complete_status')
    if all_complete:
        require(all(r['status'] == 'known' for r in requests), 'complete_with_unsettled_usage')
        require(len(import_audits) == 27 and len(snapshot['seed_pools']) == 3 and
                all(item['both_terminal_seeds_observed'] for item in import_audits), 'shared_pool_replication_shape_invalid')
    independent = independent_matrix_check(analysis, all_complete)
    if all_complete and integrity_ok and outcome['status'] == 'completed':
        expected_verdict = ('exploratory_stable_configuration_signal' if independent['both_prespecified_thresholds_met']
                            else 'stable_configuration_signal_not_demonstrated')
        require(analysis['verdict'] == expected_verdict, 'prespecified_threshold_verdict_mismatch')
    units, behavior_rows, counts, job_counts = [], [], Counter(), Counter()
    new_node_counts = Counter()
    for trial in roster:
        own = [r for r in requests if r['run_identity'] == trial.identity()]
        data = normalized.get(trial.trial_id)
        if data is not None:
            require(data['requests'] == len(own) and data['charged_tokens'] == sum(r['charged_tokens'] for r in own), 'unit_cost_mismatch')
        else:
            require(not own, 'missing_outcome_has_charged_cost')
            data = {'status': 'not_run', 'validation_accuracy': None, 'requests': 0,
                    'charged_tokens': 0, 'reason': 'not_run'}
        baseline = normalized.get(trial.pool_id, {}).get('validation_accuracy')
        gain = (100 * (data['validation_accuracy'] - baseline) if trial.kind == 'continuation'
                and data['status'] == 'completed' and data['validation_accuracy'] is not None and baseline is not None else None)
        units.append({'trial_id': trial.trial_id, 'kind': trial.kind, 'pool_id': trial.pool_id,
            'block': trial.record['repeat'] + 1, 'position': trial.record['position'] + 1,
            'config_id': trial.record['config_id'], 'inner_seed': trial.inner_seed, **trial.decoded,
            **{k: data[k] for k in ('status', 'validation_accuracy', 'requests', 'charged_tokens', 'reason')},
            'main_requests': sum(r['kind'] == 'main' for r in own),
            'debug_requests': sum(r['kind'] == 'debug' for r in own),
            'independently_verified': trial.trial_id in verified,
            'pool_baseline_accuracy': baseline, 'gain_from_pool_pp': gain})
        if trial.trial_id not in verified:
            continue
        entry = verified[trial.trial_id]
        behavior, selections = entry['behavior'], entry['behavior']['parent_selections']
        counts.update(behavior['operator_counts'])
        job_counts.update(entry['job_counts'])
        directory = runtime / 'runs' / trial.trial_id
        nodes = json_lines(directory / 'journal.jsonl')
        trace = json_lines(directory / 'operator-trace.jsonl')
        new_ids = {event['child_node_id'] for event in trace}
        require(len(new_ids) == len(trace), 'new_node_trace_duplicate')
        new_nodes = [node for node in nodes if node['id'] in new_ids]
        require(len(new_nodes) == len(new_ids), 'new_node_missing')
        invalid_python = sum('empty_candidate_code:invalid_python' in
            str(node.get('_term_out', node.get('term_out', ''))) for node in new_nodes)
        buggy = sum(node.get('is_buggy') is True for node in new_nodes)
        empty = sum(not node.get('code') for node in new_nodes)
        new_node_counts.update({'total_new_nodes': len(new_nodes), 'buggy_new_nodes': buggy,
                               'empty_code_new_nodes': empty, 'invalid_python_new_nodes': invalid_python})
        behavior_rows.append({'trial_id': trial.trial_id, 'kind': trial.kind, 'pool_id': trial.pool_id,
            'config_id': trial.record['config_id'], **behavior['operator_counts'],
            'seed_incumbent_retained': entry['seed_incumbent_retained'],
            'new_node_count': len(new_nodes), 'buggy_new_nodes': buggy, 'invalid_python_new_nodes': invalid_python,
            'initial_shared_seed_count': len(behavior.get('initial_parent_candidate_ids') or []),
            'parent_events': len(selections),
            'score_informative_parent_events': sum(s['score_component_informative'] for s in selections),
            'mean_score_weight_first_draw_tv_0_to_2': statistics.mean(s['score_weight_first_draw_tv_0_to_2'] for s in selections) if selections else None,
            'result_sha256': entry['result_sha256']})
    pool_rows = [row for row in units if row['kind'] == 'fresh_seed_pool']
    cont_rows = [row for row in units if row['kind'] == 'continuation']
    by_kind_cost = {kind: {'requests': sum(row['requests'] for row in units if row['kind'] == kind),
                         'tokens': sum(row['charged_tokens'] for row in units if row['kind'] == kind)}
                    for kind in ('fresh_seed_pool', 'continuation')}
    require(sum(row['requests'] for row in units) == cost['requests'] and
            sum(row['charged_tokens'] for row in units) == cost['tokens'], 'cost_counted_more_than_once')
    old_path = root / 'reports/calibration-round2-results-20261013/report.json'
    historical = _public_historical(read_json(old_path), digest(old_path))
    require(historical['main_calls'] == 162, 'historical_main_count_mismatch')
    cont_behavior = [row for row in behavior_rows if row['kind'] == 'continuation']
    result = {'schema': 'shared-start-public-report.v1', 'status': outcome['status'],
        'source_commit': SOURCE_COMMIT, 'manifest_identity': manifest['identity'],
        'started_at': outcome['started_at'], 'finished_at': outcome['finished_at'],
        'elapsed_seconds': outcome['elapsed_seconds'], 'stopped_reason': outcome['stopped_reason'],
        'cost': cost, 'seed_pool_cost': by_kind_cost['fresh_seed_pool'], 'continuation_cost': by_kind_cost['continuation'],
        'main_requests': sum(r['kind'] == 'main' for r in requests),
        'usage_status_counts': dict(Counter(r['status'] for r in requests)),
        'analysis': analysis, 'independent_recomputation': independent,
        'verified_units': len(verified), 'completed_pools': sum(row['status'] == 'completed' for row in pool_rows),
        'completed_continuations': sum(row['status'] == 'completed' for row in cont_rows),
        'unique_new_independently_rescored_jobs': len(jobs_seen), 'import_reaudits_not_new_jobs': len(import_audits),
        'shared_pool_import_audits': import_audits,
        'seed_incumbent_retained_continuations': sum(row['seed_incumbent_retained'] for row in cont_behavior),
        'positive_gain_continuations': sum(row['gain_from_pool_pp'] is not None and row['gain_from_pool_pp'] > 0 for row in cont_rows),
        'continuation_fallback_draft_calls': sum(row['draft'] for row in cont_behavior),
        'verified_operator_counts': dict(counts), 'verified_job_counts': dict(job_counts),
        'verified_new_node_counts': dict(new_node_counts),
        'mechanism_statistics_scope': 'Independently verified units only; all charged costs include unverified/failed units.',
        'historical_descriptive_comparison': historical,
        'original_comparison_files_unchanged': 6413, 'old_calibration_files_unchanged': 831,
        'historic_ledgers_and_final_stage_unchanged': True, 'cleanup_verified': outcome['stack_idle'],
        'historical_integrity_confirmed': True, 'fixed_preflight_jobs': outcome['preflight_jobs'],
        'new_final_evaluations': outcome['final_evaluations'],
        'private_outcome_sha256': before['outcome.json'], 'private_ledger_sha256': before['budget-ledger.json'],
        'private_runtime_file_count': len(before)}
    require(result['new_final_evaluations'] == 0, 'unexpected_final_evaluation')
    require(tree_hashes(runtime) == before, 'runtime_changed_during_read_only_report')
    result['runtime_hashes_unchanged_during_collection'] = True
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in (('report.json', result), ('protocol.json', manifest['protocol']), ('schedule.json', plan)):
        (output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, records in (('units-30.csv', units), ('continuations-27.csv', cont_rows),
                          ('pools-3.csv', pool_rows), ('operator-behavior.csv', behavior_rows)):
        with (output / name).open('w', newline='', encoding='utf-8') as handle:
            fields = list(records[0]) if records else ['trial_id', 'kind', 'status']
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(records)
    return result, units


def _value(value, digits=4):
    return f'{value:.{digits}f}' if value is not None else '无法估计'


def render(output, result, units):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries = result['analysis']['configuration_summaries']
    colors = ['#24699A', '#CB7E2D', '#458C75']
    cont = [row for row in units if row['kind'] == 'continuation']
    configs = {row['config_id']: row for row in cont}
    pools = {row['block']: row for row in units if row['kind'] == 'fresh_seed_pool'}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 9), sharex=True)
    for block in range(3):
        for index, summary in enumerate(summaries):
            row = next(r for r in cont if r['config_id'] == summary['config_id'] and r['block'] == block + 1)
            if row['status'] == 'completed' and row['validation_accuracy'] is not None:
                axes[0].scatter(index + (block-1)*.13, row['validation_accuracy']*100,
                                color=colors[block], s=38, zorder=3,
                                label=f'Block {block+1}' if index == 0 else None)
                axes[1].scatter(index + (block-1)*.13, row['gain_from_pool_pp'], color=colors[block], s=38, zorder=3)
        baseline = pools[block+1]['validation_accuracy']
        if baseline is not None:
            axes[0].axhline(baseline*100, color=colors[block], linestyle='--', linewidth=.8, alpha=.75,
                            label=f'Pool {block+1} best')
    for index, summary in enumerate(summaries):
        values = [v for v in summary['repeat_values'] if v is not None]
        if len(values) == 3:
            axes[0].errorbar(index, statistics.mean(values)*100, yerr=statistics.stdev(values)*100,
                fmt='D', color='#202020', markersize=4, capsize=4, linewidth=1.1, zorder=4)
            gains = [r['gain_from_pool_pp'] for r in cont if r['config_id'] == summary['config_id']]
            if all(value is not None for value in gains):
                axes[1].errorbar(index, statistics.mean(gains), yerr=statistics.stdev(gains), fmt='D',
                    color='#202020', markersize=4, capsize=4, linewidth=1.1, zorder=4)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
    axes[0].set_ylabel('Best trusted validation accuracy (%)')
    axes[1].set_ylabel('Gain over shared pool best (percentage points)')
    axes[1].axhline(0, color='#555555', linewidth=.8)
    axes[0].set_title('Shared-start Spambase: nine configurations conditional on three pools', loc='left', pad=12)
    axes[0].legend(ncol=3, frameon=False, fontsize=9)
    axes[1].set_xticks(range(9), [f"{s['config_id']}\np={configs[s['config_id']]['crossover_prob']:g}, w={configs[s['config_id']]['score_weight']:g}" for s in summaries], fontsize=9)
    fig.text(.08, .015, 'Black diamonds: mean ± 1 block SD (only all 3 blocks present), not confidence intervals. Missing observations remain missing.', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output / 'shared-start-blocks-and-gains.png', dpi=200)
    plt.close(fig)
    old = result['historical_descriptive_comparison']
    current = result['analysis']
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    metrics = [('Estimated configuration variance fraction', old['configuration_variance_fraction'],
                current['variance']['estimated_configuration_variance_fraction'] if current['variance'] else None),
               ('Median pairwise Spearman correlation', old['median_spearman'],
                current['rank_agreement']['median_spearman'] if current['rank_agreement'] else None)]
    for ax, (title, previous, now) in zip(axes, metrics):
        for index, value in enumerate((previous, now)):
            if value is None:
                ax.text(index, .05, 'Not estimable', ha='center', fontsize=9)
            else:
                ax.scatter(index, value, s=65, color=('#777777', '#24699A')[index], zorder=3)
                ax.annotate(f'{value:.3f}', (index, value), xytext=(0, 10), textcoords='offset points', ha='center')
        ax.axhline(.5, linestyle='--', color='#777777', linewidth=.8)
        ax.set_xlim(-.5, 1.5)
        ax.set_ylim(-1 if 'Spearman' in title else -.05, 1.1)
        ax.set_xticks([0, 1], ['Historical independent\n162 main calls', f"Shared-start\n{result['main_requests']} main calls"])
        ax.set_title(title, loc='left', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
    fig.suptitle('Descriptive historical comparison — different starting design and budget', x=.08, ha='left', fontsize=12)
    fig.text(.08, .015, 'Shared-start target: 6 pool + 108 continuation main calls = 114. No randomized causal or superiority claim.', fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .93))
    fig.savefig(output / 'historical-descriptive-comparison.png', dpi=200)
    plt.close(fig)
    write_readme(output, result, units)


def write_readme(output, result, units):
    analysis = result['analysis']
    verdicts = {'exploratory_stable_configuration_signal': '本轮达到预设的探索性稳定性门槛。',
        'stable_configuration_signal_not_demonstrated': '本轮没有达到预设的稳定性门槛。',
        'insufficient_complete_evidence': '实验未完整结束，不能按预设规则判断配置稳定性。',
        'integrity_or_cleanup_unproven': '完整性或清理证据尚未确认，不能给出稳定性结论。'}
    variance = analysis['variance']['estimated_configuration_variance_fraction'] if analysis['variance'] else None
    ranks = analysis['rank_agreement']
    rank = ranks['median_spearman'] if ranks else None
    params = {row['config_id']: row for row in units if row['kind'] == 'continuation'}
    table = []
    for summary in analysis['configuration_summaries']:
        cfg = params[summary['config_id']]
        values = summary['repeat_values']
        mean = f"{summary['mean']*100:.3f}%" if summary['mean'] is not None else '缺失'
        sd = f"{summary['sample_sd']*100:.3f}" if summary['sample_sd'] is not None else '—'
        table.append(f"| {summary['config_id']} | {cfg['crossover_prob']:g} | {cfg['score_weight']:g} | {summary['completed_observations']}/3 | {mean} | {sd} |")
    old = result['historical_descriptive_comparison']
    usage = result['usage_status_counts']
    averaging = analysis.get('leave_one_block_out') or []
    missing = [row['trial_id'] + ':' + row['status'] for row in units if row['status'] != 'completed']
    cost, pc, cc = result['cost'], result['seed_pool_cost'], result['continuation_cost']
    new = result['verified_new_node_counts']
    pool_values = [row['validation_accuracy'] for row in units if row['kind'] == 'fresh_seed_pool']
    pool_text = '、'.join(f'{value*100:.4f}%' if value is not None else '缺失' for value in pool_values)
    zero_note = ('本轮配置均方小于剩余均方，冻结估计规则将配置方差截断为 0。'
                 '这是三个批次的样本估计，不能据此证明真实配置效应为零。' if variance == 0 else '')
    text = [
        '# Spambase 共享初始候选池实验', '', '**' + verdicts[analysis['verdict']] + '**', '',
        f"已独立核验 {result['verified_units']}/30 个单元；完成 {result['completed_pools']}/3 个初始候选池、{result['completed_continuations']}/27 次续跑。全轮费用为 {cost['requests']} 次请求、{cost['tokens']:,} tokens，耗时 {result['elapsed_seconds']/60:.2f} 分钟。上限为 174 次请求、600,000 tokens、4 小时。", '',
        f"初始池费用共 {pc['requests']} 次请求、{pc['tokens']:,} tokens，仅记一次；27 次续跑合计 {cc['requests']} 次请求、{cc['tokens']:,} tokens。失败、调试和未通过验证单元的已记账费用都包含在总额中。用量状态：已知 {usage.get('known', 0)}，未知 {usage.get('unknown', 0)}，尚未结算 {usage.get('reserved', 0)}。后两类按完整预留额度计费，并非已确认实际消耗。", '',
        '每个批次先生成两份有效终端候选，必要时在预算内调试；同批九个配置导入同一份已封存候选池。每次续跑只新增四次主请求，最多两次调试。首次父代选择必须实际包含这两个共享候选，不能仅在日志中保留它们。后续原生错误恢复可以生成新 draft，仍占用四次主请求额度。', '',
        '## 稳定性判断', '',
        f'- 去除批次整体偏移后的估计配置方差占比：**{_value(variance)}**；门槛 ≥ 0.5。',
        f'- 批次间配置排名 Spearman 相关中位数：**{_value(rank)}**；门槛 ≥ 0.5。',
        '- 只有三个池及全部 27 次续跑完整且证据有效，并同时达到两个门槛，才记为观察到探索性稳定信号。缺失值不填补，不删除失败单元，不追加试验。', '',
        ('三个批次的两两排名相关：' + '、'.join(_value(v) for v in ranks['pairwise_spearman']) + '。') if ranks else '矩阵不完整，未计算完整矩阵排名相关。',
        ('两个批次取平均后，与留出批次的排名相关依次为：' + '、'.join(_value(row['spearman_two_block_mean_vs_held_out']) for row in averaging) + '。这里留出的是重复批次，所有评分仍使用同一验证集。') if averaging else '矩阵不完整，未计算留一批次的均值排名诊断。',
        ('完整矩阵已另用最小二乘投影复算方差分解、SciPy 复算排名相关，与冻结分析器一致。' if result['independent_recomputation']['complete_matrix'] else '完整矩阵的独立方差／排名复算未执行；保留已观察值和全部费用。'), '',
        zero_note, '',
        '## 九个配置与共同起点', '',
        '| 配置 | 交叉概率 | 成绩权重 | 完成批次 | 平均最佳验证准确率 | 批次标准差（百分点） |',
        '|---|---:|---:|---:|---:|---:|', *table, '',
        '![批次表现、共享池基线与增益](shared-start-blocks-and-gains.png)', '',
        '三个共享池的最佳验证准确率依次为：' + pool_text + '。相同基线的虚线会重叠。', '',
        '彩色点是各批次观察值，虚线为该批共享池最佳分数；黑色标记及误差线仅在三个批次齐全时绘制，表示均值±批次标准差，不是置信区间。下图显示相对各自共享池的增益。表中的未完整配置均值只描述现有观察，不用于完整矩阵判定。', '',
        f"有 {result['positive_gain_continuations']}/27 次续跑超过共享池最佳分数；{result['seed_incumbent_retained_continuations']} 次最终仍选择导入候选。已核验续跑中共有 {result['continuation_fallback_draft_calls']} 次新增 fallback draft。", '',
        f"按操作轨迹中的新节点 ID 去重，共核验 {new.get('total_new_nodes', 0)} 个新节点，其中 {new.get('buggy_new_nodes', 0)} 个标记为失败、{new.get('invalid_python_new_nodes', 0)} 个因无效 Python 被拒绝。导入节点未重复计入。行为及失败节点统计仅覆盖独立核验单元；费用覆盖所有单元。", '',
        '## 与上一轮的描述性比较', '',
        f"上一轮独立起点校准的配置方差占比为 {_value(old['configuration_variance_fraction'])}，排名相关中位数为 {_value(old['median_spearman'])}；本轮分别为 {_value(variance)}、{_value(rank)}。", '',
        '![与历史校准的描述性比较](historical-descriptive-comparison.png)', '',
        '两轮的估计对象不同：本轮比较的是给定同一初始池后的条件续跑表现，三个池才是独立的起点；上一轮每次搜索各自产生初始草稿。本轮完整设计只需 114 次主请求（6 次建池＋108 次续跑），上一轮为 162 次。起点、总预算和执行时段同时不同，因此这些数值不能证明共享起点导致稳定性改善，也不能证明某配置或外层算法更优。三批次规模较小，结果还条件于固定数据拆分、模型服务和短搜索预算。', '',
        '## 证据与边界', '',
        f"重新核对了全部已收录单元的账本、来源代码、预测快照、验证答案哈希和操作选择概率。独立复算 {result['unique_new_independently_rescored_jobs']} 份唯一新作业预测；另外 {result['import_reaudits_not_new_jobs']} 次共享池导入审计不计为新作业或新费用。", '',
        f"原比较实验 6,413 个文件、上一轮校准 831 个文件、历史账本及最终评测阶段均与启动前哈希一致。当前运行目录在报告读取前后哈希不变。结束时栈空闲证据：{result['cleanup_verified']}。另有 {result['fixed_preflight_jobs']} 次固定隔离探针，模型请求为零，不是测量样本。新增最终测试评测为 0；没有运行外层 iStratDE。", '',
        ('缺失／失败单元：' + '，'.join(missing) + '。停止原因：' + str(result['stopped_reason']) + '。') if missing else '全部 30 个单元均完成，没有删除或填补观察。', '',
        '- [全部 30 单元与成本](units-30.csv)', '- [27 次续跑](continuations-27.csv)、[3 个初始池](pools-3.csv)',
        '- [操作行为](operator-behavior.csv)、[机器报告](report.json)', '- [冻结协议](protocol.json)、[冻结顺序](schedule.json)', '',
        '原始候选程序、模型消息、预测行、账本及授权凭证均保留在私有运行目录。公开报告只含聚合统计、配置及证据哈希。冻结计划中的草案状态文字不代表新的执行授权。', '',
        f"执行源码：`{SOURCE_COMMIT}`；实验清单：`{result['manifest_identity']}`。", '',
    ]
    (output / 'README.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result, rows = generate(args.repo_root, args.output)
    render(args.output, result, rows)
    print(json.dumps({'status': result['status'], 'verdict': result['analysis']['verdict'],
                      'cost': result['cost'], 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
