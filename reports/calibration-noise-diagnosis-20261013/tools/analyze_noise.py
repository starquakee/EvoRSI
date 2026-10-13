"""Offline descriptive analysis of public aggregates; no live-run imports or I/O.

Requires numpy, scipy and matplotlib. See the adjacent report for limitations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def residual(matrix: np.ndarray) -> np.ndarray:
    """Remove configuration and repetition additive means from a balanced matrix."""
    return matrix - matrix.mean(axis=0) - matrix.mean(axis=1)[:, None] + matrix.mean()


def rho(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def analyze(initial: np.ndarray, final: np.ndarray) -> dict:
    if initial.shape != (9, 3) or final.shape != initial.shape:
        raise ValueError('Expected exactly nine configurations and three repetitions')
    if not np.isfinite(initial).all() or not np.isfinite(final).all():
        raise ValueError('Scores must be finite')
    if (initial < 0).any() or (final > 1).any() or (final < initial).any():
        raise ValueError('Invalid best-so-far validation accuracy')
    gain = final - initial
    # 27 observations - (intercept + 8 configuration + 2 block contrasts) = 16.
    dof = 16
    ix, gx, fx = (residual(a) for a in (initial, gain, final))
    vi, vg, vf = (float(np.sum(a * a) / dof) for a in (ix, gx, fx))
    cov = float(np.sum(ix * gx) / dof)
    folds = []
    for held in range(3):
        train = np.delete(final, held, axis=1).mean(axis=1)
        test = final[:, held]
        selected = np.flatnonzero(train == train.max())
        folds.append({
            'held_out_repeat': held + 1,
            'spearman_two_repeat_mean_vs_held_out': rho(train, test),
            'selected_configs_including_ties': [f'c{i:02d}' for i in selected],
            'held_out_selected_mean_accuracy': float(test[selected].mean()),
            'held_out_gap_to_best_pp': float(100 * (test.max() - test[selected].mean())),
        })
    fold_rhos = [r['spearman_two_repeat_mean_vs_held_out'] for r in folds]
    pairs = [rho(final[:, i], final[:, j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    corr = None if np.ptp(initial) == 0 or np.ptp(final) == 0 else float(
        np.corrcoef(initial.ravel(), final.ravel())[0, 1])
    return {
        'schema': 'calibration-noise-diagnosis.v1',
        'status': 'retrospective_descriptive_only',
        'observations': 27, 'configurations': 9, 'repetitions': 3,
        'additional_model_requests': 0, 'additional_sandbox_jobs': 0,
        'additional_final_evaluations': 0,
        'scores': {
            'mean_initial_accuracy': float(initial.mean()),
            'mean_final_accuracy': float(final.mean()),
            'mean_gain_pp': float(100 * gain.mean()),
            'positive_gain_trials': int(np.sum(gain > 0)),
            'unchanged_trials': int(np.sum(gain == 0)),
            'initial_sample_sd_pp': float(100 * initial.std(ddof=1)),
            'final_sample_sd_pp': float(100 * final.std(ddof=1)),
            'initial_final_pearson': corr,
        },
        'additive_residual_decomposition': {
            'degrees_of_freedom': dof,
            'units': 'squared percentage points',
            'initial_variance_pp2': vi * 10000,
            'gain_variance_pp2': vg * 10000,
            'twice_initial_gain_covariance_pp2': 2 * cov * 10000,
            'final_variance_pp2': vf * 10000,
            'identity_abs_error_pp2': abs(vf - vi - vg - 2 * cov) * 10000,
            'initial_residual_rms_pp': 100 * vi**0.5,
            'gain_residual_rms_pp': 100 * vg**0.5,
            'final_residual_rms_pp': 100 * vf**0.5,
            'interpretation': 'Algebraic decomposition, not independent or causal noise shares. '
                'Gain contains minus initial score; covariance includes mathematical coupling. '
                'Residual combines stochastic variation, interaction and unmodeled effects.',
        },
        'ranking': {
            'single_repeat_pairwise_spearman': pairs,
            'two_repeat_mean_vs_held_out': folds,
            'median_held_out_spearman': None if None in fold_rhos else float(np.median(fold_rhos)),
            'interpretation': 'Only three overlapping folds over nine configurations. '
                'Held-out means repetition, not unseen data; every score uses the same validation split. '
                'No confidence interval, population reliability or algorithm advantage claim.',
        },
    }


def load(source: Path) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    manifest = json.loads((source / 'export-manifest.json').read_text())
    for name in ('trials-27.csv', 'operator-behavior.csv', 'report.json', 'schedule.json'):
        if sha(source / name) != manifest['files'][name]:
            raise ValueError(f'Input hash mismatch: {name}')
    with (source / 'trials-27.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    with (source / 'operator-behavior.csv').open(newline='') as f:
        behaviors = list(csv.DictReader(f))
    ops = {r['trial_id']: r for r in behaviors}
    ids = [r['trial_id'] for r in rows]
    if len(rows) != 27 or len(set(ids)) != 27 or len(behaviors) != 27 or set(ops) != set(ids):
        raise ValueError('Inputs must have exactly the same 27 unique trials')
    schedule = {r['trial_id']: r for r in json.loads((source / 'schedule.json').read_text())['schedule']}
    if set(schedule) != set(ids):
        raise ValueError('Trial IDs differ from frozen schedule')
    initial, final = np.full((9, 3), np.nan), np.full((9, 3), np.nan)
    joined = []
    for row in rows:
        tid = row['trial_id']
        op, planned = ops[tid], schedule[tid]
        i, j = int(row['config_id'][1:]), int(row['repeat']) - 1
        if not (0 <= i < 9 and 0 <= j < 3) or np.isfinite(final[i, j]):
            raise ValueError('Duplicate or invalid matrix cell')
        if (row['status'] != 'completed' or op['config_id'] != row['config_id']
                or planned['config_id'] != row['config_id'] or planned['repeat'] != j
                or int(op['draft']) != 2 or int(op['improve']) + int(op['crossover']) != 4
                or any(float(row[k]) != v for k, v in planned['decoded'].items())):
            raise ValueError('Incomplete trial or protocol mismatch')
        initial[i, j] = float(op['initial_draft_best'])
        final[i, j] = float(row['validation_accuracy'])
        joined.append({
            'trial_id': tid, 'config_id': row['config_id'], 'repeat': j + 1,
            'initial_accuracy': initial[i, j], 'final_accuracy': final[i, j],
            'gain_pp': 100 * (final[i, j] - initial[i, j]),
            'improve': int(op['improve']), 'crossover': int(op['crossover']),
            'debug': int(op['debug']),
        })
    prior = json.loads((source / 'report.json').read_text())
    if (sum(int(r['requests']) for r in rows) != prior['cost']['requests']
            or sum(int(r['charged_tokens']) for r in rows) != prior['cost']['tokens']):
        raise ValueError('Prior costs do not reconcile')
    return initial, final, joined, {
        'input_directory': source.name,
        'input_sha256': {name: sha(source / name) for name in (
            'trials-27.csv', 'operator-behavior.csv', 'report.json', 'schedule.json', 'export-manifest.json')},
        'original_execution_commit': prior['source_commit'],
        'original_cost': prior['cost'],
        'verification_scope': 'Published aggregate bytes and consistency; does not reauthenticate private predictions.',
    }


def plot(initial: np.ndarray, final: np.ndarray, result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
    colors = ['#206b89', '#ca713d', '#647748']
    for j in range(3):
        axes[0].scatter(initial[:, j] * 100, final[:, j] * 100, label=f'Repeat {j+1}',
                        color=colors[j], s=45, alpha=.8)
    axes[0].plot([93, 94.6], [93, 94.6], color='#999999', linestyle='--', linewidth=1)
    axes[0].set(xlabel='Initial best validation accuracy (%)', ylabel='Final best validation accuracy (%)',
                title='Starting score weakly tracks final score')
    axes[0].legend(frameon=False, fontsize=8)
    decomp = result['additive_residual_decomposition']
    values = [decomp[k] for k in ('initial_variance_pp2', 'gain_variance_pp2',
                                 'twice_initial_gain_covariance_pp2', 'final_variance_pp2')]
    axes[1].bar(['Initial', 'Gain', '2 × cov.', 'Final'], values,
                color=['#206b89', '#ca713d', '#aaaaaa', '#647748'])
    axes[1].axhline(0, color='#444444', linewidth=.7)
    for n, v in enumerate(values):
        axes[1].text(n, v + (.003 if v >= 0 else -.003), f'{v:.4f}',
                     ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    axes[1].set(ylabel='Residual variance / covariance (pp²)', ylim=(-.083, .13),
                title='Initial + gain + 2 × covariance = final')
    vals = [r['spearman_two_repeat_mean_vs_held_out'] for r in result['ranking']['two_repeat_mean_vs_held_out']]
    axes[2].bar(['Hold out 1', 'Hold out 2', 'Hold out 3'], vals, color=colors)
    axes[2].set(ylim=(-.05, 1), ylabel='Spearman rank correlation',
                title='Two-repeat means still rank weakly')
    for n, v in enumerate(vals):
        axes[2].text(n, v + .025, f'{v:.3f}', ha='center')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('27 existing Spambase trials · validation only · descriptive, not causal', fontsize=14)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    initial, final, joined, provenance = load(args.input)
    result = analyze(initial, final)
    result['provenance'] = provenance
    (args.output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (args.output / 'joined-trials.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(joined[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(joined)
    plot(initial, final, result, args.output / 'noise-diagnosis.png')
    print(json.dumps({'trials': 27, 'mean_gain_pp': result['scores']['mean_gain_pp'],
                      'held_out_rhos': [r['spearman_two_repeat_mean_vs_held_out']
                        for r in result['ranking']['two_repeat_mean_vs_held_out']],
                      'model_requests': 0, 'sandbox_jobs': 0}))


if __name__ == '__main__':
    main()
