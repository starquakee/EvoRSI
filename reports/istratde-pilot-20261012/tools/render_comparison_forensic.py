"""Plot reconciled recorded costs; never plot unverified final claims as accepted."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '.runtime/comparison-reports/forensic'
x = json.loads((OUT / 'forensic-summary.json').read_text())
assert x['canonical_audit_verdict'] == 'FAIL' and not x['complete_matrix']
PLOTS = OUT / 'plots'
PLOTS.mkdir(exist_ok=True)
tasks = ['banknote', 'wdbc', 'spambase']
seeds = [11, 22, 33]
methods = ['FIXED', 'RANDOM', 'ISTRATDE']
colors = {'FIXED': '#636363', 'RANDOM': '#2563a6', 'ISTRATDE': '#c15827'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
for field, xlabel, filename in [
    ('cumulative_requests', 'Recorded model requests', 'validation-vs-requests.png'),
    ('cumulative_charged_tokens', 'Recorded tokens (thousands)', 'validation-vs-tokens.png')]:
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for i, seed in enumerate(seeds):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            task_values = [v['validation_incumbent'] for v in x['curves'] if v['task_id'] == task]
            for method in methods:
                points = [p for p in x['curves'] if p['task_id'] == task and p['search_seed'] == seed and p['method'] == method]
                divisor = 1000 if field.endswith('tokens') else 1
                ax.step([p[field]/divisor for p in points], [p['validation_incumbent'] for p in points],
                        where='post', color=colors[method], label=method, linewidth=1.6)
                ax.scatter([p[field]/divisor for p in points], [p['validation_incumbent'] for p in points],
                           color=colors[method], s=13, alpha=.7)
            ax.set_title(f'{task.upper()} / seed {seed}', loc='left', fontsize=11)
            ax.set_ylim(max(0, min(task_values)-.006), min(1.004, max(task_values)+.004))
            ax.grid(alpha=.18)
            if i == 2: ax.set_xlabel(xlabel)
            if j == 0: ax.set_ylabel('Validation incumbent accuracy')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5,.956), frameon=False)
    fig.suptitle('Validation vs actual recorded cost — forensic reconstruction', fontsize=15, y=.99)
    fig.text(.5,.016,'All 216 searches completed. Original ledger audit: FAIL (timestamp defect). These are validation scores, not final-test results.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.925))
    fig.savefig(PLOTS / filename, dpi=160)
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), sharey=True)
names = [f'{task} / {seed}' for task in tasks for seed in seeds]
for ax, baseline in zip(axes, ['RANDOM', 'FIXED']):
    ax.set_title(f'ISTRATDE - {baseline}', loc='left', fontsize=13)
    ax.set_xlim(-.05,.05)
    ax.set_ylim(8.8,-.8)
    ax.set_yticks(range(9), names)
    ax.axvline(0, color='#aaaaaa', linewidth=1)
    for y in range(9):
        ax.text(0,y,'N/A',ha='center',va='center',color='#6d6d6d',
                bbox={'facecolor':'white','edgecolor':'none','pad':4})
    ax.set_xlabel('Accepted paired final utility difference')
    ax.grid(axis='y',alpha=.15)
fig.suptitle('Predeclared paired comparisons are not estimable',fontsize=15,y=.98)
fig.text(.5,.065,'0/27 finals accepted by the strict audit; missing/unverified results are not zero-filled.',ha='center',fontsize=10)
fig.text(.5,.029,'Equal-task macroaverage and 10,000-resample paired bootstrap: N/A. Nine saved Banknote CSVs are diagnostic only.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.11,1,.93))
fig.savefig(PLOTS / 'paired-comparison-unavailable.png',dpi=160)
plt.close(fig)
print(json.dumps({'plots':[str(p) for p in PLOTS.glob('*.png')]}))
