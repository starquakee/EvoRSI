"""Render the completed amended matrix with explicit uncertainty and provenance."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[3]
out=root/'.runtime/final-only-amendment-report'
x=json.loads((out/'report.json').read_text())
assert x['matrix_complete'] and len(x['cells'])==27
plots=out/'plots';plots.mkdir(exist_ok=True)
colors={'FIXED':'#686868','RANDOM':'#2563a6','ISTRATDE':'#c15827'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(12.5,4.8),sharey=True)
for ax,task in zip(axes,['banknote','wdbc','spambase']):
    for method in colors:
        points=sorted([r for r in x['cells'] if r['task_id']==task and r['method']==method],key=lambda r:r['search_seed'])
        ax.plot([r['search_seed'] for r in points],[100*r['raw_accuracy'] for r in points],
                'o-',label=method,color=colors[method],linewidth=1.5,markersize=5)
    ax.set_title(task.upper(),loc='left');ax.set_xticks([11,22,33]);ax.set_xlabel('Search seed')
    ax.set_ylim(95,100.5);ax.grid(alpha=.18)
axes[0].set_ylabel('Final test accuracy (%)')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.90),ncol=3,frameon=False)
fig.suptitle('Complete 27-cell matrix after the final-only amendment',fontsize=15,y=.99)
fig.text(.5,.025,'10 original predictions reused under the audit qualification; 17 new final jobs. Original canonical audit remains FAIL.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.83));fig.savefig(plots/'final-accuracy.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12.5,6.8),sharey=True)
tasks=['banknote','wdbc','spambase'];seeds=['11','22','33']
names=[f'{t} / {s}' for t in tasks for s in seeds]+['Equal-task macroaverage']
for ax,name,base in zip(axes,['primary','secondary'],['RANDOM','FIXED']):
    c=x['contrasts'][name]
    vals=[100*c['paired_differences'][task][seed] for task in tasks for seed in seeds]
    ax.axvline(0,color='#aaaaaa',linewidth=1)
    ax.scatter(vals,range(9),color='#2563a6',s=40,zorder=3)
    for y,v in enumerate(vals): ax.annotate(f'{v:+.3f}',(v,y),xytext=(7,-4),textcoords='offset points',fontsize=9)
    m=100*c['estimate'];lo=100*c['interval']['low'];hi=100*c['interval']['high']
    ax.errorbar([m],[9],xerr=[[m-lo],[hi-m]],fmt='D',color='#191919',capsize=5,markersize=6)
    ax.annotate(f'{m:+.3f}  [{lo:+.3f}, {hi:+.3f}]',(m,9),xytext=(0,-23),textcoords='offset points',ha='center',fontsize=9)
    ax.axhline(8.5,color='#dddddd',linewidth=1)
    ax.set_title(f'{name.capitalize()}: ISTRATDE - {base}',loc='left',fontsize=12)
    ax.set_yticks(range(10),names);ax.set_ylim(10,-.6);ax.set_xlim(-3.2,1.05)
    ax.set_xlabel('Accuracy difference (percentage points)');ax.grid(axis='y',alpha=.12)
fig.suptitle('No demonstrated gain in this amended pilot',fontsize=15,y=.99)
fig.text(.5,.045,'Macroaverage bars: 95% within-task paired-seed bootstrap, 10,000 draws, seed 20261010.',ha='center',fontsize=9)
fig.text(.5,.018,'Exploratory intervals conditional on these three fixed test sets; only three search seeds. Both intervals include zero.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.085,1,.94));fig.savefig(plots/'paired-differences.png',dpi=160);plt.close(fig)
print(json.dumps({'plots':[str(p) for p in plots.glob('*.png')]}))
