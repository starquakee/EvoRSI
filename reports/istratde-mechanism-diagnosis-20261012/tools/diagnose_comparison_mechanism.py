"""Offline mechanism diagnosis on the immutable 216 searches; no execution."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'.runtime/comparison-istratde'
OUT=ROOT/'.runtime/comparison-mechanism-diagnosis'
input_hashes={}
def raw(path,expected=None):
    assert path.is_file() and not path.is_symlink()
    data=path.read_bytes(); digest=hashlib.sha256(data).hexdigest()
    if expected: assert digest==expected,str(path)
    input_hashes[str(path.relative_to(ROOT))]=digest
    return data
def obj(path,expected=None): return json.loads(raw(path,expected))
def jsonl(path,expected): return [json.loads(line) for line in raw(path,expected).decode().splitlines() if line.strip()]
def softmax(values,temp=1.):
    maximum=max(values);exp=[math.exp((v-maximum)/temp) for v in values];denom=sum(exp)
    return [v/denom for v in exp]
def avg(values): return statistics.mean(values) if values else None
def correlation(a,b):
    if len(a)<2 or len(set(a))<2 or len(set(b))<2: return None
    return statistics.correlation(a,b)

runs=[];selections=[];operators=[];families=Counter(); by_cell={}
for path in sorted((RUN/'cells').glob('*/cell.json')):
    cell=obj(path); by_cell[cell['cell_key']]=cell
    for index,config in sorted(cell['configs'].items(),key=lambda x:int(x[0])):
        assert config['status']=='completed'
        directory=RUN/'runs'/(cell['cell_key']+':'+index).replace(':','-')
        result=obj(directory/'run-result.json',config['result_sha256'])
        trace=jsonl(directory/'operator-trace.jsonl',result['operator_trace_sha256'])
        nodes=jsonl(directory/'journal.jsonl',result['journal_sha256'])
        nodes_by_id={n['id']:n for n in nodes}
        theta=result['decoded']; counts=Counter(t['operator'] for t in trace)
        assert all(counts[k]==v for k,v in result['operator_counts'].items())
        initial_scores=[]; realized_debug_depths=[]
        for event in trace:
            cfg=event['config']; weights=cfg['experience']['parent_selection']['weights']
            assert abs(cfg['crossover_prob']-theta['crossover_prob'])<1e-12
            assert cfg['max_debug_depth']==theta['max_debug_depth']
            for component in ('score','delta','novelty'):
                assert abs(weights[component]-theta[component+'_weight'])<1e-12
            child=nodes_by_id[event['child_node_id']]
            assert hashlib.sha256(child['code'].encode()).hexdigest()==event['child_code_sha256']
            if event['operator']=='draft' and child['metric'] is not None: initial_scores.append(child['metric'])
            parent_nodes=[nodes_by_id[p] for p in event['parent_node_ids']]
            for p,code_sha in zip(parent_nodes,event['parent_code_sha256']):
                assert hashlib.sha256(p['code'].encode()).hexdigest()==code_sha
            valid_parents=[p['metric'] for p in parent_nodes if isinstance(p['metric'],(int,float)) and not p['is_buggy']]
            score=child['metric'] if not child['is_buggy'] else None
            operators.append({'cell_key':cell['cell_key'],'config_index':int(index),'task_id':cell['task_id'],
                'method':cell['method'],'operator':event['operator'],'generation':event['generation_id'],
                'child_success':score is not None,'child_validation_score':score,
                'best_parent_score':max(valid_parents) if valid_parents else None,
                'beats_best_parent':score is not None and bool(valid_parents) and score>max(valid_parents)})
            if event['operator']=='debug':
                depth=1; parent=parent_nodes[0]
                seen=set()
                while 'debug' in parent.get('operators_used',[]) and parent['id'] not in seen:
                    seen.add(parent['id']); depth+=1
                    parent_ids=parent.get('parents',[])
                    if not parent_ids: break
                    # Journal parent references are step indices.
                    parent=next(n for n in nodes if n['step']==parent_ids[0])
                realized_debug_depths.append(depth)
            selection=event.get('selection')
            if event['operator'] not in ('improve','crossover'): continue
            assert selection and selection['enabled']
            candidates=selection['candidates']; assert candidates
            assert selection['selected_node_ids']==event['parent_node_ids']
            assert selection['weights']==weights
            utilities=[sum(weights[k]*c[k+'_component'] for k in ('score','delta','novelty'))-c['official_score_missing_penalty'] for c in candidates]
            probs=softmax(utilities,selection['temperature'])
            for c,u,p in zip(candidates,utilities,probs):
                assert abs(c['utility']-u)<1e-10 and abs(c['probability']-p)<1e-10
                assert c['fitness']==nodes_by_id[c['node_id']]['metric']
                # In this comparison, the journal metric is the trusted validation score.
                assert c['official_score']==c['fitness'] and c['official_status']=='completed'
                families[c['method_family_auto']]+=1
            row={'cell_key':cell['cell_key'],'config_index':int(index),'task_id':cell['task_id'],
                 'method':cell['method'],'operator':event['operator'],'generation':event['generation_id'],
                 'pool_size':len(candidates),'parents_selected':len(selection['selected_node_ids']),
                 'entire_parent_set_forced':len(candidates)==len(selection['selected_node_ids'])}
            for component in ('score','delta','novelty'):
                values=[c[component+'_component'] for c in candidates]
                spread=max(values)-min(values)
                altered=[]
                for weight in (0.,2.):
                    us=[u+(weight-weights[component])*c[component+'_component'] for u,c in zip(utilities,candidates)]
                    altered.append(softmax(us,selection['temperature']))
                tv=.5*sum(abs(a-b) for a,b in zip(*altered))
                row[component+'_spread']=spread
                row[component+'_informative']=spread>1e-12
                row[component+'_tv_weight_0_to_2']=tv
                if spread<=1e-12: assert tv<1e-12
            selections.append(row)
        incumbent=result['incumbent'];winner=nodes_by_id[incumbent['node_id']]
        runrow={'cell_key':cell['cell_key'],'task_id':cell['task_id'],'method':cell['method'],
            'seed':cell['search_seed'],'config_index':int(index),**theta,
            **{'operators_'+k:counts[k] for k in ('draft','improve','crossover','debug')},
            'validation_best':config['raw_score'],'initial_draft_best':max(initial_scores) if initial_scores else None,
            'improvement_over_drafts':config['raw_score']-max(initial_scores) if initial_scores else None,
            'incumbent_operator':winner['operators_used'][-1],
            'debug_executed':bool(counts['debug']),'max_realized_debug_depth':max(realized_debug_depths,default=0),
            'main_candidates':result['main_candidates'],'search_execution_failures':result['job_counts']['execution_failed']}
        runs.append(runrow)
assert len(runs)==216

def summarize_selection(values):
    return {'events':len(values),'forced_parent_sets':sum(v['entire_parent_set_forced'] for v in values),
            'components':{k:{'informative_events':sum(v[k+'_informative'] for v in values),
                'constant_events':sum(not v[k+'_informative'] for v in values),
                'mean_first_draw_tv_for_0_to_2':avg([v[k+'_tv_weight_0_to_2'] for v in values]),
                'max_first_draw_tv_for_0_to_2':max([v[k+'_tv_weight_0_to_2'] for v in values],default=0)}
                for k in ('score','delta','novelty')}}
summary={'runs':len(runs),'operator_events':len(operators),'operator_counts':dict(Counter(v['operator'] for v in operators)),
    'selection':summarize_selection(selections),'selection_by_task':{t:summarize_selection([s for s in selections if s['task_id']==t]) for t in ('banknote','wdbc','spambase')},
    'selection_by_generation':{str(g):summarize_selection([s for s in selections if s['generation']==g]) for g in (1,2)},
    'method_family_candidate_occurrences':dict(families),
    'debug_by_configured_depth':{str(d):{'runs':sum(r['max_debug_depth']==d for r in runs),
        'runs_with_debug':sum(r['max_debug_depth']==d and r['debug_executed'] for r in runs),
        'debug_requests':sum(r['operators_debug'] for r in runs if r['max_debug_depth']==d),
        'depth_two_reached_runs':sum(r['max_debug_depth']==d and r['max_realized_debug_depth']>=2 for r in runs)} for d in (0,1,2)},
    'inner_search_by_task':{t:{'runs':72,'unique_configuration_best_scores':len({r['validation_best'] for r in runs if r['task_id']==t}),
        'runs_improved_over_initial_drafts':sum((r['improvement_over_drafts'] or 0)>1e-12 for r in runs if r['task_id']==t),
        'winner_operators':dict(Counter(r['incumbent_operator'] for r in runs if r['task_id']==t))} for t in ('banknote','wdbc','spambase')},
    'crossover_probability_bins':{},'matched_initial_config_noise':{},'outer_second_batch':[]}
for low,high in [(0,.25),(.25,.5),(.5,.75),(.75,1.01)]:
    subset=[r for r in runs if low<=r['crossover_prob']<high]
    summary['crossover_probability_bins'][f'[{low},{high})']={'runs':len(subset),
        'mean_configured_probability':avg([r['crossover_prob'] for r in subset]),
        'actual_crossover_fraction':sum(r['operators_crossover'] for r in subset)/(4*len(subset)) if subset else None}
for task in ('banknote','wdbc','spambase'):
    gaps=[]
    for seed in (11,22,33):
        a=by_cell[f'{task}:{seed}:ISTRATDE'];b=by_cell[f'{task}:{seed}:RANDOM']
        for i in range(4):
            assert a['vectors'][str(i)]==b['vectors'][str(i)]
            gaps.append(abs(a['configs'][str(i)]['raw_score']-b['configs'][str(i)]['raw_score']))
    summary['matched_initial_config_noise'][task]={'pairs':12,'different_scores':sum(g>0 for g in gaps),
                                                  'mean_abs_gap':avg(gaps),'max_abs_gap':max(gaps)}
for cell in by_cell.values():
    scores=[cell['configs'][str(i)]['raw_score'] for i in range(8)]
    summary['outer_second_batch'].append({'cell_key':cell['cell_key'],'first4_best':max(scores[:4]),
        'second4_best':max(scores[4:]),'incumbent_gain':max(scores)-max(scores[:4]),
        'winning_config':cell['winner']['config_index']})
for rel,h in input_hashes.items(): assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
result={'schema':'comparison-offline-mechanism-diagnosis.v1','summary':summary,'all_five_parameters_match_actual_operator_config':True,
    'all_logged_utilities_and_probabilities_independently_reproduced':True,'runs':runs,'selections':selections,'operators':operators,
    'input_hashes':input_hashes,'original_files_unchanged':True,'new_model_requests':0,'new_sandbox_jobs':0,
    'limitations':['Counterfactual weight changes hold the logged candidate pool fixed and measure first-draw probabilities only, not new generated code or accuracy.',
                   'When all candidate parents are selected, weights cannot alter the unordered set but may change order and hence the prompt.',
                   'Observed operator success and score differences are descriptive, confounded by task, candidate pool and stochastic generation.',
                   'Method-family counts are selection-trace occurrences, not counts of unique programs.']}
OUT.mkdir()
(OUT/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
for name,values in [('runs-216.csv',runs),('parent-selections.csv',selections),('operator-events.csv',operators)]:
    with (OUT/name).open('x',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(values[0]),lineterminator='\n');w.writeheader();w.writerows(values)
print(json.dumps({k:v for k,v in summary.items() if k!='outer_second_batch'}))
