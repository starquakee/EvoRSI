"""Independent final CSV/source/backend checks and frozen paired statistics."""
import csv
import hashlib
import io
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from research.comparison import final, deploy, analysis
from research.comparison.data import load_task_bindings
from research.comparison.specs import iter_cells
from research.comparison.final_only_stage import verify_original
from research.search.acceptance_inner import load_sandbox_api_key
import httpx

STAGE=ROOT/'.runtime/final-only-amendment-001'
ORIG=ROOT/'.runtime/comparison-istratde'
OUT=ROOT/'.runtime/final-only-amendment-report'
def obj(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
outcome=obj(STAGE/'outcome.json')
assert outcome['status']=='completed' and len(outcome['results'])==17
assert not OUT.exists()
audit=obj(ROOT/'.runtime/comparison-supplemental-audit-20261012/audit.json')
verify_original(ROOT,audit)
before={p.relative_to(STAGE).as_posix():sha(p) for p in STAGE.rglob('*') if p.is_file()}
authorization=obj(STAGE/'authorization.json'); launch=obj(STAGE/'launch.json')
assert authorization['execution_authorized'] and launch['max_model_requests']==0 and launch['max_new_jobs']==17
assert outcome['elapsed_monotonic']<=14400 and outcome['finished_at']<=launch['deadline_epoch']
events=[obj(p) for p in sorted((STAGE/'events').glob('*.json'))]
assert [e['sequence'] for e in events]==list(range(len(events)))
assert events[0]['type']=='started' and events[-1]['type']=='completed'
assert sum(e['type']=='submission_intent' for e in events)==17
assert sum(e['type']=='result' for e in events)==17
assert not any(e['type']=='stopped' for e in events)
assert len(list((STAGE/'final').glob('*/intent.json')))==17
assert len(list((STAGE/'final').glob('*/final.json')))==17
old={r['cell_key']:r for r in audit['final_predictions']}
publication=obj(ORIG/'final/publication.json')
cells=list(iter_cells(load_task_bindings(),audit['executed_source_commit']))
key=load_sandbox_api_key(ROOT/deploy.AUTH_FILE)
rows=[]; receipts=[]; checked_hashes={}

def binary_csv(p):
    raw=p.read_bytes(); checked_hashes[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    records=list(csv.reader(io.StringIO(raw.decode(),newline='')))
    assert records[0]==['id','label']
    values={}
    for row in records[1:]:
        assert len(row)==2 and row[0] and row[0] not in values and row[1] in ('0','1')
        values[row[0]]=int(row[1])
    return values

for cell in cells:
    cell_key=cell.cell_key(); slug=cell_key.replace(':','-')
    source=ORIG/'cells'/slug/'winner.py'; source_bytes=source.read_bytes(); source_sha=sha(source)
    original=cell_key in old
    result=None
    if original:
        prior=old[cell_key]; job_id=prior['job_id']; expected_prediction=prior['prediction_sha256']
        assert source_sha==prior['source_sha256']
    else:
        result=obj(STAGE/'final'/slug/'final.json'); intent_path=STAGE/'final'/slug/'intent.json'; intent=obj(intent_path)
        assert result['intent_sha256']==sha(intent_path)
        assert source_sha==result['winner_source_sha256']==intent['source_sha256']
        assert result['barrier_identity']==publication['barrier_identity']
        assert intent['cell_key']==cell_key and result['cell_key']==cell_key
        assert launch['started_at']<=intent['submitted_at']<=result['finished_at']<=launch['deadline_epoch']
        assert intent['timeout_seconds']<=600
        assert result['status']=='completed', 'This concrete run expects all observed successful finals; do not impute.'
        job_id=result['job_id']; expected_prediction=result['prediction_sha256']
    response=httpx.get(deploy.GATEWAY_URL+'/api/v1/jobs/'+job_id,headers={'x-api-key':key},timeout=20)
    response.raise_for_status(); body=response.json(); backend=body['result']; ev=backend['evaluation']; gate=backend['source_gate']
    assert body['job_id']==job_id and body['status']=='completed'
    assert backend['source_identity_verified'] and backend['worker_cleanup_verified']
    prediction=deploy.job_snapshot_host_path(ev['prediction_snapshot'],job_id,storage_root=ROOT/deploy.DISPATCHER_STORAGE_ROOT)
    assert sha(prediction)==expected_prediction==ev['prediction_sha256']
    assert (prediction.parent/'code/main.py').read_bytes()==source_bytes
    answers=ROOT/'research/comparison/sealed'/cell.task_id/'final_labels.csv'
    assert sha(answers)==ev['answer_sha256']==cell.task_binding.final_labels_sha256
    p=binary_csv(prediction); a=binary_csv(answers); assert p and set(p)==set(a)
    correct=sum(p[k]==v for k,v in a.items()); exact=correct/len(a); score=backend['score']
    assert abs(round(exact,6)-score)<=1e-9 and score==ev['score']
    cleanup_path=(ORIG/'final/executor'/slug/'live-jobs.json') if original else STAGE/'executor'/slug/'live-jobs.json'
    cleanup=obj(cleanup_path)
    evidence={**ev,**{k:gate[k] for k in ('entrypoint','entrypoint_sha256','source_sha256','policy_sha256','policy_version')},
        'job_id':job_id,'status':'completed','source_identity_verified':True,'worker_cleanup_verified':True,
        'submitted_source_sha256':source_sha,'cleanup':{k:cleanup[k] for k in ('live_jobs','uncertain_submissions','submission_intents')}}
    final._verify_execution_evidence(cell,status='completed',job_id=job_id,prediction_sha256=expected_prediction,
        winner_source_sha256=source_sha,answer_sha256=cell.task_binding.final_labels_sha256,evidence=evidence,
        winner_bytes=source_bytes,registry_version=publication['registry_version'],rows=len(a))
    if result:
        assert (STAGE/'final'/slug/'prediction.csv').read_bytes()==prediction.read_bytes()
        assert result['raw_score']==result['utility']==score and abs(result['recomputed_accuracy']-exact)<1e-12
    else:
        assert prior['backend_score']==score and abs(prior['exact_accuracy']-exact)<1e-12
    rows.append({'cell_key':cell_key,'task_id':cell.task_id,'method':cell.method,'search_seed':cell.search_seed,
        'stage':'original_qualified_reuse' if original else 'final_only_amendment',
        'status':'completed','correct':correct,'test_rows':len(a),'exact_accuracy':exact,
        'raw_accuracy':score,'utility':score,'source_sha256':source_sha,'prediction_sha256':expected_prediction,
        'job_id':job_id,'source_and_cleanup_verified':True,'original_canonical_audit':'FAIL'})
    receipts.append({'cell_key':cell_key,'job_id':job_id,'status':body['status'],
        'started_at':body['started_at'],'completed_at':body['completed_at'],
        'score':score,'evaluation':{k:v for k,v in ev.items() if k!='prediction_snapshot'},'source_gate':gate,
        'source_identity_verified':True,'worker_cleanup_verified':True})
assert len(rows)==27 and len({r['job_id'] for r in rows})==27
mapped={r['cell_key']:analysis.CellOutcome(cell_key=r['cell_key'],task_id=r['task_id'],method=r['method'],
    search_seed=r['search_seed'],observed=True,raw_accuracy=r['raw_accuracy'],utility=r['utility']) for r in rows}
contrasts={}
independent={}
for name,baseline in [('primary','RANDOM'),('secondary','FIXED')]:
    contrast=analysis.analyze_contrast(mapped,treatment='ISTRATDE',baseline=baseline,expected_pairs=9)
    diffs={}
    for task in sorted({r['task_id'] for r in rows}):
        diffs[task]=[]
        for seed in (11,22,33):
            treated=next(r for r in rows if r['task_id']==task and r['search_seed']==seed and r['method']=='ISTRATDE')
            base=next(r for r in rows if r['task_id']==task and r['search_seed']==seed and r['method']==baseline)
            diffs[task].append(treated['utility']-base['utility'])
    estimate=sum(sum(v)/3 for v in diffs.values())/3
    rng=random.Random(20261010); samples=[]
    for _ in range(10000):
        resampled=[sum(values[rng.randrange(3)] for _ in range(3))/3 for values in diffs.values()]
        samples.append(sum(resampled)/3)
    samples.sort()
    def quantile(q):
        pos=(len(samples)-1)*q; idx=int(pos); weight=pos-idx
        return samples[idx]*(1-weight)+samples[min(idx+1,len(samples)-1)]*weight
    interval=[quantile(.025),quantile(.975)]
    assert abs(estimate-contrast['estimate'])<1e-12
    assert abs(interval[0]-contrast['interval']['low'])<1e-12 and abs(interval[1]-contrast['interval']['high'])<1e-12
    contrasts[name]=contrast; independent[name]={'estimate':estimate,'interval':interval,'diffs_by_task':diffs}
means={task:{method:sum(r['raw_accuracy'] for r in rows if r['task_id']==task and r['method']==method)/3
        for method in ('FIXED','RANDOM','ISTRATDE')} for task in ('banknote','wdbc','spambase')}
macros={method:sum(v[method] for v in means.values())/3 for method in ('FIXED','RANDOM','ISTRATDE')}
verify_original(ROOT,audit)
assert before=={p.relative_to(STAGE).as_posix():sha(p) for p in STAGE.rglob('*') if p.is_file()}
for relative,expected in checked_hashes.items(): assert sha(ROOT/relative)==expected
summary={'schema':'istratde-final-only-amended-report.v1','matrix_complete':True,'cells':rows,
    'qualified_original_reused':10,'new_final_jobs':17,'new_model_requests':0,'new_model_tokens':0,
    'original_canonical_audit':'FAIL','analysis_status':'complete_matrix_under_documented_post_stop_amendment',
    'stage_outcome':outcome,'authorization':authorization,'task_method_means':means,'method_macroaverages':macros,
    'contrasts':contrasts,'independent_statistics':independent,
    'mechanical_predeclared_conclusion':analysis.conclude(contrasts['primary'],matrix_complete=True),
    'all_27_independent_csv_source_answer_backend_cleanup_checks':True,'all_original_files_unchanged':True,
    'original_runtime_files_checked':len(audit['original_runtime_file_hashes']),
    'backend_receipts':receipts,'stage_file_hashes':before,'prediction_and_answer_hashes':checked_hashes,
    'limitations':['Retrospective timestamp qualification and final-phase amendment; original canonical audit remains FAIL.',
                  'Three search seeds and three fixed test sets only; exploratory conditional intervals are not broad generalization.',
                  'Prespecified score uses trusted six-decimal accuracy; exact correct/rows retained separately.',
                  'No new search/model calls or data/winner changes; all candidate selection preceded original final publication.']}
OUT.mkdir()
(OUT/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'results-27-cells.csv').open('x',newline='') as handle:
    w=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print(json.dumps({'means':means,'macroaverages':macros,'contrasts':independent,
                  'conclusion':summary['mechanical_predeclared_conclusion'],'new_jobs':17,'new_model_calls':0,
                  'elapsed_seconds':outcome['elapsed_monotonic']}))
