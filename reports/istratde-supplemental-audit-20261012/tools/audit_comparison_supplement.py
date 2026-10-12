"""Read-only supplemental audit of one immutable stopped experiment.

The production validator still rejects the original ledger. A separately named,
in-memory validator admits ONLY the exact observed pair of clock values while
retaining all other original _load checks. No original bytes or production
functions are patched. This is not a run/resume authorization or canonical PASS.
"""
import ast
import copy
import csv
import hashlib
import inspect
import io
import json
import math
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.comparison import budget, final, review
from research.comparison.data import load_task_bindings
from research.comparison.native import SubprocessNativeProvider
from research.comparison.orchestrate import Orchestrator
from research.comparison.specs import iter_cells, iter_inner_runs
from research.search.acceptance_inner import load_sandbox_api_key
import httpx

RUN = ROOT / '.runtime/comparison-istratde'
OUT = ROOT / '.runtime/comparison-supplemental-audit-20261012'
ORIGINAL_SHA = 'b832658b09995146d7e06824a998853ee9cfdba531516843b9d8df2e74634a9c'
COMMIT = '87569545793c141e393e72450094975733a2bb95'
STATE_AT = 1791741439.3650322
EVENT_AT = 1791741439.3650367
checks = []

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    assert not path.is_symlink() and path.is_file(), str(path)
    return json.loads(path.read_text())

def passed(name, detail):
    checks.append({'name':name, 'status':'pass', 'detail':detail})
    print(json.dumps({'check':name,'status':'pass','detail':detail}), flush=True)

supervisor = read(ROOT / '.runtime/comparison-live-supervisor-state.json')
assert supervisor['phase'] == 'finished' and supervisor['source_commit'] == COMMIT
assert sha(RUN / 'comparison-ledger.json') == ORIGINAL_SHA
assert not OUT.exists()
OUT.mkdir()
# Hash the complete experiment tree before/after, including all candidate evidence.
before = {p.relative_to(RUN).as_posix():sha(p) for p in RUN.rglob('*') if p.is_file()}
assert not any(p.is_symlink() for p in RUN.rglob('*'))
passed('original_snapshot', {'files':len(before),'ledger_sha256':ORIGINAL_SHA})
manifest = read(RUN / 'experiment-manifest.v1.json')
bindings = load_task_bindings()
cells = list(iter_cells(bindings, COMMIT))
try:
    budget.ReadOnlyComparisonLedger.open(RUN / 'comparison-ledger.json', cells=cells)
except budget.ComparisonBudgetError as exc:
    assert str(exc) == 'corrupt_ledger:final_phase'
else:
    raise AssertionError('Original canonical ledger must still fail.')
passed('canonical_failure_preserved','corrupt_ledger:final_phase; no repair or overwrite')

# Confirm the validator used here is semantically identical to the executed one.
original_ast = ast.parse(subprocess.check_output(['git','show',COMMIT+':research/comparison/budget.py'],cwd=ROOT,text=True))
original_load = next(n for cls in original_ast.body if isinstance(cls,ast.ClassDef) and cls.name=='ComparisonLedger'
                     for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_load')
current_load = ast.parse(textwrap.dedent(inspect.getsource(budget.ComparisonLedger._load))).body[0]
assert ast.dump(original_load,include_attributes=False)==ast.dump(current_load,include_attributes=False)
needle = ast.parse('transition["at"] != final_phase["at"]',mode='eval').body
replacement = ast.parse('(transition["at"] != final_phase["at"] and not '
                        '(transition["at"] == 1791741439.3650367 and final_phase["at"] == 1791741439.3650322))',mode='eval').body

class ClockException(ast.NodeTransformer):
    count = 0
    def visit_Compare(self,node):
        if ast.dump(node,include_attributes=False)==ast.dump(needle,include_attributes=False):
            self.count += 1
            return ast.copy_location(copy.deepcopy(replacement),node)
        return self.generic_visit(node)

transform = ClockException()
node = transform.visit(copy.deepcopy(current_load))
assert transform.count == 1
node.name = 'validate_exact_observed_clock_exception'
node.decorator_list = []
module = ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
namespace = dict(vars(budget))
exec(compile(module,'<supplemental-clock-exception>','exec'),namespace)
validate_exception = namespace[node.name]
state = validate_exception(RUN / 'comparison-ledger.json')
assert state == read(RUN / 'comparison-ledger.json')
assert state['final_phase']['at'] == STATE_AT
assert [e for e in state['events'] if e['type']=='final_phase'][0]['at'] == EVENT_AT
expected_experiment = {'identity':budget.comparison_experiment_identity(cells),
    'protocol_identity':cells[0].protocol_identity,'source_commit':COMMIT,
    'cells':sorted(c.identity() for c in cells),'runs':sorted(r.identity() for r in iter_inner_runs(cells))}
assert state['experiment'] == expected_experiment
for cell in cells:
    assert state['cells'][cell.identity()]['cell_key'] == cell.cell_key()
    for i in range(8):
        run = cell.inner_run(i)
        s = state['runs'][run.identity()]
        assert s['run_key']==run.run_key() and s['cell_identity']==cell.identity()
# The constructor is used explicitly only in this qualified audit. open() is not
# monkeypatched, and the returned state retains both original unequal times.
view = budget.ReadOnlyComparisonLedger(RUN/'comparison-ledger.json',state,lambda: supervisor['finished_at'])
snapshot = view.snapshot()
negative = []
for mutation in ('different_clock_pair','wrong_barrier','duplicate_phase','request_token_mismatch','global_total_mismatch','unknown_usage'):
    bad=copy.deepcopy(state)
    event=next(e for e in bad['events'] if e['type']=='final_phase')
    if mutation=='different_clock_pair': event['at']+=0.001
    elif mutation=='wrong_barrier': event['barrier_identity']='0'*64
    elif mutation=='duplicate_phase': bad['events'].append(copy.deepcopy(event))
    elif mutation=='request_token_mismatch': bad['requests'][0]['charged_tokens']+=1
    elif mutation=='global_total_mismatch': bad['global_committed']['requests']+=1
    elif mutation=='unknown_usage': bad['requests'][0]['usage_known']=False
    with tempfile.TemporaryDirectory(prefix='comparison-audit-negative-') as directory:
        path=Path(directory)/'synthetic-mutated-ledger.json'
        path.write_text(json.dumps(bad))
        try: validate_exception(path)
        except budget.ComparisonBudgetError: negative.append(mutation)
        else: raise AssertionError('Counterexample was accepted: '+mutation)
passed('qualified_ledger_validation',{'exception':'one exact observed state/event time pair only',
    'original_data_unchanged':True,'negative_controls_rejected':negative,
    'requests':len(snapshot['requests']),'tokens':snapshot['global']['committed']['tokens']})

class CachedNative(SubprocessNativeProvider):
    def __init__(self,root):
        super().__init__(root)
        self.cache={}
    def invoke(self,name,*args):
        key=(name,*args)
        if key not in self.cache:
            self.cache[key]=getattr(super(),name)(*args)
            print(json.dumps({'native_replay':name,'unique_calls':len(self.cache)}),flush=True)
        return self.cache[key]
    def setup_population(self,seed): return self.invoke('setup_population',seed)
    def first_batch(self,seed): return self.invoke('first_batch',seed)
    def second_batch(self,seed,fitness): return self.invoke('second_batch',seed,tuple(fitness))
    def complete_run(self,seed,first_fitness,second_fitness):
        return self.invoke('complete_run',seed,tuple(first_fitness),tuple(second_fitness))

provider=CachedNative(ROOT)
orch=Orchestrator.load(runtime_dir=RUN,repo_root=ROOT,bindings=bindings,provider=provider)
review.verify_frozen_manifest_binding(manifest,source_commit=COMMIT,schedule_identity_value=orch._schedule_identity,
                                     cells=orch.cells,bindings=bindings,repo_root=ROOT)
passed('frozen_manifest_and_orchestration',{'cells':27,'configurations':216,'native_unique_replays':len(provider.cache)})
barrier=orch.compute_barrier_payload(snapshot)
assert barrier['barrier_identity']==state['final_phase']['barrier_identity']
context=final.verify_final_context(orch,snapshot)
publication=context['publication']
assert publication['barrier_identity']==barrier['barrier_identity']
passed('barrier_and_publication',{'barrier_identity':barrier['barrier_identity'],'published_files':len(publication['files']),
    'scope':'Full checks after explicitly qualified ledger validation; canonical audit remains FAIL.'})

latest_request=max(r['committed_at'] for r in state['requests'])
latest_run=0.0
winner_rows=[]
for cell in cells:
    key=cell.cell_key()
    c=read(RUN/'cells'/key.replace(':','-')/'cell.json')
    assert len(c['configs'])==8 and c['terminal']['status']=='frozen'
    selected=max(range(8),key=lambda i:(c['configs'][str(i)]['raw_score'],-i))
    assert c['winner']['config_index']==selected
    times=[]
    for i in range(8):
        run=read(RUN/'runs'/(key+':'+str(i)).replace(':','-')/'run-result.json')
        times.append(run['finished_at'])
        assert run['runner_commit']==COMMIT and run['worker_cleanup_verified'] is True
    latest_run=max(latest_run,*times)
    winner_rows.append({'cell_key':key,'winner_config_index':selected,
        'source_sha256':c['winner']['source_sha256'],'last_search_finished_at':max(times)})
assert latest_request <= latest_run < STATE_AT < EVENT_AT <= publication['recorded_at']
preflight=read(RUN/'preflight-evidence.v1.json')
assert preflight['verdict']=='PASS' and all(g['status']=='pass' for g in preflight['gates'])
assert preflight['source_commit']==COMMIT and preflight['generated_at']<supervisor['started_at']
gates={g['name']:g for g in preflight['gates']}
assert gates['runtime_stack_identity']['detail']['phase']=='search'
assert gates['final_publication_placeholders']['status']=='pass'
for r in state['requests']:
    assert r['usage_known'] and r['reserved_at']<=r['committed_at']<STATE_AT
assert snapshot['global']['open_reservations']==0
passed('recorded_freeze_before_publication',{'latest_model_settlement':latest_request,'latest_search_finish':latest_run,
    'phase_state_at':STATE_AT,'phase_event_at':EVENT_AT,'publication_recorded_at':publication['recorded_at'],
    'winners_bound_by_original_barrier':27,'model_requests_after_final_phase':0,
    'limitation':'Durable hashes, trusted writer ordering and preflight isolation evidence; not a retrospective syscall trace.'})

def csv_rows(path):
    return list(csv.DictReader(io.StringIO(path.read_text())))

split_checks={}
data=read(ROOT/'research/comparison/data-manifest.v1.json')
for task in ('banknote','wdbc','spambase'):
    task_info=data['tasks'][task]
    for rel,expected in task_info['files'].items(): assert sha(ROOT/rel)==expected
    groups=[]
    for split,path in [('train',ROOT/f'tasks/comparison/{task}/data/public/train.csv'),
                       ('validation',ROOT/f'tasks/comparison/{task}/data/public/test.csv'),
                       ('final',ROOT/f'research/comparison/sealed/{task}/final_features.csv')]:
        rows=csv_rows(path)
        assert len(rows)==task_info['splits'][split]['rows']
        columns=[k for k in rows[0] if k not in ('id','label')]
        features={tuple(float(row[k]) for k in columns) for row in rows}
        assert all(math.isfinite(v) for row in features for v in row)
        groups.append(features)
    assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    split_checks[task]={'rows':{s:task_info['splits'][s]['rows'] for s in ('train','validation','final')},
                       'cross_split_feature_groups':0}
passed('independent_split_and_data_hashes',split_checks)

key=load_sandbox_api_key(ROOT/review.deploy.AUTH_FILE) if hasattr(review,'deploy') else None
if key is None:
    from research.comparison import deploy
    key=load_sandbox_api_key(ROOT/deploy.AUTH_FILE)
from research.comparison import deploy
final_rows=[]
backend_receipts=[]
for cell in cells:
    cell_key=cell.cell_key()
    directory=RUN/'final'/cell_key.replace(':','-')
    if not (directory/'intent.json').exists(): continue
    intent=read(directory/'intent.json')
    assert intent['winner']==barrier['cells'][cell_key]['winner']
    assert publication['recorded_at']<=intent['submitted_at']<=supervisor['finished_at']
    winner_bytes=(RUN/'cells'/cell_key.replace(':','-')/'winner.py').read_bytes()
    source_sha=hashlib.sha256(winner_bytes).hexdigest()
    assert source_sha==intent['winner']['source_sha256']
    saved=(directory/'final.json').exists()
    if saved:
        result=final._validate_final_result(orch,cell_key,directory/'final.json',ledger=view,cell=cell,
            barrier_entry=barrier['cells'][cell_key],barrier_identity=barrier['barrier_identity'],
            publication=publication,sealed_root=ROOT/'research/comparison/sealed')
        job_id=result['job_id']
    else:
        assert cell_key=='wdbc:11:FIXED'
        job_id='job_352f5c4e67b84932'
    response=httpx.get(deploy.GATEWAY_URL+'/api/v1/jobs/'+job_id,headers={'x-api-key':key},timeout=20)
    response.raise_for_status()
    body=response.json()
    backend=body['result']; ev=backend['evaluation']; gate=backend['source_gate']
    assert body['job_id']==job_id and body['status']=='completed'
    assert backend['source_identity_verified'] is True and backend['worker_cleanup_verified'] is True
    created=datetime.fromisoformat(body['created_at'].replace('Z','+00:00')).timestamp()
    assert intent['submitted_at']-1<=created<=supervisor['finished_at']
    path=ROOT/'.runtime/rsi-trustworthy/storage'/Path(ev['prediction_snapshot']).relative_to('/mnt/rsi_storage')
    assert sha(path)==ev['prediction_sha256']
    code=path.parent/'code/main.py'
    assert code.read_bytes()==winner_bytes
    answers=ROOT/f'research/comparison/sealed/{cell.task_id}/final_labels.csv'
    assert sha(answers)==ev['answer_sha256']==cell.task_binding.final_labels_sha256
    p={r['id']:r['label'] for r in csv_rows(path)}
    a={r['id']:r['label'] for r in csv_rows(answers)}
    assert len(p)==len(csv_rows(path)) and set(p)==set(a)
    assert all(v in ('0','1') for v in [*p.values(),*a.values()])
    correct=sum(p[k]==v for k,v in a.items()); exact=correct/len(a)
    assert abs(round(exact,6)-backend['score'])<=1e-9 and ev['score']==backend['score']
    cleanup=read(RUN/'final/executor'/cell_key.replace(':','-')/'live-jobs.json')
    evidence={**ev,**{k:gate[k] for k in ('entrypoint','entrypoint_sha256','source_sha256','policy_sha256','policy_version')},
              'job_id':job_id,'status':'completed','source_identity_verified':True,'worker_cleanup_verified':True,
              'submitted_source_sha256':source_sha,'cleanup':{k:cleanup[k] for k in ('live_jobs','uncertain_submissions','submission_intents')}}
    final._verify_execution_evidence(cell,status='completed',job_id=job_id,prediction_sha256=ev['prediction_sha256'],
        winner_source_sha256=source_sha,answer_sha256=ev['answer_sha256'],evidence=evidence,
        winner_bytes=winner_bytes,registry_version=publication['registry_version'],rows=len(a))
    if saved:
        assert result['prediction_sha256']==ev['prediction_sha256'] and result['raw_score']==backend['score']
        assert (directory/'prediction.csv').read_bytes()==path.read_bytes()
    backend_receipts.append({'cell_key':cell_key,'job_id':job_id,'created_at':body['created_at'],
        'completed_at':body['completed_at'],'score':backend['score'],
        'evaluation':{k:v for k,v in ev.items() if k!='prediction_snapshot'},'source_gate':gate,
        'source_identity_verified':True,'worker_cleanup_verified':True})
    final_rows.append({'cell_key':cell_key,'job_id':job_id,'saved_final_record':saved,
        'correct':correct,'rows':len(a),'exact_accuracy':exact,'backend_score':backend['score'],
        'source_sha256':source_sha,'prediction_sha256':ev['prediction_sha256'],
        'qualified_provenance_checked':True,'canonical_accepted':False})
assert len(final_rows)==10 and len({r['job_id'] for r in final_rows})==10
passed('ten_original_final_predictions',{'count':10,'saved_final_records':9,'backend_only_stopped_result':1,
    'all_source_prediction_answer_cleanup_checks':True,'new_jobs':0})

after={p.relative_to(RUN).as_posix():sha(p) for p in RUN.rglob('*') if p.is_file()}
assert before==after
passed('all_original_experiment_bytes_unchanged',{'files':len(before)})
payload={'schema':'istratde-supplemental-read-only-audit.v1','generated_at':datetime.now(timezone.utc).isoformat(),
    'executed_source_commit':COMMIT,'audit_tool_repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'original_canonical_audit':'FAIL','supplemental_verdict':'QUALIFIED_REUSABLE_SEARCH_AND_TEN_PREDICTIONS',
    'canonical_status_changed':False,'authorized_scope':'read-only supplemental audit only',
    'new_model_requests':0,'new_final_evaluations':0,'remaining_unattempted_finals':17,
    'clock_exception':{'original_ledger_sha256':ORIGINAL_SHA,'state_at':STATE_AT,'event_at':EVENT_AT,
        'original_validator_ast_sha256':hashlib.sha256(ast.dump(original_load).encode()).hexdigest(),
        'replaced_expression':ast.unparse(needle),'replacement_expression':ast.unparse(replacement),
        'one_clause_only':True,'no_production_patch':True,'no_original_data_change':True},
    'checks':checks,'winners':winner_rows,'final_predictions':final_rows,'backend_receipts':backend_receipts,
    'original_runtime_file_hashes':before,
    'limitations':['The timestamp exception is an explicit retrospective audit qualification, not canonical protocol compliance.',
                  'Isolation confidence uses frozen source, preflight probes, bound records and phase ordering; no per-syscall historical trace exists.',
                  'A separately authorized final-only stage with fixed winners and a new explicit deadline is required; the old stopped run must not be resumed.',
                  'No algorithm benefit can be inferred until the final matrix is complete.']}
(OUT/'audit.json').write_text(json.dumps(payload,indent=2)+'\n')
(OUT/'clock-exception-validator.py.txt').write_text(ast.unparse(module)+'\n')
print(json.dumps({'verdict':payload['supplemental_verdict'],'report':str(OUT/'audit.json')}),flush=True)
