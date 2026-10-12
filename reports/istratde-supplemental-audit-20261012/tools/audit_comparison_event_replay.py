"""Independent event-by-event accounting, including transient reservations."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

root=Path(__file__).resolve().parents[3]
path=root/'.runtime/comparison-istratde/comparison-ledger.json'
before=hashlib.sha256(path.read_bytes()).hexdigest()
assert before=='b832658b09995146d7e06824a998853ee9cfdba531516843b9d8df2e74634a9c'
state=json.loads(path.read_text())
requests={r['request_id']:r for r in state['requests']}
runs={r['run_key']:r for r in state['runs'].values()}
pending={}
settled=set()
global_used=Counter()
cell_used=defaultdict(Counter)
run_used=defaultdict(Counter)
peaks=Counter()
cell_peaks=defaultdict(Counter)
phase=False
previous=0.0
for event in state['events']:
    assert event['at']>=previous
    previous=event['at']
    kind=event['type']
    if kind=='reserve':
        assert not phase
        request_id=event['request_id']
        assert request_id not in pending and request_id not in settled
        record=requests[request_id]
        run=runs[event['run_key']]
        cell=state['cells'][run['cell_identity']]['cell_key']
        assert event['kind']==record['kind'] and event['tokens']==record['reserved_tokens']
        pending[request_id]={'cell':cell,'run':event['run_key'],'kind':event['kind'],'tokens':event['tokens']}
    elif kind=='commit':
        request_id=event['request_id']
        reservation=pending.pop(request_id)
        record=requests[request_id]
        assert event['usage_known'] is True
        for field in ('input_tokens','output_tokens','charged_tokens','reserved_tokens','kind'):
            assert event[field]==record[field]
        assert event['charged_tokens']==event['input_tokens']+event['output_tokens']<=reservation['tokens']
        charge=Counter(requests=1,tokens=event['charged_tokens'],**{event['kind']:1})
        global_used.update(charge)
        cell_used[reservation['cell']].update(charge)
        run_used[reservation['run']].update(charge)
        settled.add(request_id)
    elif kind=='final_phase':
        assert not pending and not phase
        phase=True
    elif kind=='stop':
        assert phase and not pending
    else:
        assert kind=='start_clock'
    global_requests=global_used['requests']+len(pending)
    global_tokens=global_used['tokens']+sum(v['tokens'] for v in pending.values())
    assert global_requests<=1728 and global_tokens<=5400000
    peaks['committed_plus_reserved_requests']=max(peaks['committed_plus_reserved_requests'],global_requests)
    peaks['committed_plus_reserved_tokens']=max(peaks['committed_plus_reserved_tokens'],global_tokens)
    peaks['simultaneous_model_reservations']=max(peaks['simultaneous_model_reservations'],len(pending))
    for cell in {v['cell'] for v in pending.values()}|set(cell_used):
        active=[v for v in pending.values() if v['cell']==cell]
        n=cell_used[cell]['requests']+len(active)
        t=cell_used[cell]['tokens']+sum(v['tokens'] for v in active)
        assert n<=64 and t<=200000
        cell_peaks[cell]['requests']=max(cell_peaks[cell]['requests'],n)
        cell_peaks[cell]['tokens']=max(cell_peaks[cell]['tokens'],t)
    for run_key in {v['run'] for v in pending.values()}|set(run_used):
        active=Counter(v['kind'] for v in pending.values() if v['run']==run_key)
        assert run_used[run_key]['main']+active['main']<=6
        assert run_used[run_key]['debug']+active['debug']<=2
assert not pending and settled==set(requests) and phase
assert peaks['simultaneous_model_reservations']==1
assert global_used['requests']==state['global_committed']['requests']
assert global_used['tokens']==state['global_committed']['tokens']
for s in state['cells'].values():
    assert s['committed']=={'requests':cell_used[s['cell_key']]['requests'],'tokens':cell_used[s['cell_key']]['tokens']}
assert hashlib.sha256(path.read_bytes()).hexdigest()==before
result={'schema':'comparison-independent-event-replay.v1','original_ledger_sha256':before,
    'events_replayed':len(state['events']),'unique_requests':len(requests),
    'all_historical_reservation_caps_respected':True,'event_timestamps_nondecreasing':True,
    'global_peaks':dict(peaks),'cell_peaks':dict(cell_peaks),'actual_settled':dict(global_used),
    'requests_after_final_phase':0,'unknown_usage':0,'ledger_unchanged':True,
    'canonical_audit_status_changed':False}
out=root/'.runtime/comparison-supplemental-audit-20261012/event-replay.json'
with out.open('x') as handle: json.dump(result,handle,indent=2); handle.write('\n')
print(json.dumps({k:v for k,v in result.items() if k!='cell_peaks'}))
