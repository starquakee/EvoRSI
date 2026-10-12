"""Read-only GET of the original rejected final job; allowlisted receipt only."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.comparison import deploy
from research.search.acceptance_inner import load_sandbox_api_key

root = Path(__file__).resolve().parents[3]
out = root / '.runtime/comparison-reports/forensic'
summary = json.loads((out / 'forensic-summary.json').read_text())
assert summary['run_supervisor']['phase'] == 'finished'
diagnosis = summary['wdbc_diagnosis']
key = load_sandbox_api_key(root / deploy.AUTH_FILE)
job = diagnosis['job_id']
response = httpx.get(deploy.GATEWAY_URL + '/api/v1/jobs/' + job,
                     headers={'x-api-key': key}, timeout=20)
response.raise_for_status()
body = response.json()
result = body['result']
assert body['job_id'] == job and body['status'] == 'completed'
assert result['score'] == diagnosis['trusted_metric_replayed_round6']
assert result['source_identity_verified'] is True and result['worker_cleanup_verified'] is True
ev = result['evaluation']
assert ev['task_id'] == 'comparison_wdbc_final' and ev['split'] == 'final'
assert ev['evaluator'] == 'external_trusted' and ev['score'] == result['score']
assert ev['row_count'] == diagnosis['rows']
assert result['source_gate']['entrypoint_sha256'] == diagnosis['source_sha256']
relative = str(Path(ev['prediction_snapshot']).relative_to('/mnt/rsi_storage'))
pred = root / '.runtime/rsi-trustworthy/storage' / relative
assert hashlib.sha256(pred.read_bytes()).hexdigest() == ev['prediction_sha256']
truth = root / 'research/comparison/sealed/wdbc/final_labels.csv'
assert hashlib.sha256(truth.read_bytes()).hexdigest() == ev['answer_sha256']
receipt = {'schema': 'comparison-post-stop-original-job-receipt.v1',
           'captured_at': datetime.now(timezone.utc).isoformat(),
           'scope': 'Read-only backend GET of the original job, no re-execution. Rejected by original final integrity gate; not an accepted final result.',
           **{k: body[k] for k in ('job_id', 'task_id', 'status', 'created_at', 'started_at', 'completed_at')},
           'score': result['score'], 'evaluation': {k:v for k,v in ev.items() if k != 'prediction_snapshot'},
           'source_gate': result['source_gate'], 'source_identity_verified': True,
           'worker_cleanup_verified': True, 'accepted_final_result': False}
with (out / 'wdbc-original-job-receipt.json').open('x') as handle:
    json.dump(receipt, handle, indent=2)
    handle.write('\n')
print(json.dumps({'job_id':job,'original_backend_score':result['score'],'prediction_and_answer_hashes_match':True}))
