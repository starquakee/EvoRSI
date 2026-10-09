"""First-party GPU fixture through the real API; no model requests."""
from __future__ import annotations

import json
from pathlib import Path
import time

from smoke_e2e import load_api_key, request

ROOT = Path(__file__).resolve().parents[2]
CODE = '''import csv, json, os
import torch
assert os.getuid() == 65432
cap = next(line.split()[1] for line in open('/proc/self/status') if line.startswith('CapEff:'))
assert int(cap, 16) == 0
assert torch.cuda.is_available()
value = torch.tensor([2.0], device='cuda').sum().item()
assert value == 2.0
print('GPU_PROBE ' + json.dumps({'uid': os.getuid(), 'cap_eff': cap, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0), 'tensor_result': value}))
with open('submission.csv', 'w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['id', 'label'])
    for identifier in range(200, 240):
        writer.writerow([identifier, 0])
'''


def main() -> int:
    key = load_api_key()
    created = request('POST', '/api/v1/jobs', key, {
        'name': 'supervisor-nonroot-gpu-fixture', 'task_id': 'hello_synth',
        'code': CODE, 'timeout': 90, 'resource_type': 'gpu', 'gpu_count': 1,
        'environment': {'EXECUTION_MODE': 'shell'},
    })
    job_id = created['job_id']
    print('submitted GPU fixture:', job_id, flush=True)
    status = 'queued'
    deadline = time.monotonic() + 120
    payload: dict = {}
    try:
        while time.monotonic() < deadline:
            payload = request('GET', '/api/v1/jobs/' + job_id, key)
            status = str(payload.get('status'))
            if status in {'completed', 'failed', 'cancelled'}:
                break
            time.sleep(1)
        if status != 'completed':
            raise RuntimeError('GPU fixture did not complete: ' + status)
    finally:
        if status not in {'completed', 'failed', 'cancelled'}:
            request('DELETE', '/api/v1/jobs/' + job_id, key)
    result = payload.get('result') or {}
    if isinstance(result, str):
        result = json.loads(result)
    assert result.get('source_identity_verified') is True
    assert result.get('worker_cleanup_verified') is True
    assert isinstance(result.get('score'), (int, float))
    probe = None
    for line in result.get('run_log', '').splitlines():
        if line.startswith('GPU_PROBE '):
            probe = json.loads(line.removeprefix('GPU_PROBE '))
    assert probe and probe['uid'] == 65432 and int(probe['cap_eff'], 16) == 0
    evidence = {'job_id': job_id, 'status': status, 'gpu': probe,
                'score': result['score'], 'source_gate': result.get('source_gate'),
                'source_identity_verified': True, 'worker_cleanup_verified': True,
                'model_requests': 0}
    target = ROOT / 'reports/worker-gpu-validation.json'
    target.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
