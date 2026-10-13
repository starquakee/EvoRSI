"""Prepare and verify the new search-only stack; zero model calls.

One fixed first-party sandbox probe checks access boundaries. It is explicitly
outside the three pools and 27 continuations and is never a candidate or cached result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from research.shared_start.gates import (
    MANIFEST, OVERRIDE, build_manifest, inspect_stack, legacy_state, read_json,
)
from research.shared_start.specs import CalibrationError, RUNTIME, sha256_file
from research.comparison import deploy, phase
from research.comparison.final_only_stage import exclusive_json, verify_original
from research.comparison.preflight import _check_historic_ledgers
from research.contracts.budget import CancellationToken
from research.search.acceptance_inner import load_sandbox_api_key

PROBE_CODE = '''import csv, os, json, hashlib
own = os.environ["DATA_DIR"]
checks = {}
for name, path in {
    "own_final": "/mnt/rsi_data/comparison/spambase/data/final/test.csv",
    "banknote_final": "/mnt/rsi_data/comparison/banknote/data/final/test.csv",
    "wdbc_final": "/mnt/rsi_data/comparison/wdbc/data/final/test.csv",
    "validation_answers": "/opt/rsi_repo/research/evaluator/tasks/comparison_spambase/validation_answer.csv",
    "other_task": "/mnt/rsi_data/comparison/banknote/data/public/train.csv",
}.items():
    try:
        with open(path, "rb") as handle:
            handle.read(1)
        checks[name] = "UNEXPECTED_READABLE"
    except OSError as exc:
        checks[name] = "denied:" + str(exc.errno)
labels = {}
with open(os.path.join(own, "train.csv"), newline="") as handle:
    for row in csv.DictReader(handle):
        labels[row["label"]] = labels.get(row["label"], 0) + 1
majority = max(labels, key=labels.get)
with open(os.path.join(own, "test.csv"), newline="") as source:
    with open("submission.csv", "w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["id", "label"])
        for row in csv.DictReader(source):
            writer.writerow([row["id"], majority])
for name in ("train.csv", "test.csv"):
    with open(os.path.join(own, name), "rb") as handle:
        checks[name] = hashlib.sha256(handle.read()).hexdigest()
print("SHARED_START_PROBE|" + json.dumps(checks, sort_keys=True))
'''


def historical_integrity(repo_root):
    audit = read_json(repo_root / '.runtime/comparison-supplemental-audit-20261012/audit.json')
    verify_original(repo_root, audit)
    ledgers = _check_historic_ledgers(repo_root)
    stage = repo_root / '.runtime/final-only-amendment-001'
    if not stage.is_dir() or stage.is_symlink():
        raise CalibrationError('historic_final_stage_missing')
    files = {path.relative_to(stage).as_posix(): sha256_file(path)
             for path in sorted(stage.rglob('*')) if path.is_file()}
    if not files:
        raise CalibrationError('historic_final_stage_empty')
    return {'original_comparison_files_verified': len(audit['original_runtime_file_hashes']),
            'original_comparison_hashes_identity': hashlib.sha256(json.dumps(
                audit['original_runtime_file_hashes'], sort_keys=True).encode()).hexdigest(),
            'historic_ledgers': ledgers, 'final_only_stage_files': files,
            'completed_calibration': calibration_integrity(repo_root)}


def calibration_integrity(repo_root):
    """Opaque hashes only; never reopens the completed calibration ledger."""
    stage = repo_root / '.runtime/calibration-round2'
    report = read_json(repo_root / 'reports/calibration-round2-results-20261013/report.json')
    if not stage.is_dir() or stage.is_symlink():
        raise CalibrationError('completed_calibration_missing')
    paths = sorted(stage.rglob('*'))
    if any(path.is_symlink() for path in paths):
        raise CalibrationError('completed_calibration_symlink')
    files = {path.relative_to(stage).as_posix(): sha256_file(path) for path in paths if path.is_file()}
    for key, name in [('ledger', 'budget-ledger.json'), ('outcome', 'outcome.json')]:
        if files.get(name) != report['private_' + key + '_sha256']:
            raise CalibrationError('completed_calibration_hash_changed')
    if not files:
        raise CalibrationError('completed_calibration_empty')
    return {'files': files, 'file_count': len(files)}


def run_probe(repo_root, manifest):
    from research.adapters.sandbox_eval_client import SandboxEvalClient
    from research.search.sandbox_task import SandboxAcceptanceTask
    root = repo_root / RUNTIME
    exclusive_json(root / 'probe-intent.json', {'model_requests': 0,
        'max_new_jobs': 1, 'candidate_source_sha256': hashlib.sha256(PROBE_CODE.encode()).hexdigest()})
    key = load_sandbox_api_key(repo_root / deploy.AUTH_FILE)
    client = SandboxEvalClient(endpoint=deploy.GATEWAY_URL, api_key=key, require_cleanup=True)
    token = CancellationToken(deadline_epoch=time.time() + 180)
    try:
        result = client.submit_and_wait(PROBE_CODE, name='shared-start-pilot-isolation-probe',
            data_dir=manifest['dataset']['worker_data_public_path'], task_id='comparison_spambase',
            resource_type='cpu', timeout=120, cancellation_token=token)
        task = SandboxAcceptanceTask(client, token,
            data_dir=manifest['dataset']['worker_data_public_path'], task_id='comparison_spambase',
            resource_type='cpu', job_timeout_seconds=120)
        task._translate(result, PROBE_CODE, 0.0)
        job = task.jobs[-1]
        body = result.raw.get('result') or {}
        if isinstance(body, str):
            body = json.loads(body)
        lines = [line for line in (body.get('run_log') or '').splitlines()
                 if line.startswith('SHARED_START_PROBE|')]
        if len(lines) != 1 or job.get('evidence_verified') is not True:
            raise CalibrationError('probe_trusted_result_missing')
        checks = json.loads(lines[0].split('|', 1)[1])
        denied = ('own_final', 'banknote_final', 'wdbc_final', 'validation_answers', 'other_task')
        if any(not str(checks.get(name)).startswith('denied:') for name in denied):
            raise CalibrationError('probe_private_path_readable')
        if checks.get('train.csv') != manifest['dataset']['train_csv_sha256'] or checks.get('test.csv') != manifest['dataset']['validation_features_sha256']:
            raise CalibrationError('probe_public_data_hash_mismatch')
        if client.live_job_ids() or client.uncertain_submission_ids():
            raise CalibrationError('probe_cleanup_unproven')
        evidence = {'schema': 'shared-start-isolation-probe.v1', 'model_requests': 0,
                    'jobs': 1, 'checks': checks, 'job': job,
                    'source_sha256': hashlib.sha256(PROBE_CODE.encode()).hexdigest()}
        exclusive_json(root / 'probe-result.json', evidence)
        return evidence
    finally:
        cleanup = client.cancel_all_live()
        exclusive_json(root / 'probe-cleanup.json', {'cancellation': cleanup,
            'live_jobs': sorted(client.live_job_ids()),
            'uncertain_submissions': sorted(client.uncertain_submission_ids())})


def prepare(repo_root):
    manifest = build_manifest(repo_root)
    root = repo_root / RUNTIME
    root.mkdir(mode=0o700, exist_ok=False)
    exclusive_json(repo_root / MANIFEST, manifest)
    before_legacy = legacy_state()
    if any(value['status'] != 'running' for value in before_legacy.values()):
        raise CalibrationError('legacy_container_not_running')
    before_history = historical_integrity(repo_root)
    exclusive_json(root / 'preparation-baseline.json', {'legacy': before_legacy, 'historical': before_history})
    phase.verify_idle(repo_root)
    for task in ('banknote', 'wdbc', 'spambase'):
        (root / 'empty-final' / task / 'data/final').mkdir(parents=True)
    # Exact tracked compose overlay; no pull, build, dependency recreation,
    # old-stack operation, volume reset, or original-publication mutation.
    command = ['docker', 'compose', '-f', str(repo_root / phase.COMPOSE_FILE),
        '-f', str(repo_root / OVERRIDE), 'up', '-d', '--no-deps', '--pull', 'never', 'worker', 'dispatcher']
    proc = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=120)
    if proc.returncode:
        raise CalibrationError('shared_start_stack_recreate_failed')
    ready_deadline = time.monotonic() + 120
    while True:
        try:
            stack = inspect_stack(repo_root, manifest['stack_image_ids'])
            phase.verify_idle(repo_root)
            break
        except (CalibrationError, phase.PhaseActivationError):
            if time.monotonic() >= ready_deadline:
                raise
            time.sleep(2)
    probe = run_probe(repo_root, manifest)
    phase.verify_idle(repo_root)
    after_history = historical_integrity(repo_root)
    if after_history != before_history or legacy_state() != before_legacy:
        raise CalibrationError('historical_state_changed')
    stack = inspect_stack(repo_root, manifest['stack_image_ids'])
    evidence = {'schema': 'shared-start-preflight.v1', 'verdict': 'PASS',
                'manifest_identity': manifest['identity'], 'stack': stack,
                'legacy': before_legacy, 'historical': after_history,
                'probe_result_sha256': sha256_file(root / 'probe-result.json'),
                'probe_job_id': probe['job']['job_id'],
                'new_model_requests': 0, 'new_fixed_preflight_jobs': 1}
    exclusive_json(root / 'preflight.json', evidence)
    return {'verdict': 'PASS', 'manifest_identity': manifest['identity'],
            'new_model_requests': 0, 'fixed_probe_jobs': 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(prepare(args.repo_root.resolve())))


if __name__ == '__main__':
    main()
