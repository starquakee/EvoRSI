"""Single-start serial coordinator for the authorized 27 calibration trials."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from research.calibration.collect import validate_trial
from research.calibration.gates import check_review, read_json
from research.calibration.ledger import CalibrationLedger
from research.calibration.plan import analyze
from research.calibration.prepare import historical_integrity
from research.calibration.specs import CalibrationError, GLOBAL_LIMITS, LEDGER, RUNTIME, RunSpec, selected_plan, trials
from research.comparison import deploy, phase
from research.comparison.final_only_stage import exclusive_json
from research.contracts.persist import atomic_write_json
from research.search.acceptance_inner import load_sandbox_api_key


def emit(event, **detail):
    print(json.dumps({'event': event, **detail}, sort_keys=True), flush=True)


def cleanup_interrupted_trial(repo_root, trial_id):
    from research.adapters.sandbox_eval_client import SandboxEvalClient
    path = repo_root / RUNTIME / 'runs' / trial_id / 'live-jobs.json'
    if not path.exists():
        phase.verify_idle(repo_root)
        return {'cleanup_verified': True, 'reason': 'no_submission_record_and_stack_idle'}
    payload = read_json(path)
    key = load_sandbox_api_key(repo_root / deploy.AUTH_FILE)
    client = SandboxEvalClient(endpoint=deploy.GATEWAY_URL, api_key=key, require_cleanup=True)
    results = [client.cancel_job(job_id) for job_id in payload.get('live_jobs', [])]
    unknown = payload.get('submission_intents', []) + payload.get('uncertain_submissions', [])
    phase.verify_idle(repo_root)
    return {'cleanup_verified': not unknown and all(item.get('cleanup_verified') for item in results),
            'cancellations': results, 'unresolved_submission_identities': unknown}


def stop_child(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=30)  # cleanup only, no budget extension
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_child(repo_root, spec_path, trial_id, timeout):
    interpreter = repo_root / 'OpenMLE-Evo/.venv/bin/python'
    if not interpreter.is_file() or timeout <= 0:
        raise CalibrationError('inner_interpreter_or_deadline_unavailable')
    env = dict(os.environ)
    env.pop('LOGGING_DIR', None)
    env['PYTHONPATH'] = str(repo_root)
    env['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
    env['PYTHONUNBUFFERED'] = '1'
    output = repo_root / RUNTIME / 'logs' / (trial_id + '.log')
    output.parent.mkdir(exist_ok=True)
    with output.open('x') as handle:
        process = subprocess.Popen([str(interpreter), '-m', 'research.calibration.runner',
            '--spec', str(spec_path), '--repo-root', str(repo_root)],
            cwd=repo_root, env=env, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            stop_child(process)
            raise


def execute(repo_root):
    manifest = check_review(repo_root)
    phase.verify_idle(repo_root)
    runtime = repo_root / RUNTIME
    roster = trials(repo_root, manifest['source_commit'])
    preflight = read_json(runtime / 'preflight.json')
    if historical_integrity(repo_root) != preflight['historical']:
        raise CalibrationError('historical_artifacts_changed_before_launch')
    if (repo_root / LEDGER).exists():
        raise CalibrationError('calibration_already_started')
    exclusive_json(runtime / 'launch-intent.json', {'schema': 'calibration-launch-intent.v1',
        'manifest_identity': manifest['identity'], 'source_commit': manifest['source_commit'],
        'resume_or_retry': False})
    ledger_args = {'roster': roster, 'manifest_identity': manifest['identity']}
    with CalibrationLedger.create(repo_root / LEDGER, **ledger_args) as ledger:
        ledger.start_clock()
        started_at, deadline = ledger.started_at, ledger.deadline_epoch
    monotonic_started = time.monotonic()
    exclusive_json(runtime / 'launch.json', {'schema': 'calibration-launch.v1',
        'manifest_identity': manifest['identity'], 'started_at': started_at,
        'deadline_epoch': deadline, 'limits': GLOBAL_LIMITS, 'source_commit': manifest['source_commit']})
    emit('started', trials=27, max_requests=216, max_tokens=600000, deadline_epoch=deadline)
    evidence, stopped_reason, current_trial, emergency = [], None, None, None
    try:
        for trial in roster:
            current_trial = trial.trial_id
            remaining = min(deadline - time.time(), GLOBAL_LIMITS['max_elapsed_seconds'] - (time.monotonic() - monotonic_started))
            if remaining <= 0:
                raise CalibrationError('global_deadline_reached')
            check_review(repo_root)
            phase.verify_idle(repo_root, deadline_epoch=deadline)
            spec_path = runtime / 'specs' / (trial.trial_id + '.json')
            exclusive_json(spec_path, RunSpec(trial, manifest['identity']).to_dict())
            emit('trial_started', trial_id=trial.trial_id, position=trial.record['position'] + 1,
                 repeat=trial.record['repeat'] + 1, config_id=trial.record['config_id'])
            remaining = min(deadline - time.time(), GLOBAL_LIMITS['max_elapsed_seconds'] - (time.monotonic() - monotonic_started))
            exit_code = run_child(repo_root, spec_path, trial.trial_id, min(remaining, 1800))
            if exit_code != 0:
                raise CalibrationError('child_exit_' + str(exit_code))
            with CalibrationLedger.open(repo_root / LEDGER, **ledger_args) as ledger:
                snapshot = ledger.snapshot()
            verified = validate_trial(repo_root, manifest, trial, snapshot)
            evidence.append(verified)
            exclusive_json(runtime / 'verified' / (trial.trial_id + '.json'), verified)
            atomic_write_json(runtime / 'progress.json', {'completed_trials': len(evidence),
                'last_trial': trial.trial_id, 'last_validation_accuracy': verified['normalized']['validation_accuracy'],
                'cost': snapshot['global']['committed'], 'stopped': snapshot['stopped'], 'updated_at': time.time()})
            emit('trial_finished', **verified['normalized'], total_cost=snapshot['global']['committed'])
            if snapshot['stopped'] is not None:
                stopped_reason = snapshot['stopped']['reason']
                break
            current_trial = None
    except BaseException as exc:
        stopped_reason = str(exc)[:160] if isinstance(exc, CalibrationError) else type(exc).__name__
        with CalibrationLedger.open(repo_root / LEDGER, **ledger_args) as ledger:
            ledger.stop('coordinator:' + stopped_reason)
        if current_trial is not None:
            try:
                emergency = cleanup_interrupted_trial(repo_root, current_trial)
            except Exception as cleanup_error:
                emergency = {'cleanup_verified': False, 'reason': type(cleanup_error).__name__}
        emit('stopped', reason=stopped_reason, cleanup=emergency)
    finally:
        with CalibrationLedger.open(repo_root / LEDGER, **ledger_args) as ledger:
            snapshot = ledger.snapshot()
        verified_ids = {entry['normalized']['trial_id'] for entry in evidence}
        outcomes = [entry['normalized'] for entry in evidence]
        for trial in roster:
            if trial.trial_id in verified_ids:
                continue
            records = [record for record in snapshot['requests'] if record['run_identity'] == trial.identity()]
            # Cost from every attempted or interrupted trial is included even
            # when its result cannot pass independent verification.
            if records or snapshot['cells'][trial.trial_id]['status'] != 'pending':
                outcomes.append({'trial_id': trial.trial_id, 'status': 'failed',
                    'validation_accuracy': None, 'requests': len(records),
                    'charged_tokens': sum(record['charged_tokens'] for record in records),
                    'reason': stopped_reason or 'unverified_trial'})
        analysis = analyze(selected_plan(), outcomes)
        if analysis['recorded_cost'] != {'requests': snapshot['global']['committed']['requests'],
                                       'charged_tokens': snapshot['global']['committed']['tokens']}:
            raise CalibrationError('final_cost_reconciliation_failed')
        historical_unchanged = historical_integrity(repo_root) == preflight['historical']
        try:
            phase.verify_idle(repo_root)
            idle = True
        except Exception:
            idle = False
        result = {'schema': 'calibration-outcome.v1', 'manifest_identity': manifest['identity'],
            'source_commit': manifest['source_commit'], 'status': 'completed' if len(evidence) == 27 and stopped_reason is None else 'incomplete',
            'started_at': started_at, 'finished_at': time.time(), 'elapsed_seconds': time.time() - started_at,
            'stopped_reason': stopped_reason, 'budget': snapshot,
            'outcomes': outcomes, 'analysis': analysis, 'verified_trials': evidence,
            'historical_artifacts_unchanged': historical_unchanged, 'stack_idle': idle,
            'emergency_cleanup': emergency, 'preflight_jobs': 1, 'final_evaluations': 0}
        if not historical_unchanged or not idle or (emergency is not None and not emergency['cleanup_verified']):
            result['status'] = 'incomplete'
            result['analysis']['verdict'] = 'integrity_or_cleanup_unproven'
        exclusive_json(runtime / 'outcome.json', result)
        emit('finished', status=result['status'], verdict=result['analysis']['verdict'],
             verified_trials=len(evidence), cost=snapshot['global']['committed'],
             elapsed_seconds=result['elapsed_seconds'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    os.umask(0o077)
    result = execute(args.repo_root.resolve())
    return 0 if result['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
