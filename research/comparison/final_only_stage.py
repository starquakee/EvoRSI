"""One explicitly authorized final-only amendment; no model generation or resume.

Original search/ledger/final artifacts are read-only. Durable exclusive intents
in a separate stage prevent duplicate submissions, including after interruption.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

STAGE = Path('.runtime/final-only-amendment-001')
ORIGINAL = Path('.runtime/comparison-istratde')
AUDIT = Path('.runtime/comparison-supplemental-audit-20261012/audit.json')
PROPOSAL = Path('reports/istratde-supplemental-audit-20261012/final-only-stage-proposal.json')


def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'unsafe_or_missing_file:{path.name}')
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path: Path) -> Any:
    digest(path)
    return json.loads(path.read_text())


def exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, sort_keys=True, separators=(',', ':'))
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class StageJournal:
    def __init__(self, stage: Path, roster: list[str], *, now: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        if len(roster) != 17 or len(set(roster)) != 17 or stage.is_symlink():
            raise ValueError('invalid_stage_roster')
        self.stage, self.roster, self.now, self.monotonic = stage, set(roster), now, monotonic
        self.started, self.monotonic_start = now(), monotonic()
        self.deadline = self.started + 14400
        self.sequence = 0
        self.reserved: set[str] = set()
        self.stopped = False
        exclusive_json(stage/'launch.json', {'schema':'final-only-stage-launch.v1',
            'started_at':self.started, 'deadline_epoch':self.deadline, 'max_elapsed_seconds':14400,
            'max_new_jobs':17, 'max_model_requests':0, 'retry_or_resume':False, 'roster':roster})
        self.event('started', {'deadline_epoch':self.deadline})

    def event(self, kind: str, detail: Any) -> None:
        exclusive_json(self.stage/'events'/f'{self.sequence:04d}.json',
                       {'sequence':self.sequence,'at':self.now(),'type':kind,'detail':detail})
        self.sequence += 1

    def remaining(self) -> float:
        return min(self.deadline-self.now(),14400-(self.monotonic()-self.monotonic_start))

    def reserve(self, key: str, intent: dict[str, Any]) -> None:
        if self.stopped or self.remaining() < 1:
            raise RuntimeError('stage_stopped_or_deadline')
        if key not in self.roster or key in self.reserved or len(self.reserved) >= 17:
            raise RuntimeError('duplicate_or_unapproved_submission')
        directory=self.stage/'final'/key.replace(':','-')
        exclusive_json(directory/'intent.json', intent)
        self.reserved.add(key)
        self.event('submission_intent', {'cell_key':key,'intent_sha256':digest(directory/'intent.json')})

    def stop(self, reason: str) -> None:
        if not self.stopped:
            self.stopped=True
            self.event('stopped', {'reason':reason})


def verify_original(root: Path, audit: dict[str, Any]) -> None:
    original=root/ORIGINAL
    actual={p.relative_to(original).as_posix() for p in original.rglob('*') if p.is_file()}
    if actual != set(audit['original_runtime_file_hashes']):
        raise ValueError('original_artifact_roster_changed')
    for relative, expected in audit['original_runtime_file_hashes'].items():
        if digest(original/relative) != expected:
            raise ValueError('original_artifact_changed:'+relative)


def run(root: Path) -> dict[str, Any]:
    from research.comparison import deploy, final
    from research.comparison.data import load_task_bindings
    from research.comparison.preflight import inspect_runtime_stack
    from research.comparison.specs import iter_cells
    from research.search.acceptance_inner import load_sandbox_api_key
    import httpx

    root=root.resolve()
    stage=root/STAGE
    authorization=read(stage/'authorization.json')
    proposal=read(root/PROPOSAL)
    audit=read(root/AUDIT)
    if authorization.get('execution_authorized') is not True:
        raise ValueError('explicit_stage_authorization_required')
    if authorization.get('proposal_sha256') != digest(root/PROPOSAL) or authorization.get('audit_sha256') != digest(root/AUDIT):
        raise ValueError('authorization_input_mismatch')
    if authorization.get('stage_path') != STAGE.as_posix():
        raise ValueError('authorization_stage_mismatch')
    if audit['supplemental_verdict'] != 'QUALIFIED_REUSABLE_SEARCH_AND_TEN_PREDICTIONS':
        raise ValueError('supplemental_audit_not_accepted')
    expected_limits={'max_new_model_requests':0,'max_new_model_tokens':0,'max_new_final_job_submissions':17,
                     'max_elapsed_seconds':14400,'per_job_timeout_seconds':600,'retry_or_resubmit':False}
    if any(proposal.get(k) != v for k,v in expected_limits.items()):
        raise ValueError('stage_limits_not_approved')
    def check_head() -> None:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
        if head != authorization['runner_commit'] or dirty:
            raise ValueError('stage_source_changed')
    check_head()
    verify_original(root,audit)
    manifest=read(root/ORIGINAL/'experiment-manifest.v1.json')
    for rel,expected in manifest['pinned_file_sha256'].items():
        if digest(root/rel)!=expected: raise ValueError('frozen_pin_changed:'+rel)
    publication=read(root/ORIGINAL/'final/publication.json')
    stack=inspect_runtime_stack(root,expected_images=manifest['stack_images']['image_ids'])
    if stack['phase'] != 'final': raise ValueError('stack_not_in_final_phase')
    key=load_sandbox_api_key(root/deploy.AUTH_FILE)
    for status in ('queued','running','submitted'):
        response=httpx.get(deploy.GATEWAY_URL+'/api/v1/jobs',params={'status':status,'limit':1},
                           headers={'x-api-key':key},timeout=20)
        response.raise_for_status()
        if response.json()['count'] != 0: raise ValueError('new_stack_not_idle:'+status)
    for rel,expected in publication['files'].items():
        if digest(Path(publication['publication_root'])/rel)!=expected:
            raise ValueError('publication_changed:'+rel)
    roster=proposal['remaining_cells']
    existing={r['cell_key'] for r in audit['final_predictions']}
    cells={c.cell_key():c for c in iter_cells(load_task_bindings(),proposal['original_source_commit'])}
    if len(existing)!=10 or set(cells)-existing != {r['cell_key'] for r in roster}:
        raise ValueError('remaining_roster_mismatch')
    journal=StageJournal(stage,[r['cell_key'] for r in roster])
    exclusive_json(stage/'preflight.json', {'stack':stack,'original_files_verified':len(audit['original_runtime_file_hashes']),
        'authorization_sha256':digest(stage/'authorization.json'),'runner_commit':authorization['runner_commit']})
    executor=final.sandbox_final_executor(root,publication,state_dir=stage/'executor',timeout_seconds=600,
                                         deadline_fn=lambda:time.time()+max(0,journal.remaining()))
    results=[]
    def terminated(signum: int, frame: Any) -> None:
        raise RuntimeError('outer_stage_timeout_or_termination')
    signal.signal(signal.SIGTERM,terminated)
    try:
        for item in roster:
            check_head()
            if digest(root/ORIGINAL/'comparison-ledger.json') != proposal['original_ledger_sha256']:
                raise ValueError('original_ledger_changed')
            cell_key=item['cell_key']; cell=cells[cell_key]
            source=root/ORIGINAL/'cells'/cell_key.replace(':','-')/'winner.py'
            if digest(source)!=item['source_sha256']: raise ValueError('winner_changed')
            source_bytes=source.read_bytes()
            answer=root/'research/comparison/sealed'/cell.task_id/'final_labels.csv'
            if digest(answer)!=cell.task_binding.final_labels_sha256: raise ValueError('answer_changed')
            for rel,expected in publication['files'].items():
                if digest(Path(publication['publication_root'])/rel)!=expected: raise ValueError('publication_changed')
            timeout=min(600,journal.remaining())
            if timeout<1: raise RuntimeError('stage_deadline')
            state=read(root/ORIGINAL/'cells'/cell_key.replace(':','-')/'cell.json')
            winner=state['winner']
            intent={'schema':'final-only-amendment-intent.v1','cell_key':cell_key,
                'winner':winner,'original_source_commit':proposal['original_source_commit'],
                'source_sha256':item['source_sha256'],'answer_sha256':cell.task_binding.final_labels_sha256,
                'submitted_at':time.time(),'timeout_seconds':timeout,'stage_deadline':journal.deadline}
            journal.reserve(cell_key,intent)
            execution=executor.execute(task_id=final.final_task_id(cell.task_id),source_bytes=source_bytes,
                cell_key=cell_key,timeout_seconds=timeout)
            if journal.remaining()<=0: raise RuntimeError('stage_deadline')
            result=final._translate_execution(SimpleNamespace(runtime_dir=stage),journal,cell,winner,execution,
                time.time,root/'research/comparison/sealed',
                intent_sha256=digest(stage/'final'/cell_key.replace(':','-')/'intent.json'),
                barrier_identity=publication['barrier_identity'],answer_sha256=cell.task_binding.final_labels_sha256,
                registry_version=publication['registry_version'],winner_bytes=source_bytes)
            # Explicit amendment provenance; this is never written into the original runtime.
            result['schema']='final-only-amendment-result.v1'
            result['original_canonical_audit']='FAIL'
            result['audit_qualification']='exact recorded clock exception; supplemental audit'
            exclusive_json(stage/'final'/cell_key.replace(':','-')/'final.json',result)
            results.append({'cell_key':cell_key,'status':result['status'],'raw_score':result['raw_score'],
                            'job_id':result.get('job_id')})
            journal.event('result',results[-1])
            print(json.dumps({'completed':len(results),'total':17,**results[-1]}),flush=True)
        verify_original(root,audit)
        check_head()
        journal.event('completed',{'results':len(results),'new_model_requests':0})
        outcome={'status':'completed','results':results,'new_model_requests':0,
                 'original_files_unchanged':True,'finished_at':time.time(),'elapsed_monotonic':time.monotonic()-journal.monotonic_start}
    except BaseException as exc:
        journal.stop(type(exc).__name__+':'+str(exc))
        outcome={'status':'stopped','reason':type(exc).__name__+':'+str(exc),'results':results,
                 'new_model_requests':0,'finished_at':time.time(),'elapsed_monotonic':time.monotonic()-journal.monotonic_start}
    exclusive_json(stage/'outcome.json',outcome)
    return outcome


if __name__ == '__main__':
    result=run(Path(__file__).resolve().parents[2])
    print(json.dumps({'stage_status':result['status'],'results':len(result['results'])}),flush=True)
    raise SystemExit(0 if result['status']=='completed' else 2)
