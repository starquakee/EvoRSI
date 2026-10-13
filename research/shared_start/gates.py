"""Shared-start-specific manifest and launch gates; no old review is reused."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from research.calibration.plan import identity
from research.shared_start.specs import (
    CalibrationError, RUNTIME, protocol, selected_plan, sha256_file, trials,
)
from research.comparison import deploy
from research.comparison.preflight import EXPECTED_STACK_IMAGES, _expected_mount_contract
from research.comparison.runner import verify_task_data_identity
from research.contracts.source_gate import load_policy
from research.evaluator.service import EvaluatorRegistry

MANIFEST = RUNTIME / 'manifest.json'
REVIEW = RUNTIME / 'supervisor-review.json'
AUTHORIZATION = RUNTIME / 'authorization.json'
OFFLINE = Path('.runtime/shared-start-offline-validation/report.json')
SEARCH_REGISTRY = '/opt/rsi_repo/research/evaluator/registry.v1.json'
OVERRIDE = Path('deploy/rsi-trustworthy/docker-compose.shared-start.yaml')
OLD_IMAGE_MANIFEST = Path('reports/istratde-amended-results-20261012/frozen/experiment-manifest.v1.json')
LEGACY = ('sandbox_api', 'sandbox_dispatcher', 'sandbox_nginx', 'sandbox_postgres',
          'sandbox_redis', 'ml-sandbox-gpu-0')


def read_json(path):
    if path.is_symlink() or not path.is_file():
        raise CalibrationError('missing_or_symlinked:' + path.name)
    return json.loads(path.read_text())


def validate_paths(repo_root):
    if repo_root != repo_root.resolve() or repo_root.is_symlink():
        raise CalibrationError('repo_path_unsafe')
    path = repo_root
    for part in RUNTIME.parts:
        path = path / part
        if path.is_symlink():
            raise CalibrationError('runtime_path_symlink')


def source_commit(repo_root):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_root, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo_root, text=True).strip():
        raise CalibrationError('source_not_clean')
    return head


def build_manifest(repo_root):
    validate_paths(repo_root)
    commit = source_commit(repo_root)
    roster = trials(repo_root, commit)
    registry_path = repo_root / 'research/evaluator/registry.v1.json'
    registry = EvaluatorRegistry.load(registry_path)
    data = verify_task_data_identity(roster[0].task_binding, repo_root=repo_root, registry=registry)
    images = read_json(repo_root / OLD_IMAGE_MANIFEST)['stack_images']['image_ids']
    policy = load_policy()
    payload = {'schema': 'shared-start-manifest.v1', 'source_commit': commit,
               'protocol': protocol(), 'plan_identity': selected_plan()['plan_identity'],
               'trials': [trial.to_dict() for trial in roster], 'dataset': data,
               'registry_sha256': sha256_file(registry_path), 'stack_image_ids': images,
               'compose_sha256': sha256_file(repo_root / 'deploy/rsi-trustworthy/docker-compose.yaml'),
               'compose_override_sha256': sha256_file(repo_root / OVERRIDE),
               'policy_version': policy.policy_version, 'policy_sha256': policy.policy_sha256,
               'runtime_dir': RUNTIME.as_posix()}
    return {**payload, 'identity': identity(payload)}


def legacy_state():
    result = {}
    for name in LEGACY:
        info = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True, timeout=30))[0]
        result[name] = {'id': info['Id'], 'status': info['State']['Status'],
                        'started_at': info['State']['StartedAt']}
    return result


def inspect_stack(repo_root, expected_images):
    """Check exact mount sources and image IDs, with empty final mounts.

    No call changes a container, copies a credential or reads final CSV bytes.
    Full docker inspect responses (which include private env) stay in memory;
    only allowlisted identities are returned.
    """
    mounts = _expected_mount_contract(repo_root)
    empty = repo_root / RUNTIME / 'empty-final'
    if empty.is_symlink() or not empty.is_dir():
        raise CalibrationError('empty_final_root_missing')
    for task in ('banknote', 'wdbc', 'spambase'):
        leaf = empty / task / 'data/final'
        if not leaf.is_dir() or any(leaf.iterdir()):
            raise CalibrationError('final_leaf_not_empty')
        mounts['rsi_worker'][f'/mnt/rsi_data/comparison/{task}/data/final'] = (str(leaf), False)
    if any(path.is_symlink() or path.is_file() for path in empty.rglob('*')):
        raise CalibrationError('empty_final_tree_contains_file_or_symlink')
    mounts['rsi_dispatcher']['/opt/rsi_final'] = (str(empty), False)
    result = {'phase': 'search', 'containers': {}}
    for name, tag in EXPECTED_STACK_IMAGES.items():
        proc = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=30)
        if proc.returncode:
            raise CalibrationError('container_unavailable:' + name)
        info = json.loads(proc.stdout)[0]
        state, config = info['State'], info['Config']
        if not state['Running'] or state.get('Health', {}).get('Status', 'healthy') != 'healthy':
            raise CalibrationError('container_not_ready:' + name)
        if config['Image'] != tag or info['Image'] != expected_images.get(name.removeprefix('rsi_')):
            raise CalibrationError('container_image_changed:' + name)
        if name in mounts:
            actual = {entry['Destination']: (entry['Source'], bool(entry['RW'])) for entry in info['Mounts']}
            if actual != mounts[name]:
                raise CalibrationError('container_mounts_changed:' + name)
        if name == 'rsi_dispatcher':
            env = dict(entry.split('=', 1) for entry in config['Env'] if '=' in entry)
            if env.get('EVALUATOR_REGISTRY_PATH') != SEARCH_REGISTRY:
                raise CalibrationError('dispatcher_not_search_phase')
        ports = info['NetworkSettings'].get('Ports') or {}
        published = [entry for entries in ports.values() if entries for entry in entries]
        if name == 'rsi_nginx':
            if published != [{'HostIp': '127.0.0.1', 'HostPort': '6581'}]:
                raise CalibrationError('gateway_bind_changed')
        elif published:
            raise CalibrationError('unexpected_published_port:' + name)
        if name == 'rsi_worker':
            host = info['HostConfig']
            if not host.get('ReadonlyRootfs') or set(host.get('CapDrop') or []) != {'ALL'}:
                raise CalibrationError('worker_privileges_changed')
            if {value.removeprefix('CAP_') for value in host.get('CapAdd') or []} != {'SETUID', 'SETGID', 'KILL'} or not any(value.startswith('no-new-privileges') for value in host.get('SecurityOpt') or []):
                raise CalibrationError('worker_privileges_changed')
            if host.get('Privileged') or any('unconfined' in value for value in host.get('SecurityOpt') or []):
                raise CalibrationError('worker_privileges_changed')
            if set(info['NetworkSettings']['Networks']) != {'rsi-trustworthy_execution'}:
                raise CalibrationError('worker_network_changed')
            if info['NetworkSettings']['Networks']['rsi-trustworthy_execution'].get('Gateway'):
                raise CalibrationError('worker_host_gateway_present')
        result['containers'][name] = {'id': info['Id'], 'image_id': info['Image'],
                                      'started_at': state['StartedAt']}
    registry_hash = subprocess.check_output(['docker', 'exec', 'rsi_dispatcher', 'sha256sum', SEARCH_REGISTRY], text=True, timeout=30).split()[0]
    if registry_hash != sha256_file(repo_root / 'research/evaluator/registry.v1.json'):
        raise CalibrationError('container_registry_hash_mismatch')
    result['registry_sha256'] = registry_hash
    return result


def check_review(repo_root, *, check_stack=True):
    """Refuse before ledger/model activity unless exact new review is present."""
    validate_paths(repo_root)
    review = read_json(repo_root / REVIEW)
    manifest = read_json(repo_root / MANIFEST)
    if manifest != build_manifest(repo_root):
        raise CalibrationError('manifest_changed')
    authorization = read_json(repo_root / AUTHORIZATION)
    if authorization.get('scope') != 'shared-start-pilot' or authorization.get('execution_authorized') is not True or authorization.get('repeats') != 3 or authorization.get('protocol') != protocol() or authorization.get('manifest_identity') != manifest['identity']:
        raise CalibrationError('authorization_not_bound')
    if review.get('accepted') is not True or review.get('source_commit') != manifest['source_commit'] or review.get('manifest_identity') != manifest['identity'] or review.get('authorization_sha256') != sha256_file(repo_root / AUTHORIZATION):
        raise CalibrationError('review_not_bound')
    offline = read_json(repo_root / OFFLINE)
    if (review.get('offline_checks_sha256') != sha256_file(repo_root / OFFLINE)
            or offline.get('source_commit') != manifest['source_commit']
            or offline.get('verdict') != 'PASS' or not offline.get('checks')
            or any(check.get('returncode') != 0 for check in offline['checks'])
            or offline.get('model_requests') != 0 or offline.get('sandbox_jobs') != 0):
        raise CalibrationError('offline_validation_not_bound')
    preflight_path = repo_root / RUNTIME / 'preflight.json'
    preflight = read_json(preflight_path)
    if review.get('preflight_sha256') != sha256_file(preflight_path) or preflight.get('verdict') != 'PASS' or preflight.get('manifest_identity') != manifest['identity']:
        raise CalibrationError('preflight_not_bound')
    if check_stack:
        if inspect_stack(repo_root, manifest['stack_image_ids']) != preflight['stack']:
            raise CalibrationError('running_stack_changed_since_preflight')
        if legacy_state() != preflight['legacy']:
            raise CalibrationError('legacy_stack_changed_since_preflight')
    return manifest
