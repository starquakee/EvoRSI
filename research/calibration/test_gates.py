import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.calibration import gates
from research.calibration.specs import CalibrationError, RUNTIME, protocol, sha256_file
from research.comparison.preflight import EXPECTED_STACK_IMAGES, _expected_mount_contract


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    manifest = {'identity': 'f' * 64, 'source_commit': 'a' * 40, 'stack_image_ids': {}}
    monkeypatch.setattr(gates, 'build_manifest', lambda root: copy.deepcopy(manifest))
    write(tmp_path / gates.MANIFEST, manifest)
    write(tmp_path / gates.AUTHORIZATION, {'execution_authorized': True, 'repeats': 3,
          'protocol': protocol(), 'manifest_identity': manifest['identity']})
    preflight = tmp_path / RUNTIME / 'preflight.json'
    write(preflight, {'verdict': 'PASS', 'manifest_identity': manifest['identity'], 'stack': {}, 'legacy': {}})
    write(tmp_path / gates.REVIEW, {'accepted': True, 'source_commit': manifest['source_commit'],
          'manifest_identity': manifest['identity'],
          'authorization_sha256': sha256_file(tmp_path / gates.AUTHORIZATION),
          'preflight_sha256': sha256_file(preflight)})
    return tmp_path, manifest


def test_review_matches_exact_new_manifest_and_three_repeat_authorization(reviewed):
    root, manifest = reviewed
    assert gates.check_review(root, check_stack=False) == manifest


@pytest.mark.parametrize('target,field,value', [
    (gates.REVIEW, 'accepted', False), (gates.REVIEW, 'source_commit', 'b' * 40),
    (gates.AUTHORIZATION, 'repeats', 5), (gates.AUTHORIZATION, 'execution_authorized', False),
    (gates.MANIFEST, 'identity', 'c' * 64),
    (RUNTIME / 'preflight.json', 'verdict', 'FAIL'),
])
def test_live_gate_refuses_altered_receipt_before_runtime_activity(reviewed, target, field, value):
    root, _ = reviewed
    payload = gates.read_json(root / target)
    payload[field] = value
    write(root / target, payload)
    with pytest.raises(CalibrationError):
        gates.check_review(root, check_stack=False)


def test_runner_missing_review_cannot_read_credentials_or_open_ledger(tmp_path, monkeypatch):
    from research.calibration import runner
    def forbidden(*args, **kwargs):
        pytest.fail('credential or ledger boundary crossed before review')
    monkeypatch.setattr(runner, 'load_sandbox_api_key', forbidden)
    monkeypatch.setattr(runner.CalibrationLedger, 'open', forbidden)
    with pytest.raises(CalibrationError, match='supervisor-review'):
        runner.run_inner(tmp_path / 'nonexistent-spec.json', repo_root=tmp_path)
    assert not (tmp_path / RUNTIME).exists()


@pytest.fixture
def fake_stack(tmp_path, monkeypatch):
    root = tmp_path
    empty = root / RUNTIME / 'empty-final'
    for task in ('banknote', 'wdbc', 'spambase'):
        (empty / task / 'data/final').mkdir(parents=True)
    registry = root / 'research/evaluator/registry.v1.json'
    write(registry, {})
    mounts = _expected_mount_contract(root)
    for task in ('banknote', 'wdbc', 'spambase'):
        mounts['rsi_worker'][f'/mnt/rsi_data/comparison/{task}/data/final'] = (str(empty / task / 'data/final'), False)
    mounts['rsi_dispatcher']['/opt/rsi_final'] = (str(empty), False)
    payloads, images = {}, {}
    for name, tag in EXPECTED_STACK_IMAGES.items():
        image = 'sha256:' + name
        images[name.removeprefix('rsi_')] = image
        payloads[name] = {'Id': name + '-id', 'Image': image,
            'State': {'Running': True, 'StartedAt': 'start', 'Health': {'Status': 'healthy'}},
            'Config': {'Image': tag, 'Env': ['EVALUATOR_REGISTRY_PATH=' + gates.SEARCH_REGISTRY]},
            'Mounts': [{'Destination': target, 'Source': source, 'RW': writable}
                       for target, (source, writable) in mounts.get(name, {}).items()],
            'NetworkSettings': {'Ports': {'80/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '6581'}]} if name == 'rsi_nginx' else {},
                                'Networks': {'rsi-trustworthy_execution': {'Gateway': ''}}},
            'HostConfig': {'ReadonlyRootfs': True, 'CapDrop': ['ALL'],
                           'CapAdd': ['CAP_KILL', 'CAP_SETUID', 'CAP_SETGID'],
                           'SecurityOpt': ['no-new-privileges:true'], 'Privileged': False}}
    monkeypatch.setattr(gates.subprocess, 'run', lambda command, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps([payloads[command[-1]]])))
    monkeypatch.setattr(gates.subprocess, 'check_output', lambda *args, **kwargs: sha256_file(registry) + '  registry')
    return root, images, payloads


def test_empty_final_mounts_and_reviewed_images_pass(fake_stack):
    root, images, _ = fake_stack
    assert gates.inspect_stack(root, images)['phase'] == 'search'


@pytest.mark.parametrize('change', ['image', 'mount', 'final_phase', 'port', 'privilege', 'gateway', 'file'])
def test_stack_contract_rejects_exposure_or_source_drift(fake_stack, change):
    root, images, payloads = fake_stack
    if change == 'image':
        payloads['rsi_worker']['Image'] += '-different'
    elif change == 'mount':
        mount = next(m for m in payloads['rsi_worker']['Mounts'] if m['Destination'].endswith('spambase/data/final'))
        mount['Source'] = str(root / '.runtime/comparison-istratde/final-publication/spambase/data/final')
    elif change == 'final_phase':
        payloads['rsi_dispatcher']['Config']['Env'] = ['EVALUATOR_REGISTRY_PATH=/opt/rsi_final/registry.final.v1.json']
    elif change == 'port':
        payloads['rsi_worker']['NetworkSettings']['Ports'] = {'8080/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '8080'}]}
    elif change == 'privilege':
        payloads['rsi_worker']['HostConfig']['Privileged'] = True
    elif change == 'gateway':
        payloads['rsi_worker']['NetworkSettings']['Networks']['rsi-trustworthy_execution']['Gateway'] = '172.1.1.1'
    else:
        (root / RUNTIME / 'empty-final/spambase/data/final/test.csv').write_text('unexpected')
    with pytest.raises(CalibrationError):
        gates.inspect_stack(root, images)
