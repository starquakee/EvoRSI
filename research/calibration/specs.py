"""Approved three-repeat calibration contract; independent of the old roster."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from research.calibration.plan import identity, make_plan
from research.comparison.data import load_task_bindings
from research.comparison.protocol import ModelProfile
from research.comparison.specs import TaskBinding

RUNTIME = Path('.runtime/calibration-round2')
LEDGER = RUNTIME / 'budget-ledger.json'
SCHEMA = 'spambase-repeatability-calibration.v1'
GLOBAL_LIMITS = {'max_requests': 216, 'max_tokens': 600000, 'max_elapsed_seconds': 14400}
TRIAL_LIMITS = {'max_requests': 8, 'max_tokens': 40000, 'max_elapsed_seconds': 1800,
                'max_main_requests': 6, 'max_debug_requests': 2}
JOB_TIMEOUT = 600


class CalibrationError(RuntimeError):
    """A calibration contract or evidence gate refused."""


def selected_plan() -> dict[str, Any]:
    folder = Path(__file__).parent
    protocol = json.loads((folder / 'protocol-draft.v1.json').read_text())
    actual = json.loads((folder / 'plan-small.v1.json').read_text())
    expected = make_plan(protocol, 'small')
    if actual != expected or actual['budget'] != {
        'repeats': 3, 'trials': 27, **GLOBAL_LIMITS,
    }:
        raise CalibrationError('selected_plan_mismatch')
    pairs = [(c['decoded']['crossover_prob'], c['decoded']['score_weight'])
             for c in actual['configurations']]
    if pairs != [(x, y) for x in (0.05, 0.5, 0.95) for y in (0.0, 1.0, 2.0)]:
        raise CalibrationError('grid_mismatch')
    if any({k: c['decoded'][k] for k in ('delta_weight', 'novelty_weight', 'max_debug_depth')}
           != {'delta_weight': 0.4, 'novelty_weight': 0.25, 'max_debug_depth': 1}
           for c in actual['configurations']):
        raise CalibrationError('fixed_parameters_mismatch')
    return actual


def protocol() -> dict[str, Any]:
    return {'schema': SCHEMA, 'task': 'spambase', 'repeats': 3, 'trials': 27,
            'plan_identity': selected_plan()['plan_identity'],
            'global_limits': GLOBAL_LIMITS, 'trial_limits': TRIAL_LIMITS,
            'job_timeout_seconds': JOB_TIMEOUT, 'model_profile': ModelProfile().to_dict(),
            'data_phase': 'validation', 'new_final_scoring': False,
            'retry_or_resume': False, 'budget_extension': False}


def protocol_identity() -> str:
    return identity(protocol())


def require_commit(value: str) -> None:
    if not isinstance(value, str) or len(value) != 40 or set(value) - set('0123456789abcdef'):
        raise CalibrationError('source_commit_invalid')


@dataclass(frozen=True)
class Trial:
    trial_id: str
    task_binding: TaskBinding
    source_commit: str

    def __post_init__(self):
        require_commit(self.source_commit)
        if not isinstance(self.task_binding, TaskBinding) or self.task_binding.task_id != 'spambase':
            raise CalibrationError('task_binding_invalid')
        self.record  # reject unknown schedule entries

    @property
    def record(self) -> dict[str, Any]:
        found = [trial for trial in selected_plan()['schedule'] if trial['trial_id'] == self.trial_id]
        if len(found) != 1:
            raise CalibrationError('unknown_trial')
        return found[0]

    @property
    def cell(self) -> 'Trial':
        # Structural adapter for existing low-level budget controls, not a
        # comparison CellSpec and never admitted into its frozen old ledger.
        return self

    @property
    def task(self) -> str:
        return 'spambase'

    @property
    def decoded(self) -> dict[str, Any]:
        return dict(self.record['decoded'])

    @property
    def inner_seed(self) -> int:
        return self.record['inner_seed']

    @property
    def config_index(self) -> int:
        return int(self.record['config_id'][1:])

    def cell_key(self) -> str:
        return self.trial_id

    def run_key(self) -> str:
        return self.trial_id

    def to_dict(self) -> dict[str, Any]:
        return {'schema': SCHEMA, 'trial': self.record,
                'task_binding': self.task_binding.to_dict(),
                'source_commit': self.source_commit, 'protocol_identity': protocol_identity()}

    def identity(self) -> str:
        return identity(self.to_dict())


def trials(repo_root: Path, source_commit: str) -> tuple[Trial, ...]:
    # load_task_bindings reads the tracked manifest only, never final bytes.
    bindings = load_task_bindings(repo_root / 'research/comparison/data-manifest.v1.json')
    return tuple(Trial(item['trial_id'], bindings['spambase'], source_commit)
                 for item in selected_plan()['schedule'])


@dataclass(frozen=True)
class RunSpec:
    run: Trial
    manifest_identity: str
    resource_type: str = 'cpu'
    job_timeout_seconds: int = JOB_TIMEOUT

    @property
    def decoded(self):
        return self.run.decoded

    @property
    def runner_commit(self):
        return self.run.source_commit

    @property
    def run_dir(self):
        return str(RUNTIME / 'runs' / self.run.trial_id)

    @property
    def ledger_path(self):
        return str(LEDGER)

    def validate(self):
        if not isinstance(self.run, Trial):
            raise CalibrationError('trial_type_invalid')
        self.run.__post_init__()
        if not isinstance(self.manifest_identity, str) or len(self.manifest_identity) != 64 or set(self.manifest_identity) - set('0123456789abcdef'):
            raise CalibrationError('manifest_identity_invalid')
        if self.resource_type != 'cpu' or type(self.job_timeout_seconds) is not int or self.job_timeout_seconds != JOB_TIMEOUT:
            raise CalibrationError('execution_parameters_invalid')

    def to_dict(self):
        self.validate()
        return {'schema': 'calibration-run-spec.v1', 'trial_id': self.run.trial_id,
                'run_identity': self.run.identity(), 'source_commit': self.runner_commit,
                'manifest_identity': self.manifest_identity,
                'run_dir': self.run_dir, 'ledger_path': self.ledger_path,
                'decoded': self.decoded, 'resource_type': self.resource_type,
                'job_timeout_seconds': self.job_timeout_seconds}

    @classmethod
    def from_file(cls, path: Path, *, repo_root: Path, manifest: dict[str, Any]):
        payload = json.loads(path.read_text())
        roster = {trial.trial_id: trial for trial in trials(repo_root, manifest['source_commit'])}
        if payload.get('trial_id') not in roster:
            raise CalibrationError('spec_trial_unknown')
        result = cls(roster[payload['trial_id']], manifest['identity'])
        if payload != result.to_dict():
            raise CalibrationError('run_spec_mismatch')
        return result


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
