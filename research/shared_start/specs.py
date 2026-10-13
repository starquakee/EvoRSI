"""Frozen shared-start pilot contract; independent of all historic ledgers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from research.calibration.plan import identity
from research.comparison.data import load_task_bindings
from research.comparison.protocol import ModelProfile
from research.comparison.specs import TaskBinding

RUNTIME = Path('.runtime/shared-start-pilot')
LEDGER = RUNTIME / 'budget-ledger.json'
SCHEMA = 'spambase-shared-start-pilot.v1'
GLOBAL_LIMITS = {'max_requests': 174, 'max_tokens': 600000, 'max_elapsed_seconds': 14400}
POOL_LIMITS = {'max_requests': 4, 'max_tokens': 20000, 'max_elapsed_seconds': 900,
               'max_main_requests': 2, 'max_debug_requests': 2}
CONTINUATION_LIMITS = {'max_requests': 6, 'max_tokens': 40000, 'max_elapsed_seconds': 1800,
                       'max_main_requests': 4, 'max_debug_requests': 2}
# Compatibility export; execution and accounting must use Trial.limits.
TRIAL_LIMITS = CONTINUATION_LIMITS
LIMITS_BY_KIND = {'fresh_seed_pool': POOL_LIMITS, 'continuation': CONTINUATION_LIMITS}
APPROVED_PLAN_SHA256 = 'c633c56c2536e12a6957dc81161e3f20bff3a11b7a87569f49a36d409e117ff4'
JOB_TIMEOUT = 600


class CalibrationError(RuntimeError):
    """A calibration contract or evidence gate refused."""


def selected_plan() -> dict[str, Any]:
    """Normalize the exact human-approved draft; never derive from live results.

    The draft status is retained verbatim in its frozen source file. Authorization
    is separately documented by the approved execution protocol and source gate.
    """
    path = Path(__file__).parent / 'approved-plan.v1.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != APPROVED_PLAN_SHA256:
        raise CalibrationError('approved_plan_hash_mismatch')
    actual = json.loads(raw)
    if (actual['schema'] != 'shared-start-pilot-draft.v1'
            or actual['blocks'] != 3 or actual['fresh_seed_pools'] != 3
            or actual['continuation_trials'] != 27 or len(actual['units']) != 30
            or actual['global_caps'] != {'requests': 174, 'tokens': 600000,
                                        'elapsed_seconds': 14400}
            or actual['per_job_timeout_seconds'] != JOB_TIMEOUT
            or actual['new_final_evaluations'] != 0
            or actual['automatic_retry_resume_or_extension'] is not False):
        raise CalibrationError('selected_plan_mismatch')
    configurations: list[dict[str, Any]] = [
        {'config_id': f'c{index:02d}', 'decoded': {
            'crossover_prob': crossover, 'score_weight': score,
            'delta_weight': 0.4, 'novelty_weight': 0.25, 'max_debug_depth': 1}}
        for index, (crossover, score) in enumerate(
            ( (x, y) for x in (0.05, 0.5, 0.95) for y in (0.0, 1.0, 2.0)))
    ]
    by_config: dict[str, dict[str, Any]] = {c['config_id']: c['decoded'] for c in configurations}
    orders = ((6, 1, 0, 8, 2, 5, 4, 3, 7),
              (1, 2, 7, 5, 8, 4, 6, 3, 0),
              (4, 7, 0, 6, 5, 8, 1, 2, 3))
    seeds = (2675261934, 4247893452, 2889346271)
    schedule: list[dict[str, Any]] = []
    for repeat, order in enumerate(orders):
        pool_id = f'b{repeat:02d}-pool'
        for position_in_block, index in enumerate((None, *order)):
            kind = 'fresh_seed_pool' if index is None else 'continuation'
            config_id = 'c04' if index is None else f'c{index:02d}'
            trial_id = pool_id if index is None else f'b{repeat:02d}-{config_id}'
            unit = actual['units'][len(schedule)]
            expected: dict[str, Any] = {'unit_id': trial_id, 'kind': kind, 'block': repeat + 1,
                        **LIMITS_BY_KIND[kind]}
            if kind == 'continuation':
                expected.update(pool_id=pool_id, inner_seed=seeds[repeat],
                                configuration={'config_id': config_id, **by_config[config_id]})
            if unit != expected:
                raise CalibrationError('schedule_or_unit_limits_mismatch')
            schedule.append({'trial_id': trial_id, 'unit_id': trial_id,
                             'repeat': repeat, 'position': len(schedule),
                             'position_in_block': position_in_block, 'kind': kind,
                             'pool_id': pool_id, 'config_id': config_id,
                             'decoded': dict(by_config[config_id]),
                             'inner_seed': seeds[repeat], 'limits': dict(LIMITS_BY_KIND[kind])})
    result = {'schema': 'shared-start-selected-plan.v1',
              'approved_plan_sha256': APPROVED_PLAN_SHA256,
              'schedule_seed': actual['schedule_seed'], 'configurations': configurations,
              'schedule': schedule,
              'budget': {'repeats': 3, 'trials': 30, 'fresh_seed_pools': 3,
                         'continuation_trials': 27, **GLOBAL_LIMITS}}
    result['plan_identity'] = identity(result)
    return result


def protocol() -> dict[str, Any]:
    return {'schema': SCHEMA, 'task': 'spambase', 'repeats': 3, 'trials': 30,
            'fresh_seed_pools': 3, 'continuation_trials': 27,
            'plan_identity': selected_plan()['plan_identity'],
            'approved_plan_sha256': APPROVED_PLAN_SHA256,
            'global_limits': dict(GLOBAL_LIMITS),
            'trial_limits': {kind: dict(limits) for kind, limits in LIMITS_BY_KIND.items()},
            'job_timeout_seconds': JOB_TIMEOUT, 'model_profile': ModelProfile().to_dict(),
            'data_phase': 'validation', 'new_final_scoring': False,
            'retry_or_resume': False, 'budget_extension': False,
            'candidate_cache_between_continuations': False,
            'shared_seed_candidates_per_block': 2}


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
    def kind(self) -> str:
        return self.record['kind']

    @property
    def limits(self) -> dict[str, int]:
        return dict(LIMITS_BY_KIND[self.kind])

    @property
    def main_target(self) -> int:
        return self.limits['max_main_requests']

    @property
    def pool_id(self) -> str:
        return self.record['pool_id']

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
        return {'schema': 'shared-start-run-spec.v1', 'trial_id': self.run.trial_id,
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
