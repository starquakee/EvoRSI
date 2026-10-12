"""One durable global/trial ledger for the separately authorized calibration.

Limits are fixed here; no reset, expansion, hidden retry, or resumable trial.
Old comparison ledgers cannot be opened by this schema/manifest-bound class.
"""
from __future__ import annotations

import copy
import fcntl
import math
import os
from pathlib import Path
import threading
import time
import uuid

from research.calibration.specs import GLOBAL_LIMITS, TRIAL_LIMITS, Trial, selected_plan
from research.comparison.budget import (
    CellBudgetExhausted, ComparisonBudgetError, ComparisonLedgerLocked,
    ComparisonLedgerStopped, ComparisonReservation, GlobalBudgetExhausted,
    InnerRequestLimitReached, ProviderUsage, _atomic_write_durable,
)
from research.comparison.protocol import ModelProfile
from research.contracts.budget import ModelUsage

SCHEMA = 'calibration-budget-ledger.v1'


def _integer(value):
    return type(value) is int and value >= 0


def _timestamp(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


class CalibrationLedger:
    def __init__(self, path, roster, manifest_identity, state, lock_handle, now_fn):
        self.path = Path(path)
        self._roster = {trial.identity(): trial for trial in roster}
        self._manifest_identity = manifest_identity
        self._state = state
        self._lock_handle = lock_handle
        self._now_fn = now_fn
        self._mutex = threading.RLock()
        self._attempt_mutex = threading.Lock()
        self._closed = False

    @classmethod
    def _acquire(cls, path, roster, manifest_identity, now_fn):
        path = Path(path)
        if path.is_symlink() or path.with_suffix('.lock').is_symlink():
            raise ComparisonBudgetError('ledger_symlink_refused')
        for trial in roster:
            if not isinstance(trial, Trial):
                raise ComparisonBudgetError('ledger_roster_invalid')
        expected = {item['trial_id'] for item in selected_plan()['schedule']}
        if len(roster) != 27 or {r.trial_id for r in roster} != expected or len({r.identity() for r in roster}) != 27 or len({r.source_commit for r in roster}) != 1 or len({r.task_binding.identity() for r in roster}) != 1:
            raise ComparisonBudgetError('ledger_roster_invalid')
        lock = path.with_suffix('.lock').open('a+')
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock.close()
            raise ComparisonLedgerLocked('calibration_ledger_locked') from exc
        return cls(path, roster, manifest_identity, {}, lock, now_fn)

    @classmethod
    def create(cls, path, *, roster, manifest_identity, now_fn=time.time):
        ledger = cls._acquire(path, roster, manifest_identity, now_fn)
        try:
            if ledger.path.exists():
                raise ComparisonBudgetError('ledger_already_exists')
            ledger._state = {'schema': SCHEMA, 'manifest_identity': manifest_identity,
                'limits': {'global': GLOBAL_LIMITS, 'trial': TRIAL_LIMITS},
                'started_at': None, 'stopped': None, 'requests': [],
                'trials': {trial.identity(): {'trial_id': trial.trial_id, 'started_at': None,
                           'status': 'pending', 'terminated': None}
                           for trial in roster}}
            ledger._persist()
            return ledger
        except BaseException:
            ledger.close()
            raise

    @classmethod
    def open(cls, path, *, roster, manifest_identity, now_fn=time.time):
        import json
        ledger = cls._acquire(path, roster, manifest_identity, now_fn)
        try:
            ledger._state = json.loads(ledger.path.read_text())
            ledger._validate()
            for record in ledger._state['requests']:
                if record['status'] == 'reserved':
                    record['status'] = 'unknown'
                    ledger._stop('unresolved_reservation_on_open')
            if any(value['status'] == 'running' for value in ledger._state['trials'].values()):
                ledger._stop('interrupted_trial_no_resume')
            ledger._persist()
            return ledger
        except BaseException:
            ledger.close()
            raise

    def _validate(self):
        state = self._state
        if state.get('schema') != SCHEMA or state.get('manifest_identity') != self._manifest_identity:
            raise ComparisonBudgetError('ledger_identity_mismatch')
        if state.get('limits') != {'global': GLOBAL_LIMITS, 'trial': TRIAL_LIMITS}:
            raise ComparisonBudgetError('ledger_limits_mismatch')
        if set(state.get('trials', {})) != set(self._roster):
            raise ComparisonBudgetError('ledger_roster_mismatch')
        if state.get('started_at') is not None and not _timestamp(state['started_at']):
            raise ComparisonBudgetError('ledger_clock_invalid')
        if state.get('stopped') is not None and (not isinstance(state['stopped'], dict)
                or not isinstance(state['stopped'].get('reason'), str)
                or not _timestamp(state['stopped'].get('at'))):
            raise ComparisonBudgetError('ledger_stop_invalid')
        running = 0
        for key, trial in state['trials'].items():
            if trial.get('trial_id') != self._roster[key].trial_id or trial.get('status') not in ('pending', 'running', 'completed', 'terminated', 'failed'):
                raise ComparisonBudgetError('ledger_trial_invalid')
            if trial['status'] == 'pending' and trial.get('started_at') is not None:
                raise ComparisonBudgetError('ledger_trial_clock_invalid')
            if trial['status'] != 'pending' and not _timestamp(trial.get('started_at')):
                raise ComparisonBudgetError('ledger_trial_clock_invalid')
            running += trial['status'] == 'running'
        if running > 1:
            raise ComparisonBudgetError('concurrent_trials_forbidden')
        ids, provider_ids = set(), set()
        for record in state['requests']:
            key = record.get('run_identity')
            if key not in self._roster or record.get('cell_identity') != key:
                raise ComparisonBudgetError('request_trial_mismatch')
            request_id = record.get('request_id')
            if not isinstance(request_id, str) or not request_id or request_id in ids:
                raise ComparisonBudgetError('request_identity_invalid')
            ids.add(request_id)
            if record.get('kind') not in ('main', 'debug') or record.get('status') not in ('reserved', 'known', 'unknown'):
                raise ComparisonBudgetError('request_record_invalid')
            if not _integer(record.get('reserved_tokens')) or record['reserved_tokens'] < ModelProfile().max_output_tokens:
                raise ComparisonBudgetError('reservation_invalid')
            if not _integer(record.get('charged_tokens')) or not _timestamp(record.get('created_at')):
                raise ComparisonBudgetError('request_accounting_invalid')
            usage = record.get('usage')
            if record['status'] == 'known':
                if not isinstance(usage, dict) or set(usage) != {'input_tokens', 'output_tokens'} or not all(_integer(v) for v in usage.values()) or sum(usage.values()) != record['charged_tokens']:
                    raise ComparisonBudgetError('known_usage_invalid')
            elif usage is not None or record['charged_tokens'] != record['reserved_tokens']:
                raise ComparisonBudgetError('unsettled_usage_invalid')
            provider_id = record.get('provider_request_id')
            if provider_id is not None:
                if not isinstance(provider_id, str) or not provider_id or provider_id in provider_ids:
                    raise ComparisonBudgetError('provider_identity_invalid')
                provider_ids.add(provider_id)
        if state['requests'] and state['started_at'] is None:
            raise ComparisonBudgetError('requests_before_clock')
        if len(state['requests']) > GLOBAL_LIMITS['max_requests']:
            raise ComparisonBudgetError('request_limit_corrupt')
        for key in self._roster:
            records = [r for r in state['requests'] if r['run_identity'] == key]
            if len(records) > 8 or sum(r['kind'] == 'main' for r in records) > 6 or sum(r['kind'] == 'debug' for r in records) > 2:
                raise ComparisonBudgetError('trial_shape_corrupt')
            if records and state['trials'][key]['status'] == 'pending':
                raise ComparisonBudgetError('requests_before_trial')
        if not self.stopped:
            if any(r['status'] == 'unknown' for r in state['requests']):
                raise ComparisonBudgetError('unknown_usage_without_stop')
            if sum(r['charged_tokens'] for r in state['requests']) > GLOBAL_LIMITS['max_tokens']:
                raise ComparisonBudgetError('global_token_limit_corrupt')
            if any(sum(r['charged_tokens'] for r in state['requests'] if r['run_identity'] == key) > TRIAL_LIMITS['max_tokens'] for key in self._roster):
                raise ComparisonBudgetError('trial_token_limit_corrupt')

    def _persist(self):
        if self._closed:
            raise ComparisonBudgetError('ledger_closed')
        _atomic_write_durable(self.path, self._state)

    @property
    def stopped(self):
        return self._state['stopped'] is not None

    @property
    def final_phase(self):
        return None

    @property
    def started_at(self):
        return self._state['started_at']

    @property
    def deadline_epoch(self):
        if self.started_at is None:
            return None
        deadline = self.started_at + GLOBAL_LIMITS['max_elapsed_seconds']
        for trial in self._state['trials'].values():
            if trial['status'] == 'running':
                deadline = min(deadline, trial['started_at'] + TRIAL_LIMITS['max_elapsed_seconds'])
        return deadline

    def _stop(self, reason):
        if not self.stopped:
            self._state['stopped'] = {'reason': str(reason)[:200], 'at': self._now_fn()}

    def stop(self, reason):
        with self._mutex:
            self._stop(reason)
            self._persist()

    def _active(self):
        if self.stopped:
            raise ComparisonLedgerStopped(self._state['stopped']['reason'])
        if self.started_at is None:
            raise ComparisonBudgetError('clock_not_started')
        now = self._now_fn()
        if not _timestamp(now) or now < self.started_at or now >= self.deadline_epoch:
            self._stop('deadline_or_clock_invalid')
            self._persist()
            raise GlobalBudgetExhausted('deadline_or_clock_invalid')

    def start_clock(self):
        with self._mutex:
            if self.started_at is not None or self.stopped:
                raise ComparisonBudgetError('clock_already_started_or_stopped')
            now = self._now_fn()
            if not _timestamp(now):
                raise ComparisonBudgetError('clock_invalid')
            self._state['started_at'] = now
            self._persist()

    def begin_trial(self, run):
        with self._mutex:
            self._active()
            key = run.identity()
            if key not in self._roster or self._state['trials'][key]['status'] != 'pending' or any(t['status'] == 'running' for t in self._state['trials'].values()):
                self.stop('trial_not_fresh_or_concurrent')
                raise ComparisonBudgetError('trial_not_fresh_or_concurrent')
            self._state['trials'][key].update(status='running', started_at=self._now_fn())
            self._persist()

    def finish_trial(self, run, status):
        with self._mutex:
            key = run.identity()
            if key not in self._roster or status not in ('completed', 'terminated', 'failed'):
                raise ComparisonBudgetError('trial_finish_invalid')
            if self._state['trials'][key]['status'] not in ('running', 'terminated'):
                raise ComparisonBudgetError('trial_finish_not_running')
            if any(r['status'] == 'reserved' for r in self._state['requests']):
                self.stop('unsettled_trial_end')
                raise ComparisonBudgetError('unsettled_trial_end')
            self._state['trials'][key]['status'] = 'failed' if self.stopped else status
            self._persist()

    def terminate_cell(self, cell_identity, reason):
        with self._mutex:
            if cell_identity not in self._roster:
                raise ComparisonBudgetError('unknown_trial')
            trial = self._state['trials'][cell_identity]
            if trial['status'] not in ('running', 'terminated'):
                raise ComparisonBudgetError('trial_not_active')
            trial['terminated'] = trial['terminated'] or {'reason': str(reason)[:200], 'at': self._now_fn()}
            trial['status'] = 'terminated'
            self._persist()

    def snapshot(self):
        with self._mutex:
            records = self._state['requests']
            def view(items, limits):
                requests, tokens = len(items), sum(item['charged_tokens'] for item in items)
                return {'committed': {'requests': requests, 'tokens': tokens},
                        'remaining': {'requests': max(0, limits['max_requests'] - requests),
                                      'tokens': max(0, limits['max_tokens'] - tokens)}}
            return copy.deepcopy({'schema': SCHEMA, 'manifest_identity': self._manifest_identity,
                'started_at': self.started_at, 'stopped': self._state['stopped'],
                'global': view(records, GLOBAL_LIMITS), 'requests': records,
                'cells': {trial['trial_id']: {**trial, **view([r for r in records if r['run_identity'] == key], TRIAL_LIMITS)}
                          for key, trial in self._state['trials'].items()},
                'deadline_epoch': self.deadline_epoch})

    def reserve(self, run, kind, *, estimated_input_tokens, debug_limit=None):
        with self._mutex:
            self._active()
            key = run.identity()
            if key not in self._roster or self._state['trials'][key]['status'] != 'running':
                raise ComparisonBudgetError('trial_not_active')
            if kind not in ('main', 'debug') or not _integer(estimated_input_tokens) or estimated_input_tokens < 1:
                raise ComparisonBudgetError('request_shape_invalid')
            if kind == 'debug' and (type(debug_limit) is not int or not 0 <= debug_limit <= 2):
                raise ComparisonBudgetError('debug_limit_invalid')
            existing = [r for r in self._state['requests'] if r['run_identity'] == key]
            limit = 6 if kind == 'main' else debug_limit
            if sum(r['kind'] == kind for r in existing) >= limit:
                raise InnerRequestLimitReached('inner_request_limit')
            reserved = estimated_input_tokens + ModelProfile().max_output_tokens
            snapshot = self.snapshot()
            if snapshot['global']['remaining']['requests'] < 1 or snapshot['global']['remaining']['tokens'] < reserved:
                self.stop('global_budget_exhausted')
                raise GlobalBudgetExhausted('global_budget_exhausted')
            remaining = snapshot['cells'][run.cell_key()]['remaining']
            if remaining['requests'] < 1 or remaining['tokens'] < reserved:
                self.terminate_cell(key, 'trial_budget_exhausted')
                raise CellBudgetExhausted('trial_budget_exhausted')
            request_id = uuid.uuid4().hex
            record = {'request_id': request_id, 'run_identity': key, 'cell_identity': key,
                      'kind': kind, 'created_at': self._now_fn(), 'reserved_tokens': reserved,
                      'charged_tokens': reserved, 'status': 'reserved', 'usage': None,
                      'provider_request_id': None}
            self._state['requests'].append(record)
            self._persist()
            return ComparisonReservation(request_id, key, key, kind, reserved, record['created_at'])

    def commit(self, request_id, usage, *, provider_request_id=None):
        with self._mutex:
            matches = [r for r in self._state['requests'] if r['request_id'] == request_id]
            if len(matches) != 1 or matches[0]['status'] != 'reserved':
                raise ComparisonBudgetError('reservation_not_open')
            record = matches[0]
            if provider_request_id is not None and (not isinstance(provider_request_id, str) or not provider_request_id or any(r['provider_request_id'] == provider_request_id for r in self._state['requests'])):
                raise ComparisonBudgetError('provider_identity_duplicate_or_invalid')
            if usage is None:
                record['status'] = 'unknown'
                self._stop('unknown_provider_usage')
            else:
                if not isinstance(usage, (ProviderUsage, ModelUsage)):
                    raise ComparisonBudgetError('usage_type_invalid')
                if not _integer(usage.input_tokens) or not _integer(usage.output_tokens):
                    raise ComparisonBudgetError('usage_invalid')
                record.update(status='known', charged_tokens=usage.input_tokens + usage.output_tokens,
                              usage={'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens},
                              provider_request_id=provider_request_id)
                if record['charged_tokens'] > record['reserved_tokens']:
                    self._stop('usage_exceeded_reservation')
            self._persist()

    def close(self):
        if not self._closed:
            self._closed = True
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
