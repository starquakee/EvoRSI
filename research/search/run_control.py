"""Production control for one real Evo run under a shared acceptance ledger."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from research.contracts.budget import BudgetError, BudgetLedger, CancellationToken, Cancelled, LedgerStopped
from research.contracts.persist import atomic_write_json
from research.search.budget_transport import TransportGuardConfig, install_budget_guard


def run_guarded_search(
    solver: Any, task: Any, state: Any, *, ledger: BudgetLedger,
    token: CancellationToken, transport_config: TransportGuardConfig,
    cancel_live_jobs: Callable[[], list[dict[str, Any]]],
    checkpoint_path: Path,
    prepare_operator: Callable[[], None] | None = None,
) -> Any:
    """Use the real solver, real operator clients, and real transport guards.

    The checkpoint contains only accounting/cancellation state. The solver's
    own journal stays in its configured ignored run directory. Caller must
    supply the shared ledger; this function never creates or resets one.
    """
    if getattr(solver, 'before_operator', None) is not None:
        raise BudgetError('search_control_already_installed')
    mode = str(getattr(solver.cfg, 'execution_mode', 'generation'))
    if mode != 'generation':
        raise BudgetError('acceptance_requires_serial_generation_mode')
    for name in ('draft', 'improve', 'debug', 'crossover', 'analyze', 'rich_memory_summary'):
        operator = getattr(solver, name + '_fn', None)
        if operator is None:
            continue
        args = getattr(operator, 'args', ())
        client = getattr(args[0], 'client', None) if args else None
        if client is None:
            raise BudgetError('operator_client_unavailable:' + name)
        install_budget_guard(client, ledger, token, transport_config, operator=name)

    def before_operator() -> None:
        token.check()
        snapshot = ledger.snapshot()
        if snapshot['stopped'] is not None:
            raise LedgerStopped(str(snapshot['stopped']['reason']))
        if snapshot['remaining']['elapsed_seconds'] <= 0:
            raise Cancelled('deadline_exceeded')
        if getattr(task, 'stop_requested', False):
            raise Cancelled('task_stop_requested')
        if solver.state.current_step >= solver.cfg.step_limit:
            raise Cancelled('solver_step_limit')
        # SDK debug logging can include request options. Auth is runtime-only
        # and must not enter config/journal/log artifacts.
        for name in ('openai', 'httpx', 'httpcore', 'LiteLLM', 'litellm'):
            logging.getLogger(name).setLevel(logging.WARNING)
        if prepare_operator is not None:
            prepare_operator()

    solver.before_operator = before_operator
    termination = 'completed'
    cleanup: list[dict[str, Any]] = []
    cleanup_error: str | None = None
    try:
        before_operator()
        return solver.search(task, state)
    except BaseException as exc:
        termination = type(exc).__name__
        token.stop('search_stopped:' + termination)
        ledger.stop('search_stopped:' + termination)
        raise
    finally:
        try:
            cleanup = cancel_live_jobs()
            if any(item.get('cleanup_verified') is not True for item in cleanup):
                cleanup_error = 'sandbox_cleanup_unproven'
        except Exception as exc:
            cleanup_error = 'cancel_failed:' + type(exc).__name__
        if cleanup_error is not None:
            token.stop(cleanup_error)
            ledger.stop(cleanup_error)
        try:
            solver.save_checkpoint()
        finally:
            atomic_write_json(checkpoint_path, {
                'schema': 'search-control.v1', 'termination': termination,
                'token': token.to_dict(), 'budget': ledger.snapshot(),
                'cancellation': cleanup, 'cleanup_error': cleanup_error,
                'inner_generation': solver.state.current_generation,
                'journal_steps': solver.state.current_step,
            })
        if cleanup_error is not None:
            raise BudgetError(cleanup_error)
