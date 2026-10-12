"""One fresh calibration trial, using shared trusted low-level execution.

The explicit execution body retains the comparison runner's request provenance,
credential redaction, cancellation and winner/snapshot binding. This module has
its own live gate, spec and ledger; it cannot open the original experiment.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from research.calibration.gates import check_review
from research.calibration.ledger import CalibrationLedger
from research.calibration.specs import RunSpec, protocol_identity, trials
from research.comparison import deploy as ac
from research.comparison.control import ComparisonMainBudgetControl, run_comparison_search
from research.comparison.protocol import ModelProfile
from research.comparison.runner import (
    ComparisonRunnerError, _TRACE_REQUEST_KEYS, _bind_incumbent,
    _operator_counts, _check_cooperative_stop, _sha256_file,
    bind_comparison_prediction_snapshots, build_comparison_solver,
    comparison_config_identity, select_incumbent, verify_task_data_identity,
)
from research.comparison.task import ComparisonSandboxTask
from research.contracts.budget import CancellationToken
from research.contracts.persist import atomic_write_json
from research.contracts.source_gate import load_policy
from research.evaluator.service import EvaluatorRegistry
from research.search.acceptance_inner import (
    job_outcome_counts, load_sandbox_api_key, make_generation_outcome_sink,
    make_prepare_operator, operator_clients,
)
from research.search.credentials import KimiAuthHelper, ManagedKimiCredentials, assert_redacted

RUN_RESULT_SCHEMA = 'calibration-inner-result.v1'
LIVE_JOBS_SCHEMA = 'calibration-live-jobs.v1'

def run_inner(spec_path: Path, *, repo_root: Path) -> int:
    """Execute one comparison inner run; write run-result.json.

    Exit 0: run-result.json written (check its status/termination fields).
    Exit 2: refused/failed before any result could be produced.
    """
    manifest = check_review(repo_root)  # before auth/model/ledger use
    spec = RunSpec.from_file(spec_path, repo_root=repo_root, manifest=manifest)
    roster = trials(repo_root, manifest['source_commit'])
    registry = EvaluatorRegistry.load(repo_root / 'research/evaluator/registry.v1.json')
    data_evidence = verify_task_data_identity(
        spec.run.task_binding, repo_root=repo_root, registry=registry
    )
    ledger_path = repo_root / spec.ledger_path
    if ledger_path.is_symlink() or not ledger_path.exists():
        # The runner never creates or resets the one comparison ledger, and
        # never follows a symlinked ledger (foreign accounting target).
        raise ComparisonRunnerError("comparison_ledger_missing")
    run_dir = repo_root / spec.run_dir
    # Fresh controller-private run directory ONLY: an interrupted config is
    # never silently rerun (existing dir/checkpoint refused), and no
    # symlinked path component may redirect artifacts outside the runtime
    # (checked BEFORE any ledger/auth/dojo activity).
    if run_dir.exists() or run_dir.is_symlink():
        raise ComparisonRunnerError("run_dir_not_fresh")
    ancestor = run_dir.parent
    while ancestor != repo_root and ancestor != ancestor.parent:
        if ancestor.is_symlink():
            raise ComparisonRunnerError("run_dir_symlink_ancestor")
        ancestor = ancestor.parent
    try:
        run_dir.mkdir(parents=True)
    except FileExistsError as exc:
        raise ComparisonRunnerError("run_dir_not_fresh") from exc
    os.environ["LOGGING_DIR"] = str(run_dir / "logs")

    # Deferred imports: dojo needs LOGGING_DIR; keep them out of module
    # scope so the spec/result types stay importable in the research venv.
    # litellm must use its BUNDLED model-cost map: frozen comparison startup
    # never fetches mutable public pricing metadata.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    import random  # noqa: PLC0415

    import numpy  # noqa: PLC0415
    from dojo.core.interpreters.base import ExecutionResult  # noqa: PLC0415
    from dojo.solvers.evo.operator_trace import OperatorTraceWriter  # noqa: PLC0415
    from dojo.utils.logger import config_logger  # noqa: PLC0415
    from research.adapters.sandbox_eval_client import SandboxEvalClient  # noqa: PLC0415

    config_logger(None)  # low-resource global logger: no file/wandb handlers

    # Seed the SEARCH RNGs with the paired inner seed (method-independent by
    # design). Provider generation stays nondeterministic: no unsupported
    # provider seed parameter is sent, and no result is cached or shared.
    random.seed(spec.run.inner_seed)
    numpy.random.seed(spec.run.inner_seed)
    rng_evidence = {
        "python_random_seed": spec.run.inner_seed,
        "numpy_random_seed": spec.run.inner_seed,
        "scope": "inner solver branch/selection seeding only; provider generation is nondeterministic",
    }

    secrets: list[str] = []
    sandbox_key = load_sandbox_api_key(repo_root / ac.AUTH_FILE)
    secrets.append(sandbox_key)
    credentials = ManagedKimiCredentials()
    auth_helper = KimiAuthHelper()
    try:
        credentials.attach_helper(auth_helper)

        policy = load_policy()
        request_trace_path = run_dir / "request-trace.jsonl"
        attempt_counter = {"n": 0}
        debug_budget = 0 if int(spec.decoded["max_debug_depth"]) == 0 else 2

        incumbent: Any | None = None
        end_snapshot: dict[str, Any] = {}
        cleanup_verified = False
        with CalibrationLedger.open(ledger_path, roster=roster, manifest_identity=manifest['identity']) as ledger:
            if ledger.stopped:
                # Never run against a stopped comparison ledger: refuse
                # BEFORE any provider call or sandbox submission.
                raise ComparisonRunnerError("comparison_ledger_stopped")
            if ledger.final_phase is not None:
                # The irreversible final phase is active: no inner run may
                # start generation afterwards — not even an unused
                # invalid/skipped config with remaining quota or an
                # alternative fresh run directory. Refuse BEFORE any
                # provider call or sandbox submission (the reservation
                # guard independently refuses, defense in depth).
                raise ComparisonRunnerError("comparison_final_phase_active")
            ledger.begin_trial(spec.run)
            budget_control = ComparisonMainBudgetControl(
                ledger,
                spec.run,
                debug_budget=debug_budget,
                control_log_path=run_dir / "budget-control.jsonl",
            )

            def attempt_sink(
                reservation: Any,
                messages: Any,
                model_kwargs: Any,
                operator: Any = None,
            ) -> None:
                """Link every provider attempt to its ledger reservation and
                persist the PUBLIC request (never headers/credentials). A
                redaction or write failure aborts the attempt before the
                provider is called (reservation retained + global stop)."""
                attempt_counter["n"] += 1
                if operator == "debug":
                    budget_control.note_debug_attempt()
                record = {
                    "run_key": spec.run.run_key(),
                    "attempt": attempt_counter["n"],
                    "operator": operator or "unknown",
                    "kind": "debug" if operator == "debug" else "main",
                    "request_id": reservation.request_id,
                    "reserved_tokens": reservation.tokens,
                    "messages": messages,
                    "request": {
                        key: model_kwargs[key]
                        for key in _TRACE_REQUEST_KEYS
                        if isinstance(model_kwargs, Mapping) and key in model_kwargs
                    },
                }
                assert_redacted(record, *secrets)
                with open(request_trace_path, "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(record, sort_keys=True, default=str) + "\n"
                    )

            token = CancellationToken(deadline_epoch=ledger.deadline_epoch)
            live_jobs_path = run_dir / "live-jobs.json"
            submission_intents: set[str] = set()

            def persist_live_jobs(_event: Any = None) -> None:
                """Persist submission intents / live jobs / uncertain
                submissions BEFORE polling; survives interruption for
                emergency cleanup."""
                if isinstance(_event, Mapping):
                    if _event.get("event") == "submission_intent":
                        submission_intents.add(str(_event.get("trace_id")))
                    elif _event.get("event") == "job_registered":
                        submission_intents.discard(str(_event.get("trace_id")))
                atomic_write_json(live_jobs_path, {
                    "schema": LIVE_JOBS_SCHEMA,
                    "run_key": spec.run.run_key(),
                    "submission_intents": sorted(submission_intents),
                    "live_jobs": sorted(client.live_job_ids()),
                    "uncertain_submissions": sorted(client.uncertain_submission_ids()),
                    "updated_at": time.time(),
                })

            client = SandboxEvalClient(
                endpoint=ac.GATEWAY_URL,
                api_key=sandbox_key,
                require_cleanup=True,
                observer=persist_live_jobs,
            )
            persist_live_jobs()  # initial empty state: file must always exist
            task = ComparisonSandboxTask(
                client,
                token,
                ledger=ledger,
                data_dir=str(data_evidence["worker_data_public_path"]),
                task_id=str(data_evidence["registry_task_id"]),
                resource_type=spec.resource_type,
                job_timeout_seconds=spec.job_timeout_seconds,
                name_prefix="calibration-round2-" + spec.run.trial_id,
                execution_result_cls=ExecutionResult,
            )
            solver = build_comparison_solver(spec, repo_root=repo_root)
            trace_path = run_dir / "operator-trace.jsonl"
            solver.operator_tracer = OperatorTraceWriter(trace_path)
            stop_file = run_dir / "stop-request.json"
            solver.generation_outcome_sink = make_generation_outcome_sink(
                run_dir=run_dir,
                run_id=spec.run.run_key(),
                secrets=secrets,
                stop_file=stop_file,
                solver=solver,
            )
            prepare = make_prepare_operator(
                credentials, operator_clients(solver), secrets
            )
            termination = run_comparison_search(
                solver,
                task,
                {},
                ledger=ledger,
                run=spec.run,
                token=token,
                budget_control=budget_control,
                cancel_live_jobs=client.cancel_all_live,
                checkpoint_path=run_dir / "search-control-checkpoint.json",
                attempt_sink=attempt_sink,
                prepare_operator=prepare,
                cooperative_stop=lambda: _check_cooperative_stop(stop_file),
            )
            incumbent = select_incumbent(solver.journal)
            control_checkpoint = run_dir / "search-control-checkpoint.json"
            cancellation_ok = True
            if control_checkpoint.exists():
                try:
                    control = json.loads(
                        control_checkpoint.read_text(encoding="utf-8")
                    )
                    cancellation_ok = (
                        control.get("cleanup_error") is None
                        and all(
                            entry.get("cleanup_verified") is True
                            for entry in control.get("cancellation", [])
                        )
                    )
                except ValueError:
                    cancellation_ok = False
            else:
                cancellation_ok = False
            cleanup_verified = (
                cancellation_ok
                and not client.live_job_ids()
                and not client.uncertain_submission_ids()
                and not submission_intents  # durable intent/registration gap
            )
            if not cleanup_verified:
                # Unresolved live/uncertain jobs after cancellation are a
                # GLOBAL stop condition, durably recorded inside the
                # ledger's ownership — never a silent evidence gap.
                ledger.stop("sandbox_cleanup_unproven")
                if termination == "completed":
                    termination = "cleanup_unproven"
            # Strict prediction binding INSIDE ledger ownership: every job's
            # snapshot hash is verified; a SCORED job without a proven
            # snapshot stops the ledger instead of being silently skipped.
            prediction_artifacts = bind_comparison_prediction_snapshots(
                task.jobs, run_dir, repo_root, ledger
            )
            incumbent_evidence = _bind_incumbent(
                incumbent, task.jobs, prediction_artifacts, ledger
            )
            realized = _operator_counts(trace_path)
            realized_main = sum(realized[name] for name in ('draft', 'improve', 'crossover'))
            if termination == 'completed' and (realized_main != 6 or int(solver.state.current_generation) != 3):
                ledger.stop('inner_shape_incomplete')
            trial_status = ('completed' if termination == 'completed' and cleanup_verified else
                            'terminated' if termination.startswith('CellBudgetExhausted') else 'failed')
            ledger.finish_trial(spec.run, trial_status)
            end_snapshot = ledger.snapshot()
            deadline_epoch = ledger.deadline_epoch

        # Bind the generated-candidate journal into the run artifacts.
        journal_src = run_dir / "checkpoint" / "journal.jsonl"
        journal_path = run_dir / "journal.jsonl"
        if journal_src.exists():
            journal_path.write_bytes(journal_src.read_bytes())
        counts = _operator_counts(trace_path)
        main_candidates = sum(
            counts[name] for name in ("draft", "improve", "crossover")
        )
        generations_completed = int(getattr(solver.state, "current_generation", 0))
        outcomes_path = run_dir / "generation-outcomes.jsonl"

        run_requests = [
            rec
            for rec in end_snapshot.get("requests", [])
            if rec.get("run_identity") == spec.run.identity()
        ]
        stopped_globally = end_snapshot.get("stopped") is not None
        if (
            termination == "completed"
            and not stopped_globally
            and cleanup_verified
        ):
            status = "completed"
        elif (
            termination.startswith("CellBudgetExhausted")
            and not stopped_globally
        ):
            status = "cell_terminated"
        else:
            # A globally stopped ledger can never report a completed run,
            # and a refused winner never survives as evidence.
            status = "failed"
        result = {
            "schema": RUN_RESULT_SCHEMA,
            "manifest_identity": manifest['identity'],
            "trial_id": spec.run.trial_id,
            "repeat": spec.run.record['repeat'],
            "config_id": spec.run.record['config_id'],
            "run_key": spec.run.run_key(),
            "cell_key": spec.run.cell.cell_key(),
            "config_index": spec.run.config_index,
            "inner_seed": spec.run.inner_seed,
            "decoded": dict(spec.decoded),
            "config_identity": comparison_config_identity(spec),
            "run_identity": spec.run.identity(),
            "cell_identity": spec.run.cell.identity(),
            "protocol_identity": protocol_identity(),
            "model_profile_identity": ModelProfile().identity(),
            "policy_version": policy.policy_version,
            "policy_sha256": policy.policy_sha256,
            "runner_commit": spec.runner_commit,
            "dataset": data_evidence,
            "status": status,
            "termination": termination,
            "incumbent": incumbent_evidence,
            "metric": "accuracy",
            "metric_direction": "maximize",
            "debug_budget": debug_budget,
            "main_candidates": main_candidates,
            "generations_completed": generations_completed,
            "rng": rng_evidence,
            "operator_counts": counts,
            "operator_trace_sha256": (
                _sha256_file(trace_path) if trace_path.exists() else ""
            ),
            "request_trace_sha256": (
                _sha256_file(request_trace_path) if request_trace_path.exists() else ""
            ),
            "journal_sha256": (
                _sha256_file(journal_path) if journal_path.exists() else ""
            ),
            "prediction_artifacts": prediction_artifacts,
            "jobs": task.jobs,
            "job_counts": job_outcome_counts(task.jobs),
            "generation_outcomes_sha256": (
                _sha256_file(outcomes_path) if outcomes_path.exists() else ""
            ),
            "job_ids": sorted({j["job_id"] for j in task.jobs if j.get("job_id")}),
            "worker_cleanup_verified": cleanup_verified,
            "requests": {
                "main": sum(1 for r in run_requests if r["kind"] == "main"),
                "debug": sum(1 for r in run_requests if r["kind"] == "debug"),
                "charged_tokens": sum(int(r["charged_tokens"]) for r in run_requests),
            },
            "budget": {
                "global_committed": end_snapshot.get("global", {}).get("committed"),
                "stopped": end_snapshot.get("stopped"),
                "cell": end_snapshot.get("cells", {}).get(
                    spec.run.cell.cell_key()
                ),
                "deadline_epoch": deadline_epoch,
            },
            "started_at": end_snapshot.get("started_at"),
            "finished_at": time.time(),
        }
        assert_redacted(result, *secrets)
        atomic_write_json(run_dir / "run-result.json", result)
        return 0
    finally:
        auth_helper.close()  # only this invocation's owned helpers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    return run_inner(args.spec, repo_root=args.repo_root.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
