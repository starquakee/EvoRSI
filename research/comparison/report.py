"""Offline full report for the iStratDE comparison (EXP-008).

Builds the complete 27-cell report from the ACTUAL persisted artifacts —
the frozen manifest/schedule, the original budget ledger (read-only,
never flocked or mutated), the orchestration cell states and the final
phase results — plus the predeclared FR7 paired bootstrap analysis and an
honest positive / no-gain / insufficient-evidence conclusion.

Everything here is read-only and offline: no network, no credentials, no
job submission or cancellation, no phase activation, no ledger creation
and no mutation of the original ledger, winners, predictions, journals,
sources or deadline. A matrix that was interrupted, capped or globally
stopped is reported truthfully: unobserved cells stay explicit, unknown
usage stays unknown, and no final score is ever inferred from validation
or filled with a success.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from research.comparison import review
from research.comparison.analysis import (
    ANALYSIS_CAVEATS,
    CONTRASTS,
    CellOutcome,
    analyze_contrast,
    conclude,
)
from research.comparison.budget import (
    ComparisonBudgetError,
    ReadOnlyComparisonLedger,
)
from research.comparison.orchestrate import OrchestrationError, load_orchestrator
from research.comparison.protocol import canonical_protocol
from research.comparison.review import ComparisonManifestError
from research.comparison.specs import CellSpec, iter_cells

FULL_REPORT_SCHEMA = "istratde-comparison-full-report.v1"

#: Final statuses that count as an OBSERVED final outcome.
_OBSERVED_FINAL_STATUSES = ("completed", "failed", "no_valid_program")


class ComparisonReportError(RuntimeError):
    """The report cannot be produced; the message IS the rule."""


def _cell_dir(runtime_dir: Path, cell_key: str) -> Path:
    return runtime_dir / "cells" / cell_key.replace(":", "-")


def _read_json(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _config_summary(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Per-status configuration counts with explicit reasons."""
    configs = (state or {}).get("configs")
    summary: dict[str, Any] = {
        "completed": 0,
        "gate_rejected": 0,
        "not_launched_cell_exhausted": 0,
        "launched_interrupted": 0,
        "reasons": [],
    }
    if not isinstance(configs, dict):
        return summary
    for key in sorted(configs, key=lambda k: int(k) if k.isdigit() else -1):
        record = configs[key]
        if not isinstance(record, dict):
            continue
        status = record.get("status")
        if status == "completed":
            summary["completed"] += 1
            termination = record.get("termination")
            if termination != "completed":
                summary["reasons"].append(
                    f"config {key}: {termination}"
                )
        elif status in ("gate_rejected", "not_launched_cell_exhausted"):
            summary[status] += 1
            reason = record.get("reason")
            summary["reasons"].append(
                f"config {key}: {status}" + (f" ({reason})" if reason else "")
            )
        elif status == "launched":
            # A durable launched record without a completed boundary means
            # the config was INTERRUPTED — never a silent success/failure.
            summary["launched_interrupted"] += 1
            summary["reasons"].append(f"config {key}: interrupted")
    return summary


def _count_candidates(runtime_dir: Path, cell: CellSpec) -> int | None:
    """Executed candidate jobs across the cell's runs (None if unknown).

    Counts ACTUAL producer jobs including execution failures
    (executed_and_cleaned from the shared job_outcome_counts contract),
    not just scored ones.
    """
    total = 0
    found = False
    for config_index in range(canonical_protocol().configs_per_cell):
        run = cell.inner_run(config_index)
        result = _read_json(
            runtime_dir
            / "runs"
            / run.run_key().replace(":", "-")
            / "run-result.json"
        )
        if result is None:
            continue
        counts = result.get("job_counts")
        if not isinstance(counts, dict):
            continue
        executed = counts.get("executed_and_cleaned")
        if isinstance(executed, bool):
            continue
        if isinstance(executed, int):
            total += executed
            found = True
            continue
        completed = counts.get("completed")
        if isinstance(completed, int) and not isinstance(completed, bool):
            total += completed
            found = True
    return total if found else None


def _incumbent_curve(
    cell: CellSpec,
    state: Mapping[str, Any] | None,
    requests_by_run: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    """Validation incumbent vs ACTUAL cumulative requests/tokens.

    One point per evaluated configuration boundary in schedule position
    order; requests/tokens come from the ORIGINAL ledger (never planned
    counts). Configs never launched contribute no point. This is a
    validation curve only — final test scores are never plotted as
    feedback curves.
    """
    configs = (state or {}).get("configs")
    if not isinstance(configs, dict):
        return []
    records = [
        record
        for record in configs.values()
        if isinstance(record, dict)
        and record.get("status") in ("completed", "gate_rejected")
        and isinstance(record.get("position"), int)
    ]
    records.sort(key=lambda record: int(record["position"]))
    points: list[dict[str, Any]] = []
    cumulative_requests = 0
    cumulative_tokens = 0
    incumbent: float | None = None
    for record in records:
        run = cell.inner_run(int(record["config_index"]))
        usage = requests_by_run.get(run.identity(), {})
        cumulative_requests += int(usage.get("requests", 0))
        cumulative_tokens += int(usage.get("charged_tokens", 0))
        # Only a genuinely evaluated configuration contributes its ACTUAL
        # validation score. A gate-rejected or all-failed config has
        # utility 0 but NO observed validation accuracy — never plot a
        # fabricated incumbent 0.
        raw_score = record.get("raw_score")
        if (
            record["status"] == "completed"
            and bool(record.get("evaluated"))
            and isinstance(raw_score, (int, float))
            and not isinstance(raw_score, bool)
        ):
            incumbent = (
                float(raw_score)
                if incumbent is None
                else max(incumbent, float(raw_score))
            )
        points.append(
            {
                "position": int(record["position"]),
                "config_index": int(record["config_index"]),
                "cumulative_requests": cumulative_requests,
                "cumulative_charged_tokens": cumulative_tokens,
                "validation_incumbent": incumbent,
            }
        )
    return points


def _best_observed_validation(state: Mapping[str, Any] | None) -> float | None:
    """Best ACTUALLY evaluated validation score among completed configs.

    A real partial search can have a validation incumbent before a winner
    is frozen; gate-rejected/all-failed configs contribute nothing.
    """
    configs = (state or {}).get("configs")
    if not isinstance(configs, dict):
        return None
    best: float | None = None
    for record in configs.values():
        if not isinstance(record, dict):
            continue
        raw = record.get("raw_score")
        if (
            record.get("status") == "completed"
            and bool(record.get("evaluated"))
            and isinstance(raw, (int, float))
            and not isinstance(raw, bool)
        ):
            best = float(raw) if best is None else max(best, float(raw))
    return best


def _build_rows(
    runtime_dir: Path,
    cells: Sequence[CellSpec],
    requests_by_run: Mapping[str, Mapping[str, int]],
    verified_finals: set[str],
) -> tuple[list[dict[str, Any]], dict[str, CellOutcome]]:
    rows: list[dict[str, Any]] = []
    outcomes: dict[str, CellOutcome] = {}
    for cell in cells:
        cell_key = cell.cell_key()
        state = _read_json(_cell_dir(runtime_dir, cell_key) / "cell.json")
        terminal = (state or {}).get("terminal")
        winner = (state or {}).get("winner")
        configs = _config_summary(state)
        if isinstance(terminal, dict):
            terminal_status = terminal.get("status")
        elif state is None:
            terminal_status = "not_started"
        elif configs["completed"] or configs["gate_rejected"]:
            terminal_status = "incomplete"
        elif configs["launched_interrupted"]:
            terminal_status = "interrupted"
        else:
            terminal_status = "not_started"
        final = _read_json(
            runtime_dir
            / "final"
            / cell_key.replace(":", "-")
            / "final.json"
        )
        final_status = final.get("status") if final is not None else None
        if final_status not in _OBSERVED_FINAL_STATUSES:
            final_status = None
        # A persisted final CLAIM counts as observed ONLY after the full
        # bound evidence chain (orchestration + original ledger + barrier +
        # publication + intent/winner/prediction/answer re-hash and
        # independent score recomputation) verified it. Bare or corrupt
        # claims stay explicit unverified diagnostics — never accepted
        # outcomes and never genuine zero-utility failures.
        final_verified = cell_key in verified_finals
        observed = final_status is not None and final_verified
        raw = final.get("raw_score") if final is not None and observed else None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raw = None
        utility: float | None
        if not observed:
            utility = None
        elif final_status == "completed" and raw is not None:
            utility = float(raw)
        else:
            utility = 0.0  # genuine observed failure
        rows.append(
            {
                "cell_key": cell_key,
                "task_id": cell.task_id,
                "method": cell.method,
                "search_seed": cell.search_seed,
                "terminal_status": terminal_status,
                "terminal_reason": (
                    terminal.get("reason") if isinstance(terminal, dict) else None
                ),
                "validation_incumbent": (
                    winner.get("validation_score")
                    if isinstance(winner, dict)
                    else _best_observed_validation(state)
                ),
                "winner_config_index": (
                    winner.get("config_index")
                    if isinstance(winner, dict)
                    else None
                ),
                "configs": configs,
                "candidates": _count_candidates(runtime_dir, cell),
                "incumbent_curve": _incumbent_curve(cell, state, requests_by_run),
                "final_status": final_status or (
                    "missing_final" if terminal_status in ("frozen", "failed_no_valid_program") else "not_reached"
                ),
                "final_verified": final_verified,
                "final_finished_at": (
                    final.get("finished_at") if final is not None else None
                ),
                "raw_accuracy": raw,
                "utility": utility,
            }
        )
        outcomes[cell_key] = CellOutcome(
            cell_key=cell_key,
            task_id=cell.task_id,
            method=cell.method,
            search_seed=cell.search_seed,
            observed=observed,
            raw_accuracy=float(raw) if raw is not None else None,
            utility=utility,
        )
    return rows, outcomes


def _summarize_ledger(
    snapshot: Mapping[str, Any],
    cells: Sequence[CellSpec],
) -> dict[str, Any]:
    """Actual accounting totals — failed calls included, unknown stays
    unknown (never zero-filled)."""
    requests = snapshot["requests"]
    by_kind = {"main": 0, "debug": 0}
    unknown_usage = 0
    input_tokens = 0
    output_tokens = 0
    charged_tokens = 0
    for record in requests:
        kind = record["kind"]
        by_kind[kind] = by_kind.get(kind, 0) + 1
        charged_tokens += int(record["charged_tokens"])
        if record["usage_known"]:
            input_tokens += int(record["input_tokens"])
            output_tokens += int(record["output_tokens"])
        else:
            unknown_usage += 1
    last_activity = max(
        (float(record["committed_at"]) for record in requests), default=None
    )
    per_cell: dict[str, Any] = {}
    cell_by_identity = {cell.identity(): cell.cell_key() for cell in cells}
    for record in requests:
        cell_key = cell_by_identity.get(record["cell_identity"])
        if cell_key is None:
            continue
        entry = per_cell.setdefault(
            cell_key,
            {
                "requests": 0,
                "main_requests": 0,
                "debug_requests": 0,
                "charged_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "unknown_usage_requests": 0,
                "first_request_at": None,
                "last_request_at": None,
            },
        )
        entry["requests"] += 1
        entry[f"{record['kind']}_requests"] += 1
        entry["charged_tokens"] += int(record["charged_tokens"])
        if record["usage_known"]:
            entry["input_tokens"] += int(record["input_tokens"])
            entry["output_tokens"] += int(record["output_tokens"])
        else:
            entry["unknown_usage_requests"] += 1
        committed_at = float(record["committed_at"])
        first = entry["first_request_at"]
        entry["first_request_at"] = (
            committed_at if first is None else min(first, committed_at)
        )
        last = entry["last_request_at"]
        entry["last_request_at"] = (
            committed_at if last is None else max(last, committed_at)
        )
    # Cells with no requests remain explicit with zero OBSERVED requests.
    for cell in cells:
        per_cell.setdefault(
            cell.cell_key(),
            {
                "requests": 0,
                "main_requests": 0,
                "debug_requests": 0,
                "charged_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "unknown_usage_requests": 0,
                "first_request_at": None,
                "last_request_at": None,
            },
        )
    stopped = snapshot["stopped"]
    started_at = snapshot["started_at"]
    elapsed_observed = None
    if started_at is not None:
        end = None
        if isinstance(stopped, dict):
            end = float(stopped["at"])
        elif last_activity is not None:
            end = last_activity
        if end is not None:
            elapsed_observed = max(0.0, end - float(started_at))
    cell_views = snapshot["cells"]
    return {
        "experiment_identity": snapshot["experiment_identity"],
        "started_at": started_at,
        "deadline_epoch": snapshot["deadline_epoch"],
        "stopped": stopped,
        "final_phase": snapshot["final_phase"],
        "elapsed_seconds_observed": elapsed_observed,
        "last_request_committed_at": last_activity,
        "requests": {
            "total": len(requests),
            "main": by_kind.get("main", 0),
            "debug": by_kind.get("debug", 0),
            "unknown_usage": unknown_usage,
        },
        "tokens": {
            "charged_total": charged_tokens,
            "input_known": input_tokens,
            "output_known": output_tokens,
            "unknown_usage_requests": unknown_usage,
            "note": (
                "charged_total includes full retained reservations for "
                "unknown-usage requests; realized provider cost for those "
                "requests is UNKNOWN, not zero"
            ),
        },
        "open_reservations": snapshot["global"]["open_reservations"],
        "limits": snapshot["limits"],
        "cells": {
            cell_key: {
                **per_cell[cell_key],
                "terminated": cell_views[cell_key]["terminated"],
            }
            for cell_key in sorted(per_cell)
        },
    }


def _execution_evidence_present(
    runtime_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> bool:
    """True when ANY persisted execution evidence exists at all.

    A single completed/interrupted configuration (even in a NONterminal
    cell), a terminal cell, a run result, an intent or a final artifact
    proves the experiment was active; the ORIGINAL ledger is then required
    and its absence is an integrity failure, not an honest not-started
    state.
    """
    for row in rows:
        if row["terminal_status"] != "not_started" or row["candidates"] is not None:
            return True
    for sub in ("runs", "intents", "final"):
        root = runtime_dir / sub
        if root.is_symlink() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                return True
    return False


def build_full_report(
    repo_root: Path,
    *,
    bindings: Mapping[str, Any] | None = None,
    provider: Any = None,
    runtime_dir: Path | None = None,
    sealed_root: Path | None = None,
    public_root: Path | None = None,
    evaluator_root: Path | None = None,
    now_fn: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """The complete read-only report over actual persisted artifacts.

    Final outcomes count as observed ONLY when the complete bound evidence
    chain re-validates them (strict orchestration state, the original
    ledger, the recomputed barrier, the hash-bound publication and the
    per-cell final re-verification with independent score recomputation).
    Copied or bare claims remain explicit unverified diagnostics.
    """
    repo_root = Path(repo_root)
    runtime_dir = Path(runtime_dir) if runtime_dir is not None else (
        repo_root / review.RUNTIME_DIR
    )
    if runtime_dir.is_symlink() or not runtime_dir.is_dir():
        raise ComparisonReportError("comparison_runtime_missing")
    # The frozen manifest is REQUIRED: it binds the reported experiment to
    # the reviewed source commit / schedule / roster identities.
    manifest = review.load_frozen_manifest(repo_root, runtime_dir=runtime_dir)
    manifest_path = runtime_dir / review.MANIFEST_FILENAME
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if bindings is None:
        from research.comparison.data import load_task_bindings

        bindings = load_task_bindings()
    # Strict orchestration revalidation first; a failure stays VISIBLE and
    # the rows fall back to the raw (unverified) persisted states.
    orch = None
    orchestration_error: str | None = None
    if provider is not None:
        try:
            orch = load_orchestrator(
                runtime_dir=runtime_dir,
                repo_root=repo_root,
                bindings=bindings,
                provider=provider,
                now_fn=now_fn,
            )
        except OrchestrationError as exc:
            orchestration_error = str(exc)
    if orch is not None:
        cells = list(orch.cells)
    else:
        cells = list(iter_cells(bindings, str(manifest["source_commit"])))
    # The frozen manifest must bind the ACTUAL reviewed pins (protocol,
    # model profile, data, registry, native, source files); a copied or
    # mutually edited manifest never authorizes accepted outcomes.
    manifest_error: str | None = None
    if orch is not None:
        try:
            review.verify_frozen_manifest_binding(
                manifest,
                source_commit=orch.source_commit,
                schedule_identity_value=orch._schedule_identity,
                cells=orch.cells,
                bindings=bindings,
                repo_root=repo_root,
            )
        except ComparisonManifestError as exc:
            manifest_error = str(exc)
    # Ledger accounting: READ-ONLY — never flocked, created or stopped.
    ledger_summary: dict[str, Any] | None = None
    ledger_error: str | None = None
    ledger_view: ReadOnlyComparisonLedger | None = None
    ledger_snapshot: Mapping[str, Any] | None = None
    requests_by_run: dict[str, dict[str, int]] = {}
    ledger_path = runtime_dir / review.LEDGER_FILENAME
    ledger_exists = ledger_path.exists() and not ledger_path.is_symlink()
    if ledger_exists:
        try:
            ledger_view = ReadOnlyComparisonLedger.open(
                ledger_path, cells=cells, now_fn=now_fn
            )
            ledger_snapshot = ledger_view.snapshot()
            ledger_summary = _summarize_ledger(ledger_snapshot, cells)
            for record in ledger_snapshot["requests"]:
                entry = requests_by_run.setdefault(
                    str(record["run_identity"]),
                    {"requests": 0, "charged_tokens": 0},
                )
                entry["requests"] += 1
                entry["charged_tokens"] += int(record["charged_tokens"])
        except ComparisonBudgetError as exc:
            ledger_error = str(exc)
    # Verify the complete final evidence chain before ANY persisted final
    # claim is accepted as an observed outcome. The SHARED verified context
    # (final.verify_final_context) requires the ORIGINAL validated ledger
    # to hold an ENTERED final phase whose barrier_identity equals the
    # recomputed all-27 barrier, plus the revalidated hash-bound
    # publication. A failed context leaves verified_finals EMPTY — the
    # populated barrier/publication never leaks downstream.
    verified_finals: set[str] = set()
    verification_error: str | None = None
    if (
        orch is not None
        and manifest_error is None
        and ledger_view is not None
        and ledger_snapshot is not None
    ):
        from research.comparison.final import (
            FinalPhaseError,
            _validate_final_result,
            verify_final_context,
        )

        if sealed_root is None:
            from research.comparison.data import SEALED_ROOT

            sealed_root = SEALED_ROOT
        try:
            context = verify_final_context(
                orch,
                ledger_snapshot,
                public_root=public_root,
                evaluator_root=evaluator_root,
            )
        except (OrchestrationError, FinalPhaseError) as exc:
            verification_error = str(exc)
        else:
            barrier = context["barrier"]
            publication = context["publication"]
            for cell in cells:
                cell_key = cell.cell_key()
                result_path = (
                    runtime_dir
                    / "final"
                    / cell_key.replace(":", "-")
                    / "final.json"
                )
                try:
                    _validate_final_result(
                        orch,
                        cell_key,
                        result_path,
                        ledger=ledger_view,
                        cell=cell,
                        barrier_entry=barrier["cells"][cell_key],
                        barrier_identity=barrier["barrier_identity"],
                        publication=publication,
                        sealed_root=sealed_root,
                    )
                except FinalPhaseError:
                    continue  # unverified claim stays explicit, not accepted
                verified_finals.add(cell_key)
    rows, outcomes = _build_rows(
        runtime_dir, cells, requests_by_run, verified_finals
    )
    if ledger_view is None and ledger_error is None and not ledger_exists:
        if _execution_evidence_present(runtime_dir, rows):
            # Distinct from an honest not-started matrix: ANY persisted
            # execution evidence (one completed/interrupted configuration
            # in even a NONterminal cell, a terminal cell, a run result,
            # an intent or a final artifact) already proves the experiment
            # was active; a missing ORIGINAL ledger is then an integrity
            # failure with UNKNOWN state, never "not started".
            ledger_error = "ledger_missing_with_execution_evidence"
    proto = canonical_protocol()
    expected_pairs = len(proto.tasks) * len(proto.search_seeds)
    contrasts = {
        name: analyze_contrast(
            outcomes,
            treatment=treatment,
            baseline=baseline,
            expected_pairs=expected_pairs,
        )
        for name, treatment, baseline in CONTRASTS
    }
    observed_cells = sum(1 for o in outcomes.values() if o.observed)
    unverified_final_claims = sum(
        1
        for row in rows
        if row["final_status"] in _OBSERVED_FINAL_STATUSES
        and not row["final_verified"]
    )
    matrix_complete = (
        observed_cells == proto.num_cells
        and unverified_final_claims == 0
        and orchestration_error is None
        and orch is not None
        and ledger_summary is not None
        and all(
            row["terminal_status"] in ("frozen", "failed_no_valid_program")
            for row in rows
        )
    )
    success_only: dict[str, Any] = {}
    for method in proto.methods:
        method_outcomes = [o for o in outcomes.values() if o.method == method]
        successes = [
            o.raw_accuracy
            for o in method_outcomes
            if o.observed and o.raw_accuracy is not None
        ]
        failures = sum(
            1
            for o in method_outcomes
            if o.observed and o.raw_accuracy is None
        )
        success_only[method] = {
            "observed_cells": sum(1 for o in method_outcomes if o.observed),
            "successful": len(successes),
            "failed": failures,
            "unobserved": sum(1 for o in method_outcomes if not o.observed),
            "accuracy_mean": (
                sum(successes) / len(successes) if successes else None
            ),
        }
    conclusion = conclude(contrasts["primary"], matrix_complete=matrix_complete)
    return {
        "schema": FULL_REPORT_SCHEMA,
        "generated_at": float(now_fn()),
        "source_commit": manifest.get("source_commit"),
        "experiment_manifest_sha256": manifest_sha256,
        "manifest_verified": manifest_error is None and orch is not None,
        "manifest_error": manifest_error,
        "orchestration_state_verified": orchestration_error is None and orch is not None,
        "orchestration_error": orchestration_error,
        "final_evidence_verification": {
            "verified_finals": len(verified_finals),
            "unverified_final_claims": unverified_final_claims,
            "error": verification_error,
        },
        "schedule_identity": manifest.get("schedule_identity"),
        "ledger": ledger_summary,
        "ledger_error": ledger_error,
        "matrix": {
            "num_cells": proto.num_cells,
            "terminal_cells": sum(
                1
                for row in rows
                if row["terminal_status"] in ("frozen", "failed_no_valid_program")
            ),
            "observed_final_cells": observed_cells,
            "unverified_final_claims": unverified_final_claims,
            "final_last_finished_at": max(
                (
                    float(row["final_finished_at"])
                    for row in rows
                    if isinstance(row["final_finished_at"], (int, float))
                    and not isinstance(row["final_finished_at"], bool)
                ),
                default=None,
            ),
            "timing_note": (
                "ledger elapsed endpoints are settled model requests/stop "
                "records only; final-phase sandbox execution time is "
                "reported separately via final_last_finished_at"
            ),
            "complete": matrix_complete,
            "rows": rows,
        },
        "success_only_accuracy": success_only,
        "analysis": {
            "contrasts": contrasts,
            "caveats": list(ANALYSIS_CAVEATS),
        },
        "conclusion": conclusion,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    """Human-readable rendering of the full report."""
    lines: list[str] = []
    lines.append("# iStratDE controlled pilot comparison — report")
    lines.append("")
    lines.append(f"- schema: `{report['schema']}`")
    lines.append(f"- source commit: `{report.get('source_commit')}`")
    lines.append(
        f"- orchestration state verified: {report['orchestration_state_verified']}"
    )
    if report.get("orchestration_error"):
        lines.append(f"- ORCHESTRATION ERROR: `{report['orchestration_error']}`")
    verification = report.get("final_evidence_verification") or {}
    lines.append(
        f"- final evidence verified: {verification.get('verified_finals', 0)} cells; "
        f"unverified final claims: {verification.get('unverified_final_claims', 0)}"
    )
    if verification.get("error"):
        lines.append(f"- final verification context: `{verification['error']}`")
    matrix = report["matrix"]
    lines.append(
        f"- matrix: {matrix['terminal_cells']}/{matrix['num_cells']} cells "
        f"terminal, {matrix['observed_final_cells']}/{matrix['num_cells']} "
        f"final outcomes observed, complete={matrix['complete']}"
    )
    lines.append("")
    lines.append("## Ledger costs (actual, failures included)")
    ledger = report.get("ledger")
    if ledger is None:
        lines.append("")
        if report.get("ledger_error"):
            # Missing/unreadable ORIGINAL ledger: the accounting state is
            # UNKNOWN. With execution evidence this is an integrity
            # failure; absence alone never proves the experiment started
            # or did not start.
            lines.append(
                "Ledger missing or unreadable "
                f"(`{report['ledger_error']}`): the original accounting "
                "state is unknown — an integrity failure, not a "
                "not-started experiment."
            )
        else:
            lines.append(
                "No ledger and no execution evidence: the experiment has "
                "not started (honest initialized state)."
            )
    else:
        req = ledger["requests"]
        tok = ledger["tokens"]
        lines.append("")
        lines.append(
            f"- requests: {req['total']} total ({req['main']} main, "
            f"{req['debug']} debug, {req['unknown_usage']} unknown-usage)"
        )
        lines.append(
            f"- tokens charged: {tok['charged_total']} "
            f"(known input {tok['input_known']}, known output "
            f"{tok['output_known']}); unknown usage is never zero-filled"
        )
        lines.append(
            f"- started_at: {ledger['started_at']}, deadline: "
            f"{ledger['deadline_epoch']}, stopped: {ledger['stopped']}"
        )
        lines.append(
            f"- elapsed observed: {ledger['elapsed_seconds_observed']} s"
        )
        lines.append("")
        lines.append("### Per-cell costs")
        lines.append("")
        lines.append("| cell | requests | main | debug | charged tokens | unknown | terminated |")
        lines.append("|---|---|---|---|---|---|---|")
        for cell_key, costs in ledger["cells"].items():
            terminated = costs["terminated"]
            lines.append(
                f"| {cell_key} | {costs['requests']} | {costs['main_requests']} "
                f"| {costs['debug_requests']} | {costs['charged_tokens']} "
                f"| {costs['unknown_usage_requests']} "
                f"| {(terminated or {}).get('reason') or ''} |"
            )
    lines.append("")
    lines.append("## 27-cell matrix")
    lines.append("")
    lines.append(
        "| cell | terminal | validation incumbent | final status | "
        "raw accuracy | utility | configs (done/gate/skip/interrupted) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in matrix["rows"]:
        configs = row["configs"]
        validation = row["validation_incumbent"]
        final_status = row["final_status"]
        if final_status in ("completed", "failed", "no_valid_program") and not row.get(
            "final_verified"
        ):
            final_status = f"{final_status} (UNVERIFIED)"
        lines.append(
            f"| {row['cell_key']} | {row['terminal_status']} "
            f"| {'' if validation is None else f'{validation:.6f}'} "
            f"| {final_status} "
            f"| {'' if row['raw_accuracy'] is None else f'{row['raw_accuracy']:.6f}'} "
            f"| {'' if row['utility'] is None else f'{row['utility']:.6f}'} "
            f"| {configs['completed']}/{configs['gate_rejected']}"
            f"/{configs['not_launched_cell_exhausted']}"
            f"/{configs['launched_interrupted']} |"
        )
    lines.append("")
    lines.append("## Success-only accuracy (failures explicit)")
    lines.append("")
    lines.append("| method | observed | successful | failed | unobserved | accuracy mean |")
    lines.append("|---|---|---|---|---|---|")
    for method, stats in report["success_only_accuracy"].items():
        mean = stats["accuracy_mean"]
        lines.append(
            f"| {method} | {stats['observed_cells']} | {stats['successful']} "
            f"| {stats['failed']} | {stats['unobserved']} "
            f"| {'' if mean is None else f'{mean:.6f}'} |"
        )
    lines.append("")
    lines.append("## Predeclared paired analysis (utility, FR7)")
    lines.append("")
    for name, contrast in report["analysis"]["contrasts"].items():
        lines.append(
            f"- {name}: {contrast['treatment']} - {contrast['baseline']}, "
            f"observed pairs {contrast['observed_pairs']}/"
            f"{contrast['expected_pairs']} (partial={contrast['partial']})"
        )
        estimate = contrast.get("estimate")
        interval = contrast.get("interval")
        if estimate is None:
            lines.append("  - estimate: none (no observed pairs)")
        else:
            lines.append(f"  - task-macroaveraged estimate: {estimate:.6f}")
            if interval is not None:
                lines.append(
                    f"  - percentile {interval['level']:.0%} interval: "
                    f"[{interval['low']:.6f}, {interval['high']:.6f}] "
                    f"({interval['replicates']} within-task paired-seed "
                    f"bootstrap replicates, RNG seed {interval['seed']})"
                )
        if contrast["partial"]:
            lines.append(
                "  - PARTIAL estimate over observed pairs only — the "
                "predeclared full-matrix comparison was NOT completed"
            )
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(f"**{report['conclusion']['conclusion']}** — {report['conclusion']['reason']}")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    for caveat in report["analysis"]["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def write_report_bundle(
    report: Mapping[str, Any],
    output_dir: Path,
    *,
    markdown: str | None = None,
) -> dict[str, str]:
    """Durably write report.json + report.md; returns {name: sha256}.

    Writes go through a fresh temporary file + atomic replace, so a
    pre-existing SYMLINK at the destination is itself replaced — the
    original evidence it points at is never overwritten.
    """
    import os
    import uuid

    from research.comparison.protocol import canonical_json

    output_dir = Path(output_dir)
    if output_dir.is_symlink():
        raise ComparisonReportError("report_output_symlinked")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    payloads = {
        "report.json": (canonical_json(dict(report)) + "\n").encode("utf-8"),
        "report.md": (markdown if markdown is not None else render_markdown(report)).encode("utf-8"),
    }
    for name, data in payloads.items():
        path = output_dir / name
        if path.is_symlink():
            raise ComparisonReportError(f"report_output_symlinked:{name}")
        tmp = output_dir / f".{name}.{uuid.uuid4().hex[:12]}.tmp"
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        written[name] = hashlib.sha256(data).hexdigest()
    return written
