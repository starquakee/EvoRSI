"""Independent offline audit for the iStratDE comparison (EXP-008).

Revalidates the ACTUAL persisted artifacts instead of trusting copied
result fields: manifest/source-commit/schedule binding, original ledger
accounting (full internal validation + roster identity, READ-ONLY),
completed-boundary journal/winner source bytes, per-job trusted
task/split/metric/answer/registry/source/prediction/row/cleanup evidence
(via the shared orchestration verifier), barrier recomputation, the
hash-bound final publication, and every accepted final prediction
independently recomputed against the pinned sealed answers.

The audit NEVER opens the ledger for writing, resubmits or cancels jobs,
activates a phase, publishes data or mutates any original artifact. A
tampered result/prediction, a missing artifact, incomplete execution or
unknown usage stays visible as a failed/incomplete check.

Verdict: FAIL when any integrity check fails; INCOMPLETE when integrity
holds but the matrix was not fully observed (not started, interrupted,
capped or globally stopped); PASS only when every check passes.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from research.comparison import review
from research.comparison.budget import (
    ComparisonBudgetError,
    ReadOnlyComparisonLedger,
)
from research.comparison.orchestrate import OrchestrationError, load_orchestrator
from research.comparison.protocol import canonical_protocol
from research.comparison.specs import CellSpec

AUDIT_SCHEMA = "istratde-comparison-audit.v1"


class ComparisonAuditError(RuntimeError):
    """The audit cannot run at all; the message IS the rule."""


def _check(
    checks: list[dict[str, Any]],
    name: str,
    category: str,
    fn: Callable[[], str],
) -> bool:
    """Run one audit check; failures stay visible, never crash the audit."""
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - audit boundary: record, continue
        checks.append(
            {
                "name": name,
                "category": category,
                "status": "fail",
                "detail": f"{type(exc).__name__}:{exc}",
            }
        )
        return False
    checks.append(
        {"name": name, "category": category, "status": "pass", "detail": detail}
    )
    return True


def _skip(
    checks: list[dict[str, Any]], name: str, category: str, detail: str
) -> None:
    checks.append(
        {"name": name, "category": category, "status": "skip", "detail": detail}
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_evidence(runtime_dir: Path, orch: Any) -> list[str]:
    """Persisted execution evidence proving the experiment was active.

    One COMPLETED (or even launched/interrupted) configuration record —
    including in a NONterminal cell — a terminal cell, a run result, an
    intent or a final artifact is enough. The ORIGINAL ledger is then
    required; its absence is an integrity failure with unknown accounting
    state, never an honest "experiment not started".
    """
    evidence: list[str] = []
    if orch is not None:
        for cell in orch.cells:
            state = orch._cell_state(cell.cell_key())
            if isinstance(state.get("terminal"), dict):
                evidence.append(f"terminal:{cell.cell_key()}")
            configs = state.get("configs")
            if isinstance(configs, dict):
                for key in sorted(configs):
                    record = configs[key]
                    if isinstance(record, dict) and isinstance(
                        record.get("status"), str
                    ):
                        evidence.append(
                            f"config:{cell.cell_key()}:{key}:{record['status']}"
                        )
    for sub in ("runs", "intents", "final"):
        root = runtime_dir / sub
        if root.is_symlink() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                evidence.append(f"{sub}:{path.relative_to(runtime_dir)}")
    return evidence


def _artifact(artifacts: list[dict[str, Any]], path: Path, role: str) -> None:
    artifacts.append(
        {"path": str(path), "role": role, "sha256": _sha256_file(path)}
    )


def audit_experiment(
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
    """Audit the persisted experiment artifacts; returns the audit report."""
    repo_root = Path(repo_root)
    runtime_dir = Path(runtime_dir) if runtime_dir is not None else (
        repo_root / review.RUNTIME_DIR
    )
    if runtime_dir.is_symlink() or not runtime_dir.is_dir():
        raise ComparisonAuditError("comparison_runtime_missing")
    if bindings is None:
        from research.comparison.data import load_task_bindings

        bindings = load_task_bindings()
    if sealed_root is None:
        from research.comparison.data import SEALED_ROOT

        sealed_root = SEALED_ROOT

    checks: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    proto = canonical_protocol()

    # -- orchestration state: full shared revalidation (schedule bytes,
    # vectors, config boundaries, journal/winner bytes, native replay).
    orch = None

    def _load_state() -> str:
        nonlocal orch
        if provider is None:
            raise ComparisonAuditError("audit_provider_required")
        orch = load_orchestrator(
            runtime_dir=runtime_dir,
            repo_root=repo_root,
            bindings=bindings,
            provider=provider,
            now_fn=now_fn,
        )
        return "Orchestrator.load revalidated schedule, all cell states, vectors and winner bytes"

    state_ok = _check(
        checks, "orchestration_state", "integrity", _load_state
    )

    # -- manifest binding: the frozen manifest must match the actual
    # roster AND the actual frozen protocol/model/budget/data/registry/
    # native/source-file pins (offline re-hash of the real files).
    manifest_path = runtime_dir / review.MANIFEST_FILENAME

    def _manifest() -> str:
        if orch is None:
            raise ComparisonAuditError("orchestration_unavailable")
        manifest = review.load_frozen_manifest(repo_root, runtime_dir=runtime_dir)
        review.verify_frozen_manifest_binding(
            manifest,
            source_commit=orch.source_commit,
            schedule_identity_value=orch._schedule_identity,
            cells=orch.cells,
            bindings=bindings,
            repo_root=repo_root,
        )
        _artifact(artifacts, manifest_path, "experiment_manifest")
        pins = manifest.get("pinned_file_sha256", {})
        return (
            f"manifest bound to source {orch.source_commit}, 27/216 roster, "
            f"protocol/model/budget/data/registry/native pins and "
            f"{len(pins)} pinned files"
        )

    _check(checks, "manifest_binding", "integrity", _manifest)

    cells: list[CellSpec] = list(orch.cells) if orch is not None else []

    # -- ledger: full _load validation + roster binding, READ-ONLY.
    ledger_view: ReadOnlyComparisonLedger | None = None
    snapshot: Mapping[str, Any] | None = None
    ledger_path = runtime_dir / review.LEDGER_FILENAME
    ledger_exists = ledger_path.exists() and not ledger_path.is_symlink()
    terminal_count_pre = 0
    if orch is not None:
        terminal_count_pre = sum(
            1
            for cell in orch.cells
            if isinstance(orch._cell_state(cell.cell_key()).get("terminal"), dict)
        )

    def _ledger() -> str:
        nonlocal ledger_view, snapshot
        if not cells:
            raise ComparisonAuditError("orchestration_unavailable")
        ledger_view = ReadOnlyComparisonLedger.open(
            ledger_path, cells=cells, now_fn=now_fn
        )
        snapshot = ledger_view.snapshot()
        _artifact(artifacts, ledger_path, "budget_ledger")
        requests = snapshot["requests"]
        return (
            f"ledger valid: {len(requests)} settled requests, "
            f"{snapshot['global']['committed']['tokens']} tokens committed, "
            f"stopped={ledger_view.stopped}"
        )

    if not ledger_exists:
        # A missing ledger is honest "not started" only when NO execution
        # evidence exists at all; otherwise the experiment provably ran and
        # the missing ORIGINAL ledger is an integrity failure with UNKNOWN
        # accounting state.
        evidence = _execution_evidence(runtime_dir, orch)
        if evidence:
            checks.append(
                {
                    "name": "ledger_accounting",
                    "category": "integrity",
                    "status": "fail",
                    "detail": (
                        f"ledger_missing_with_execution_evidence:"
                        f"{len(evidence)} artifact(s), first: {evidence[0]}"
                    ),
                }
            )
        else:
            _skip(
                checks,
                "ledger_accounting",
                "integrity",
                "ledger_missing:experiment_not_started",
            )
        ledger_ok = False
    else:
        ledger_ok = _check(checks, "ledger_accounting", "integrity", _ledger)

    # -- per-config evidence reconciled against the ORIGINAL ledger.
    def _reconcile() -> str:
        if orch is None or snapshot is None:
            raise ComparisonAuditError("ledger_or_state_unavailable")
        verified = 0
        for cell in orch.cells:
            state = orch._cell_state(cell.cell_key())
            orch._validate_cell_state(cell, state, snapshot)
            verified += 1
        # Every settled request/reservation must belong to a COMPLETED
        # config boundary of its run: a charge without boundary evidence is
        # an integrity failure, never silently absorbed into totals.
        if snapshot["global"]["open_reservations"]:
            raise OrchestrationError("ledger_open_reservations")
        runs_index: dict[str, tuple[str, int]] = {}
        for cell in orch.cells:
            for config_index in range(proto.configs_per_cell):
                run = cell.inner_run(config_index)
                runs_index[run.identity()] = (cell.cell_key(), config_index)
        for record in snapshot["requests"]:
            owner = runs_index.get(record["run_identity"])
            if owner is None:
                raise OrchestrationError("ledger_request_unknown_run")
            cell_key, config_index = owner
            state = orch._cell_state(cell_key)
            configs = state.get("configs")
            record_cfg = (
                configs.get(str(config_index))
                if isinstance(configs, dict)
                else None
            )
            if (
                not isinstance(record_cfg, dict)
                or record_cfg.get("status") != "completed"
            ):
                raise OrchestrationError(
                    f"ledger_request_orphan:{cell_key}:{config_index}"
                )
        schedule_path = runtime_dir / "schedule.json"
        if schedule_path.exists() and not schedule_path.is_symlink():
            _artifact(artifacts, schedule_path, "frozen_schedule")
        for cell in orch.cells:
            cell_dir = runtime_dir / "cells" / cell.cell_key().replace(":", "-")
            _artifact(artifacts, cell_dir / "cell.json", "cell_state")
            winner_path = cell_dir / "winner.py"
            if winner_path.exists() and not winner_path.is_symlink():
                _artifact(artifacts, winner_path, "winner_source")
        return (
            f"all {verified} cell states and "
            f"{len(snapshot['requests'])} settled requests reconciled with "
            "the original ledger"
        )

    if state_ok and ledger_ok:
        _check(checks, "ledger_reconciliation", "integrity", _reconcile)
    else:
        _skip(
            checks,
            "ledger_reconciliation",
            "integrity",
            "requires valid orchestration state and ledger",
        )

    # -- barrier recomputation (read-only) vs the durable final-phase record.
    # Final evidence locations are resolved FIRST so the barrier check can
    # distinguish a legitimate pre-publication boundary (no entered final
    # phase AND no final artifacts) from a rolled-back ledger (final
    # artifacts present but the original ledger has no entered phase).
    barrier: dict[str, Any] | None = None
    final_dir = runtime_dir / "final"
    publication_path = final_dir / "publication.json"
    publication_exists = (
        publication_path.exists() and not publication_path.is_symlink()
    )
    final_results_present = 0
    if orch is not None:
        for cell in orch.cells:
            result_path = (
                final_dir / cell.cell_key().replace(":", "-") / "final.json"
            )
            if result_path.exists() and not result_path.is_symlink():
                final_results_present += 1

    def _barrier() -> str:
        nonlocal barrier
        if orch is None or snapshot is None:
            raise ComparisonAuditError("ledger_or_state_unavailable")
        recomputed = orch.compute_barrier_payload(snapshot)
        final_phase = snapshot["final_phase"]
        if final_phase is None:
            if publication_exists or final_results_present:
                # Final artifacts but NO entered final phase in the
                # ORIGINAL validated ledger: a rolled-back/coherent-but-
                # incomplete ledger, never accepted final evidence.
                raise OrchestrationError(
                    "final_phase_absent_with_final_evidence"
                )
            barrier = recomputed
            return (
                f"barrier {recomputed['barrier_identity'][:16]}... recomputed "
                "over 27 terminal cells; final phase not entered and no "
                "final artifacts (legitimate pre-publication boundary)"
            )
        if final_phase.get("barrier_identity") != recomputed["barrier_identity"]:
            raise OrchestrationError("barrier_identity_mismatch")
        barrier = recomputed
        return (
            f"barrier {recomputed['barrier_identity'][:16]}... recomputed over "
            "27 terminal cells; durable final phase entered and bound"
        )

    terminal_count = terminal_count_pre
    if state_ok and ledger_ok and terminal_count == proto.num_cells:
        _check(checks, "barrier_recomputation", "integrity", _barrier)
    else:
        _skip(
            checks,
            "barrier_recomputation",
            "integrity",
            f"requires all 27 cells terminal (actual: {terminal_count})",
        )

    # -- final publication: re-hash every published file + frozen anchor.
    publication: dict[str, Any] | None = None

    def _publication() -> str:
        nonlocal publication
        if orch is None or barrier is None:
            raise ComparisonAuditError("barrier_unavailable")
        from research.comparison.final import _load_publication_record

        publication = _load_publication_record(
            orch,
            barrier,
            public_root=public_root,
            evaluator_root=evaluator_root,
        )
        _artifact(artifacts, publication_path, "publication_record")
        for rel in sorted(publication["files"]):
            path = Path(str(publication["publication_root"])) / rel
            _artifact(artifacts, path, f"publication:{rel}")
        return (
            f"publication re-hashed and anchored to frozen bindings "
            f"({len(publication['files'])} files)"
        )

    final_phase_entered = (
        snapshot is not None and snapshot.get("final_phase") is not None
    )
    if final_phase_entered and barrier is not None:
        if not publication_exists:
            checks.append(
                {
                    "name": "final_publication",
                    "category": "integrity",
                    "status": "fail",
                    "detail": "final phase entered but publication record missing",
                }
            )
        else:
            _check(checks, "final_publication", "integrity", _publication)
    elif final_phase_entered:
        _skip(
            checks,
            "final_publication",
            "integrity",
            "final phase entered but barrier recomputation failed",
        )
    elif publication_exists or final_results_present:
        checks.append(
            {
                "name": "final_publication",
                "category": "integrity",
                "status": "fail",
                "detail": (
                    "final artifacts present but the ORIGINAL ledger has no "
                    "entered final phase bound to the recomputed barrier"
                ),
            }
        )
    else:
        _skip(
            checks,
            "final_publication",
            "integrity",
            "no final phase entered and no publication present",
        )

    # -- final results: full re-verification incl. INDEPENDENT accuracy
    # recomputation against the pinned sealed answers. Acceptance requires
    # the SHARED verified context (final.verify_final_context): recomputed
    # all-27 barrier + an ENTERED final phase in the ORIGINAL validated
    # ledger bound to that barrier + the revalidated publication. A failed
    # context leaves NO accepted final results downstream.
    final_context: dict[str, Any] | None = None
    if (
        orch is not None
        and snapshot is not None
        and ledger_view is not None
        and barrier is not None
        and publication is not None
    ):
        from research.comparison.final import (
            FinalPhaseError,
            verify_final_context,
        )

        try:
            final_context = verify_final_context(
                orch,
                snapshot,
                public_root=public_root,
                evaluator_root=evaluator_root,
            )
        except (OrchestrationError, FinalPhaseError):
            final_context = None

    def _final_results() -> str:
        if orch is None or ledger_view is None or final_context is None:
            raise ComparisonAuditError("final_context_unavailable")
        from research.comparison.final import _validate_final_result

        verified_barrier = final_context["barrier"]
        verified_publication = final_context["publication"]
        counts = {"completed": 0, "failed": 0, "no_valid_program": 0}
        for cell in orch.cells:
            cell_key = cell.cell_key()
            result_path = final_dir / cell_key.replace(":", "-") / "final.json"
            if not result_path.exists() or result_path.is_symlink():
                continue
            result = _validate_final_result(
                orch,
                cell_key,
                result_path,
                ledger=ledger_view,
                cell=cell,
                barrier_entry=verified_barrier["cells"][cell_key],
                barrier_identity=verified_barrier["barrier_identity"],
                publication=verified_publication,
                sealed_root=sealed_root,
            )
            counts[str(result["status"])] += 1
            cell_final_dir = result_path.parent
            _artifact(artifacts, result_path, "final_result")
            for name, role in (
                ("intent.json", "final_intent"),
                ("prediction.csv", "final_prediction"),
            ):
                path = cell_final_dir / name
                if path.exists() and not path.is_symlink():
                    _artifact(artifacts, path, role)
        return (
            f"final results re-verified with independent accuracy "
            f"recomputation: {counts}"
        )

    if final_context is not None:
        _check(checks, "final_results_recomputed", "integrity", _final_results)
    else:
        _skip(
            checks,
            "final_results_recomputed",
            "integrity",
            "requires the verified final context (recomputed barrier + "
            "entered barrier-bound final phase + revalidated publication)",
        )

    # -- no duplicate final execution: one intent per cell, unique job ids.
    def _no_duplicate_final() -> str:
        if orch is None:
            raise ComparisonAuditError("orchestration_unavailable")
        job_ids: dict[str, str] = {}
        intents = 0
        for cell in orch.cells:
            cell_key = cell.cell_key()
            cell_final = final_dir / cell_key.replace(":", "-")
            if not cell_final.is_dir():
                continue
            intent_files = [
                p for p in cell_final.glob("intent*.json") if not p.is_symlink()
            ]
            if len(intent_files) > 1:
                raise OrchestrationError(f"final_intent_duplicate:{cell_key}")
            intents += len(intent_files)
            result = None
            result_path = cell_final / "final.json"
            if result_path.exists() and not result_path.is_symlink():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                result = payload if isinstance(payload, dict) else None
            if result and result.get("status") in ("completed", "failed"):
                # Unique execution covers failed jobs too, not only scores.
                job_id = result.get("job_id")
                if job_id is None:
                    continue
                if not isinstance(job_id, str) or not job_id:
                    raise OrchestrationError(f"final_job_invalid:{cell_key}")
                if job_id in job_ids:
                    raise OrchestrationError(
                        f"final_job_duplicate:{job_ids[job_id]}:{cell_key}"
                    )
                job_ids[job_id] = cell_key
        return f"{intents} durable intents, {len(job_ids)} unique final job ids"

    _check(
        checks, "no_duplicate_final_execution", "integrity", _no_duplicate_final
    )

    # -- unknown usage must stay visible and globally stopped.
    def _unknown_usage() -> str:
        if snapshot is None or ledger_view is None:
            raise ComparisonAuditError("ledger_unavailable")
        unknown = [
            r for r in snapshot["requests"] if not r["usage_known"]
        ]
        if unknown and not ledger_view.stopped:
            raise ComparisonBudgetError(
                f"unknown_usage_not_stopped:{len(unknown)}"
            )
        return (
            f"{len(unknown)} unknown-usage requests (retained in full; "
            f"ledger stopped={ledger_view.stopped})"
        )

    if ledger_ok:
        _check(checks, "unknown_usage_visible", "integrity", _unknown_usage)
    else:
        _skip(
            checks,
            "unknown_usage_visible",
            "integrity",
            "requires a valid ledger",
        )

    # -- completeness (truthful state, not integrity). final_results_present
    # was computed above alongside the barrier/publication context.

    def _completeness() -> str:
        if terminal_count != proto.num_cells:
            raise OrchestrationError(
                f"matrix_incomplete:{terminal_count}/{proto.num_cells} cells terminal"
            )
        if final_results_present != proto.num_cells:
            raise OrchestrationError(
                f"final_incomplete:{final_results_present}/{proto.num_cells} final results"
            )
        return "all 27 cells terminal with observed final results"

    _check(checks, "matrix_completeness", "completeness", _completeness)

    integrity_failed = any(
        c["category"] == "integrity" and c["status"] == "fail" for c in checks
    )
    completeness_failed = any(
        c["category"] == "completeness" and c["status"] == "fail" for c in checks
    )
    if integrity_failed:
        verdict = "FAIL"
    elif completeness_failed:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"
    return {
        "schema": AUDIT_SCHEMA,
        "generated_at": float(now_fn()),
        "verdict": verdict,
        "checks": checks,
        "artifacts": artifacts,
        "summary": {
            "terminal_cells": terminal_count,
            "final_results": final_results_present,
            "integrity_failures": [
                c["name"]
                for c in checks
                if c["category"] == "integrity" and c["status"] == "fail"
            ],
        },
    }
