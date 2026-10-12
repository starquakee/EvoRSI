"""Command surfaces for the iStratDE comparison (EXP-007 wiring).

- ``prepare``: OFFLINE. Freezes the full immutable experiment manifest at
  the exact clean HEAD and initializes the frozen schedule/theta-vector
  state. No ledger, no credentials, no model calls.
- ``preflight``: OFFLINE. Re-verifies every pre-live gate and persists
  evidence; the supervisor review gate is reported, never opened.
- ``run`` / ``resume``: LIVE. Refuse unless the ignored supervisor review
  (bound to the exact clean HEAD and the frozen manifest) verifies. Run
  creates the one ledger and starts the immutable clock; resume requires
  the existing ledger and original clock. Both execute the frozen
  schedule, then the barrier, final publication and final phase under the
  ORIGINAL deadline.
- ``report``: OFFLINE read-only full report bundle (EXP-008): the complete
  27-cell matrix with actual ledger costs, the predeclared paired
  bootstrap analysis, rendered plots and the independent audit. Never
  opens the ledger for writing, never touches credentials/network/stack.

Every refusal exits 2 with a machine-readable reason on stderr and creates
no artifacts.

Entry point: ``python -m research.comparison.cli <command>``.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

COMMANDS: tuple[str, ...] = ("prepare", "preflight", "run", "resume", "report")

#: Which story owns each command's behavior (documentation only).
COMMAND_STORIES = {
    "prepare": "EXP-007",
    "preflight": "EXP-007",
    "run": "EXP-007 (live, review-gated)",
    "resume": "EXP-007 (live, review-gated)",
    "report": "EXP-007 (matrix); EXP-008 (analysis/plots)",
}


class ComparisonRefused(RuntimeError):
    """A comparison command is refused; the message IS the rule."""


def _cmd_prepare(args: argparse.Namespace) -> None:
    from research.comparison import preflight, review
    from research.comparison.protocol import canonical_json

    repo_root = args.repo_root.resolve()
    evidence = preflight.prepare_experiment(repo_root)
    evidence["invocation"] = _invocation(args)
    preflight._write_durable(
        repo_root / review.RUNTIME_DIR / "prepare-evidence.json",
        canonical_json(evidence) + "\n",
    )
    _emit(evidence, args.output)


def _cmd_preflight(args: argparse.Namespace) -> None:
    from research.comparison import preflight

    evidence = preflight.run_preflight(
        args.repo_root.resolve(), check_stack=not args.no_stack
    )
    evidence["invocation"] = _invocation(args)
    _emit(evidence, args.output)
    if evidence["verdict"] != "PASS":
        raise ComparisonRefused(
            "preflight_gates_failed:"
            + ",".join(
                g["name"] for g in evidence["gates"] if g["status"] != "pass"
            )
        )


def _cmd_run(args: argparse.Namespace) -> None:
    _execute_live(args, resume=False)


def _cmd_resume(args: argparse.Namespace) -> None:
    _execute_live(args, resume=True)


def _execute_live(args: argparse.Namespace, *, resume: bool) -> None:
    """Search schedule -> barrier -> final publication -> final phase.

    The review gate runs FIRST; without the accepted supervisor review this
    makes no ledger, reads no credential and submits nothing. The child
    runner and the final executor both receive the ORIGINAL ledger deadline
    (never a restarted duration).
    """
    from research.comparison import (
        budget,
        final,
        native,
        orchestrate,
        phase,
        preflight,
        review,
    )
    from research.comparison.data import load_task_bindings

    repo_root = args.repo_root.resolve()
    # Gate BEFORE any ledger/auth/model/sandbox activity.
    review.check_review_acceptance(repo_root)
    bindings = load_task_bindings()
    provider = native.SubprocessNativeProvider(repo_root)
    orch = orchestrate.load_orchestrator(
        repo_root=repo_root, bindings=bindings, provider=provider
    )
    runtime_dir = repo_root / review.RUNTIME_DIR
    ledger_path = runtime_dir / review.LEDGER_FILENAME
    if ledger_path.is_symlink():
        raise ComparisonRefused("comparison_ledger_symlinked")
    if not resume and ledger_path.exists():
        raise ComparisonRefused("experiment_already_started:use_resume")
    if resume and not ledger_path.exists():
        raise ComparisonRefused("resume_requires_started_experiment")
    if not resume:
        # Recheck the ACTUAL running-stack identity (images incl. frozen
        # digests, real mount sources, active phase) against the reviewed
        # manifest BEFORE the one experiment ledger exists: search jobs
        # need the search registry.
        stack = preflight.inspect_runtime_stack(
            repo_root, expected_images=_manifest_image_baseline(repo_root)
        )
        if stack["phase"] != phase.PHASE_SEARCH:
            raise ComparisonRefused("stack_phase_mismatch:search_required")
    ledger = budget.ComparisonLedger.open(ledger_path, cells=orch.cells)
    try:
        if not resume:
            ledger.start_clock()
        elif ledger.started_at is None:
            raise ComparisonRefused("resume_requires_started_clock")
        elif ledger.final_phase is None:
            # Resumed search: the dispatcher must still be in search phase,
            # with the actual stack identity matching the frozen baseline.
            stack = preflight.inspect_runtime_stack(
                repo_root, expected_images=_manifest_image_baseline(repo_root)
            )
            if stack["phase"] != phase.PHASE_SEARCH:
                ledger.stop("stack_phase_mismatch:search_required")
                raise ComparisonRefused("stack_phase_mismatch:search_required")
        else:
            # Resumed FINAL phase: recheck the SAME reviewed runtime identity
            # (frozen image digests, real mount sources) BEFORE any final
            # execution. The current registry phase itself is not required
            # here: activate_final_phase below performs the explicit
            # dispatcher-only final transition idempotently; an unknown
            # registry already fails inside the inspection.
            try:
                preflight.inspect_runtime_stack(
                    repo_root, expected_images=_manifest_image_baseline(repo_root)
                )
            except review.ComparisonManifestError as exc:
                ledger.stop(f"stack_identity_mismatch:{exc}")
                raise ComparisonRefused(f"stack_identity_mismatch:{exc}") from exc
        deadline = ledger.deadline_epoch  # ORIGINAL deadline, never restarted
        child = orchestrate.evo_child_runner(
            repo_root, deadline_fn=lambda: deadline
        )
        terminal = orch.run_schedule(ledger, child)  # closes its handle
    finally:
        if not ledger.closed:
            ledger.close()
    ledger = budget.ComparisonLedger.open(ledger_path, cells=orch.cells)
    try:
        barrier = orch.verify_final_barrier(ledger)
        publication = final.publish_final_data(orch, ledger, barrier)
        # SERVER-SIDE phase activation (the dispatcher loaded its registry
        # once at startup): switch to the hash-bound phase registry while
        # idle, BEFORE any final submission; failures durably stop the
        # original ledger. Idempotent on resume.
        registry_rel = str(publication["registry_file"])
        registry_sha = str(publication["files"][registry_rel])
        phase.activate_final_phase(
            repo_root,
            expected_registry_sha256=registry_sha,
            deadline_epoch=deadline,
            on_failure=lambda reason: ledger.stop(
                f"phase_activation_failed:{reason}"
            ),
        )
        executor = final.sandbox_final_executor(
            repo_root,
            publication,
            state_dir=runtime_dir / "final" / "executor",
            deadline_fn=lambda: deadline,
        )
        results = final.run_final_phase(orch, ledger, barrier, executor)
    finally:
        if not ledger.closed:
            ledger.close()
    _emit(
        {
            "terminal_cells": len(terminal),
            "barrier_identity": barrier["barrier_identity"],
            "publication": publication.get("publication_root"),
            "final_results": len(results),
        },
        args.output,
    )


def _cmd_report(args: argparse.Namespace) -> None:
    """EXP-008: full read-only report bundle (JSON + Markdown + plots +
    independent audit). Never creates/opens the live ledger for writing,
    never touches credentials, the network or the stack."""
    from research.comparison import audit, report, review
    from research.comparison.native import SubprocessNativeProvider

    repo_root = args.repo_root.resolve()
    runtime_dir = repo_root / review.RUNTIME_DIR
    provider = SubprocessNativeProvider(repo_root)
    full = report.build_full_report(
        repo_root, provider=provider, runtime_dir=runtime_dir
    )
    audit_report = audit.audit_experiment(
        repo_root, provider=provider, runtime_dir=runtime_dir
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        # Outside the experiment runtime: a report/ entry inside it would
        # trip the pre-live re-preparation freshness gate. Still ignored.
        else repo_root / ".runtime" / "comparison-reports"
    )
    written = report.write_report_bundle(full, output_dir)
    if not args.no_plots:
        from research.comparison import plots

        written.update(
            {
                f"plots/{name}": digest
                for name, digest in plots.render_report_plots(
                    full, output_dir / "plots"
                ).items()
            }
        )
    from research.comparison.protocol import canonical_json

    audit_text = canonical_json(audit_report) + "\n"
    audit_path = output_dir / "audit.json"
    audit_path.write_text(audit_text, encoding="utf-8")
    written["audit.json"] = hashlib.sha256(audit_text.encode()).hexdigest()
    _emit(
        {
            "schema": "istratde-comparison-report-bundle.v1",
            "output_dir": str(output_dir),
            "files": written,
            "matrix_complete": full["matrix"]["complete"],
            "conclusion": full["conclusion"],
            "audit_verdict": audit_report["verdict"],
        },
        args.output,
    )


def _manifest_image_baseline(repo_root: Path) -> Mapping[str, str] | None:
    """Frozen stack image digests from the experiment manifest (if bound)."""
    from research.comparison import review

    manifest = review.load_frozen_manifest(repo_root)
    images = (manifest.get("stack_images") or {}).get("image_ids")
    return images if isinstance(images, dict) else None


def _invocation(args: argparse.Namespace) -> dict[str, Any]:
    return {"argv": sys.argv, "python": sys.executable, "command": args.command}


def _emit(payload: Mapping[str, Any], output: Path | None) -> None:
    from research.comparison.protocol import canonical_json

    text = canonical_json(dict(payload)) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research.comparison.cli",
        description=(
            "EvoRSI iStratDE controlled pilot comparison. prepare/preflight/"
            "report are offline; run/resume are review-gated live commands."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        sub = subparsers.add_parser(
            command,
            help=f"({COMMAND_STORIES[command]})",
        )
        sub.add_argument(
            "--repo-root",
            type=Path,
            default=Path.cwd(),
            help="repository root (default: current directory)",
        )
        sub.add_argument(
            "--output",
            type=Path,
            default=None,
            help="write the JSON evidence/report here instead of stdout",
        )
    subparsers.choices["preflight"].add_argument(
        "--no-stack",
        action="store_true",
        help="skip the docker compose mount-contract gate",
    )
    subparsers.choices["report"].add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="report bundle directory (default: .runtime/comparison-reports)",
    )
    subparsers.choices["report"].add_argument(
        "--no-plots",
        action="store_true",
        help="skip figure rendering",
    )
    handlers: dict[str, Callable[[argparse.Namespace], None]] = {
        "prepare": _cmd_prepare,
        "preflight": _cmd_preflight,
        "run": _cmd_run,
        "resume": _cmd_resume,
        "report": _cmd_report,
    }
    for command, handler in handlers.items():
        subparsers.choices[command].set_defaults(handler=handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one comparison command; refusals exit 2 with the reason."""
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except ComparisonRefused as exc:
        print(f"refused:{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, never crash
        print(f"failed:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
