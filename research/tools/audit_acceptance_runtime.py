"""Acceptance runtime evidence + budget ledger audit (US-010).

Reproducible, read-only audit of the two US-009 acceptance rounds:

1. Every file listed in `runtime_evidence_sha256` of
   `reports/us009-real-acceptance.json` (round 1) and
   `reports/us009-round2-acceptance.json` (round 2) is re-hashed and compared
   with the recorded digest; a missing file or mismatch fails the audit
   (fail closed — the acceptance runtime trees are immutable evidence).
2. Each round's `budget-ledger.json` is parsed and checked against the
   acceptance pins (200000 tokens / 30 requests / 5400 s per round): schema
   v2, finite integer committed totals within limits, no unresolved
   reservations, and round-consistent stop state (round 1 stopped at the
   graceful stop, round 2 completed within limits).

Usage:
    .venv-research/bin/python research/tools/audit_acceptance_runtime.py [--json OUT]

Exit 0 and prints AUDIT OK on success; exit 1 with machine-readable issues
otherwise. Never modifies any file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

ROUNDS = [
    ("round1", REPO_ROOT / "reports/us009-real-acceptance.json"),
    ("round2", REPO_ROOT / "reports/us009-round2-acceptance.json"),
]

PIN_TOKENS = 200000
PIN_REQUESTS = 30
PIN_SECONDS = 5400.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_finite_int(value: object, what: str, issues: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        issues.append(f"ledger_{what}_not_nonnegative_int:{value!r}")
        return 0
    return value


def audit_round(round_id: str, report_path: Path, issues: list[str]) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = report.get("runtime_evidence_sha256")
    if not isinstance(evidence, dict) or not evidence:
        issues.append(f"{round_id}:runtime_evidence_sha256_missing")
        evidence = {}
    checked = 0
    for rel, recorded in sorted(evidence.items()):
        target = REPO_ROOT / rel
        if not target.is_file():
            issues.append(f"{round_id}:evidence_missing:{rel}")
            continue
        actual = _sha256(target)
        checked += 1
        if actual != recorded:
            issues.append(f"{round_id}:evidence_hash_mismatch:{rel}")

    ledger_rel = f".runtime/acceptance-us009{'-round2' if round_id == 'round2' else ''}/budget-ledger.json"
    ledger_path = REPO_ROOT / ledger_rel
    ledger_summary: dict = {"path": ledger_rel}
    if not ledger_path.is_file():
        issues.append(f"{round_id}:ledger_missing:{ledger_rel}")
    else:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("schema_version") != 2:
            issues.append(f"{round_id}:ledger_schema_unexpected:{ledger.get('schema_version')!r}")
        committed = ledger.get("committed") or {}
        limits = ledger.get("limits") or {}
        requests = _check_finite_int(committed.get("requests"), "requests", issues)
        tokens = _check_finite_int(committed.get("tokens"), "tokens", issues)
        limit_tokens = _check_finite_int(limits.get("max_tokens"), "limit_tokens", issues)
        limit_requests = _check_finite_int(limits.get("max_requests"), "limit_requests", issues)
        limit_seconds = limits.get("max_elapsed_seconds")
        if not isinstance(limit_seconds, (int, float)) or isinstance(limit_seconds, bool) or not math.isfinite(limit_seconds):
            issues.append(f"{round_id}:ledger_limit_seconds_invalid:{limit_seconds!r}")
            limit_seconds = 0.0
        if limit_tokens > PIN_TOKENS or limit_requests > PIN_REQUESTS or float(limit_seconds) > PIN_SECONDS:
            issues.append(f"{round_id}:ledger_limits_exceed_pins:{limits!r}")
        if requests > limit_requests or tokens > limit_tokens:
            issues.append(f"{round_id}:ledger_committed_exceeds_limits:{committed!r}>{limits!r}")
        reservations = ledger.get("reservations")
        if not isinstance(reservations, list):
            issues.append(f"{round_id}:ledger_reservations_missing")
            reservations = []
        # The ledger pops reservations on settle (budget.py): any remaining
        # entry is an OPEN reservation with unresolved provider usage.
        if reservations:
            issues.append(f"{round_id}:ledger_unresolved_reservations:{len(reservations)}")
        # A stopped ledger can never match a COMPLETE acceptance report.
        verdict = str(report.get("verdict", "")).upper()
        if verdict == "COMPLETE" and ledger.get("stopped") is not None:
            issues.append(f"{round_id}:stopped_ledger_matches_complete_report")
        ledger_summary.update(
            committed_requests=requests,
            committed_tokens=tokens,
            stopped=ledger.get("stopped") is not None,
        )
    return {"round": round_id, "evidence_files_verified": checked, "ledger": ledger_summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_out", default=None,
                        help="optional path to write the JSON audit report")
    args = parser.parse_args()

    issues: list[str] = []
    rounds = []
    for round_id, report_path in ROUNDS:
        if not report_path.is_file():
            issues.append(f"{round_id}:report_missing:{report_path.name}")
            continue
        rounds.append(audit_round(round_id, report_path, issues))

    totals = {
        "requests": sum(r["ledger"].get("committed_requests", 0) for r in rounds),
        "tokens": sum(r["ledger"].get("committed_tokens", 0) for r in rounds),
    }
    verdict = {"verdict": "AUDIT OK" if not issues else "AUDIT FAILED",
               "rounds": rounds, "aggregate_committed": totals, "issues": issues}
    text = json.dumps(verdict, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    if issues:
        print(text, file=sys.stderr)
        return 1
    print(f"AUDIT OK: {len(rounds)} rounds, "
          f"{sum(r['evidence_files_verified'] for r in rounds)} evidence files verified, "
          f"aggregate committed {totals['requests']} requests / {totals['tokens']} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
