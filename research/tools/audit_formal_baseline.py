"""Independent read-only audit of the legacy formal_baseline experiment (US-002).

Recomputes every headline claim of the 2026-09-30 8-D formal baseline run from
the ORIGINAL legacy records (read-only source root configured in the data file
`research/tools/audit_formal_baseline_roots.json`):

  * 60 eval_log records, 46 unique configs, 14 cache hits, 44 actual Evo runs,
    2 pre-run policy rejects (empty_parent_selection_weights).
  * All actual operators are draft=44 / debug=15 (plus auxiliary
    rich_memory_summary/analysis); no improve/crossover ever fired.
  * Uncached recorded tokens = 930962 and execution wall sum = 15345.9 s.
    The all-row token sum 1315148 is duplicate-inclusive (it re-counts the 14
    cache hits) and is NOT a cost figure.

The source tree is opened strictly read-only: this tool never writes into the
audited directory and refuses to place its report inside it. Missing records,
missing fields, missing per-run journals, or a summary.json that disagrees
with the log are all reported as problems and fail the audit (fail closed).

Source root is read from that JSON data config, so machine-specific absolute
paths stay out of Python source.

Usage:
    <venv-python> research/tools/audit_formal_baseline.py            # print facts
    <venv-python> research/tools/audit_formal_baseline.py --check    # verify claims, exit 1 on failure
    <venv-python> research/tools/audit_formal_baseline.py --check --json reports/formal-baseline-audit/audit-report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS_CONFIG = Path(__file__).resolve().parent / "audit_formal_baseline_roots.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "formal-baseline-audit" / "audit-report.json"

# Fields every eval_log record must carry regardless of outcome.
REQUIRED_FIELDS = ("gen", "tag", "cfg", "run_status", "fitness", "policy_safety")
# Fields every executed (non-rejected) record must carry, even cache hits.
EXECUTED_FIELDS = ("wall_s", "score", "status_count", "total_tokens", "code_safety")
RUN_STATUS_REJECTED = "rejected"
REJECT_POLICY_REASON = "empty_parent_selection_weights"
WALL_TOLERANCE_S = 0.05

# The documented claims under audit (tasks/prd-trustworthy-rsi.md US-002).
EXPECTED_CLAIMS: dict[str, Any] = {
    "records": 60,
    "unique_configs": 46,
    "cached": 14,
    "actual_runs": 44,
    "prerun_rejects": 2,
    "op_draft": 44,
    "op_debug": 15,
    "op_improve": 0,
    "op_crossover": 0,
    "uncached_tokens": 930962,
    "uncached_wall_s": 15345.9,
    "all_row_tokens": 1315148,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_root(config_path: Path = DEFAULT_ROOTS_CONFIG) -> Path:
    """Resolve the audited tree root from the JSON data config (fail closed)."""
    problems: list[str] = []
    if not config_path.is_file():
        raise FileNotFoundError(f"roots config missing: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    raw = data.get("formal_baseline_root")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"roots config {config_path} lacks 'formal_baseline_root'")
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    if not root.is_dir():
        problems.append(f"source root is not a directory: {root}")
    if problems:
        raise FileNotFoundError("; ".join(problems))
    return root


def _canonical_cfg(cfg: Any) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def _journal_path(run_dir: Path) -> Path | None:
    """Locate the aira_evo checkpoint journal inside one run directory."""
    candidates = sorted(run_dir.glob("program_ep_0/*/aira_evo/checkpoint/journal.jsonl"))
    return candidates[0] if candidates else None


def audit(root: Path) -> dict[str, Any]:
    """Recompute all audit facts from the original records (read-only)."""
    problems: list[str] = []
    eval_log = root / "eval_log.jsonl"
    summary_path = root / "summary.json"
    runs_dir = root / "runs"
    for required in (eval_log, summary_path, runs_dir):
        if not required.exists():
            problems.append(f"missing evidence: {required}")
    if problems:
        return {"root": str(root), "problems": problems, "facts": {}, "checks": {}}

    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(eval_log.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"eval_log.jsonl line {lineno}: invalid JSON: {exc}")
            continue
        if not isinstance(record, dict):
            problems.append(f"eval_log.jsonl line {lineno}: not an object")
            continue
        rows.append(record)

    for record in rows:
        tag = record.get("tag", "<unknown>")
        for field in REQUIRED_FIELDS:
            if field not in record:
                problems.append(f"record {tag}: missing required field '{field}'")
        if record.get("run_status") != RUN_STATUS_REJECTED:
            for field in EXECUTED_FIELDS:
                if field not in record:
                    problems.append(
                        f"executed record {tag}: missing field '{field}' (fail closed)"
                    )

    cached_rows = [r for r in rows if r.get("cached")]
    rejected_rows = [r for r in rows if r.get("run_status") == RUN_STATUS_REJECTED]
    actual_rows = [
        r for r in rows
        if not r.get("cached") and r.get("run_status") != RUN_STATUS_REJECTED
    ]
    unique_configs = {_canonical_cfg(r["cfg"]) for r in rows if "cfg" in r}

    uncached_rows = [r for r in rows if not r.get("cached")]
    uncached_tokens = sum(r.get("total_tokens", 0) for r in uncached_rows)
    all_row_tokens = sum(r.get("total_tokens", 0) for r in rows)
    uncached_wall_s = round(sum(r.get("wall_s", 0.0) for r in uncached_rows), 1)

    scored_rows = [r for r in rows if r.get("score") is not None]
    status_totals: dict[str, int] = {}
    for record in rows:
        for name, count in (record.get("status_count") or {}).items():
            status_totals[name] = status_totals.get(name, 0) + count

    # Per-run operator counts from the actual Evo checkpoint journals.
    operator_totals: dict[str, int] = {}
    journal_nodes = 0
    run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []
    run_dir_tags = {p.name for p in run_dirs}
    actual_tags = {r["tag"] for r in actual_rows if "tag" in r}
    for tag in sorted(actual_tags - run_dir_tags):
        problems.append(f"actual run {tag}: missing runs/ directory and journal")
    for tag in sorted(run_dir_tags - actual_tags):
        problems.append(f"runs/{tag}: directory without a matching actual (non-cached, non-rejected) log record")
    for run_dir in run_dirs:
        if run_dir.name not in actual_tags:
            continue
        journal = _journal_path(run_dir)
        if journal is None:
            problems.append(f"run {run_dir.name}: no aira_evo checkpoint journal.jsonl")
            continue
        for line in journal.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            node = json.loads(line)
            journal_nodes += 1
            for op in node.get("operators_used") or []:
                operator_totals[op] = operator_totals.get(op, 0) + 1

    # Cross-check summary.json against the log (inconsistent evidence fails).
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_checks = {
        "evaluations": len(rows),
        "unique_configs": len(unique_configs),
        "scored": len(scored_rows),
    }
    for key, computed in summary_checks.items():
        if summary.get(key) != computed:
            problems.append(
                f"summary.json '{key}'={summary.get(key)} disagrees with recomputed {computed}"
            )
    fitness_rows = [(r["fitness"], r["tag"]) for r in rows if "fitness" in r and "tag" in r]
    if fitness_rows:
        best_fitness, best_tag = min(fitness_rows)
        best = summary.get("best") or {}
        if best.get("tag") != best_tag or best.get("fitness") != best_fitness:
            problems.append(
                f"summary.json best ({best.get('tag')},{best.get('fitness')}) disagrees "
                f"with recomputed minimum ({best_tag},{best_fitness})"
            )

    facts: dict[str, Any] = {
        "records": len(rows),
        "unique_configs": len(unique_configs),
        "cached": len(cached_rows),
        "prerun_rejects": len(rejected_rows),
        "actual_runs": len(actual_rows),
        "cached_tags": sorted(r["tag"] for r in cached_rows),
        "rejected": [
            {"tag": r["tag"], "policy_safety": r.get("policy_safety")}
            for r in rejected_rows
        ],
        "scored": len(scored_rows),
        "scored_uncached": sum(1 for r in scored_rows if not r.get("cached")),
        "scored_cached": sum(1 for r in scored_rows if r.get("cached")),
        "status_count_totals": status_totals,
        "operator_totals": operator_totals,
        "journal_nodes": journal_nodes,
        "uncached_tokens": uncached_tokens,
        "cached_tokens_recounted": all_row_tokens - uncached_tokens,
        "all_row_tokens": all_row_tokens,
        "uncached_wall_s": uncached_wall_s,
        "run_dirs": len(run_dirs),
        "source_hashes": {
            "eval_log.jsonl": _sha256(eval_log),
            "summary.json": _sha256(summary_path),
        },
    }

    checks = {
        "records": len(rows) == EXPECTED_CLAIMS["records"],
        "unique_configs": len(unique_configs) == EXPECTED_CLAIMS["unique_configs"],
        "cached": len(cached_rows) == EXPECTED_CLAIMS["cached"],
        "actual_runs": len(actual_rows) == EXPECTED_CLAIMS["actual_runs"],
        "prerun_rejects": len(rejected_rows) == EXPECTED_CLAIMS["prerun_rejects"],
        "op_draft": operator_totals.get("draft", 0) == EXPECTED_CLAIMS["op_draft"],
        "op_debug": operator_totals.get("debug", 0) == EXPECTED_CLAIMS["op_debug"],
        "op_improve": operator_totals.get("improve", 0) == EXPECTED_CLAIMS["op_improve"],
        "op_crossover": operator_totals.get("crossover", 0) == EXPECTED_CLAIMS["op_crossover"],
        "uncached_tokens": uncached_tokens == EXPECTED_CLAIMS["uncached_tokens"],
        "uncached_wall_s": abs(uncached_wall_s - EXPECTED_CLAIMS["uncached_wall_s"]) <= WALL_TOLERANCE_S,
        "all_row_tokens": all_row_tokens == EXPECTED_CLAIMS["all_row_tokens"],
    }

    return {
        "root": str(root),
        "expected_claims": EXPECTED_CLAIMS,
        "facts": facts,
        "checks": checks,
        "problems": problems,
    }


def write_report(report: dict[str, Any], output: Path, source_root: Path) -> Path:
    """Write the audit report; refuse to write inside the audited tree."""
    resolved = output.resolve()
    if resolved == source_root or source_root in resolved.parents:
        raise ValueError(f"refusing to write report inside audited tree: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None,
                        help="formal_baseline tree to audit (default: roots config)")
    parser.add_argument("--check", action="store_true",
                        help="verify documented claims; exit 1 on any mismatch/problem")
    parser.add_argument("--json", type=Path, default=None,
                        help=f"write full report JSON (default with --check: {DEFAULT_REPORT_PATH})")
    args = parser.parse_args(argv)

    root = args.root if args.root is not None else load_source_root()
    root = root.resolve()
    report = audit(root)

    failed = [name for name, ok in report["checks"].items() if not ok]
    report["verdict"] = "PASS" if not failed and not report["problems"] else "FAIL"
    report["failed_checks"] = failed

    if args.json or args.check:
        output = args.json or DEFAULT_REPORT_PATH
        written = write_report(report, output, root)
        print(f"report written: {written}")

    facts = report["facts"]
    if facts:
        print(f"root: {report['root']}")
        print(f"records={facts['records']} unique_configs={facts['unique_configs']} "
              f"cached={facts['cached']} actual_runs={facts['actual_runs']} "
              f"prerun_rejects={facts['prerun_rejects']}")
        print(f"operators={facts['operator_totals']}")
        print(f"uncached_tokens={facts['uncached_tokens']} "
              f"all_row_tokens={facts['all_row_tokens']} "
              f"(duplicate-inclusive, +{facts['cached_tokens_recounted']} recounted cache hits) "
              f"uncached_wall_s={facts['uncached_wall_s']}")
        print(f"scored={facts['scored']} (uncached={facts['scored_uncached']}, "
              f"cached={facts['scored_cached']}) status_counts={facts['status_count_totals']}")
    for problem in report["problems"]:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    if args.check:
        for name, ok in report["checks"].items():
            print(f"check {name}: {'ok' if ok else 'MISMATCH'}")
        print(f"verdict: {report['verdict']}")
        return 0 if report["verdict"] == "PASS" else 1
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
