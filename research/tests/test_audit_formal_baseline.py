"""Contract tests for research/tools/audit_formal_baseline.py (US-002).

Pins the audit semantics on synthetic fixtures: cached rows are counted but
excluded from actual runs and from cost figures, pre-run rejects carry no
execution fields, duplicate configs collapse for the unique-config count,
operators come only from the real per-run journals, and missing fields or
missing evidence fail closed. A separate read-only integration test re-runs
the audit against the original legacy records and asserts the documented
numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.tools import audit_formal_baseline as afb

LEGACY_ROOT = Path(
    json.loads(afb.DEFAULT_ROOTS_CONFIG.read_text(encoding="utf-8"))["formal_baseline_root"]
)


def make_record(
    tag: str,
    *,
    cfg: dict | None = None,
    cached: bool = False,
    rejected: bool = False,
    score: float | None = 1.5,
    tokens: int = 100,
    wall: float = 10.0,
    drop: tuple[str, ...] = (),
) -> dict:
    record: dict = {
        "gen": 0,
        "tag": tag,
        "theta": [0.1],
        "cfg": cfg if cfg is not None else {"crossover_prob": 0.1, "seed_tag": tag},
        "run_status": "rejected" if rejected else "exit_0",
        "fitness": 10.0 if score is None else score,
        "policy_safety": "empty_parent_selection_weights" if rejected else "ok",
    }
    if not rejected:
        record.update({
            "wall_s": wall,
            "score": score,
            "status_count": {"success": 1} if score is not None else {"code_execution_error": 1},
            "total_tokens": tokens,
            "code_safety": "ok",
        })
    if cached:
        record["cached"] = True
    for field in drop:
        record.pop(field, None)
    return record


def make_journal_node(operators: list[str]) -> str:
    return json.dumps({"step": 0, "operators_used": operators})


def write_run_dir(root: Path, tag: str, operators: list[list[str]]) -> None:
    journal = root / "runs" / tag / "program_ep_0" / "task@1" / "aira_evo" / "checkpoint"
    journal.mkdir(parents=True)
    lines = [make_journal_node(ops) for ops in operators]
    (journal / "journal.jsonl").write_text("\n".join(lines) + "\n")


def write_tree(root: Path, records: list[dict], summary: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    (root / "eval_log.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    if summary is None:
        scored = sum(1 for r in records if r.get("score") is not None)
        unique = len({json.dumps(r["cfg"], sort_keys=True) for r in records})
        best = min(((r["fitness"], r["tag"]) for r in records), default=None)
        summary = {
            "evaluations": len(records),
            "unique_configs": unique,
            "scored": scored,
            "best": {"tag": best[1], "fitness": best[0]} if best else None,
        }
    (root / "summary.json").write_text(json.dumps(summary))
    return root


@pytest.fixture()
def mini_tree(tmp_path: Path) -> Path:
    """2 actual runs, 1 cache hit (duplicate config), 1 pre-run reject."""
    cfg_a = {"crossover_prob": 0.1, "depth": 1}
    cfg_b = {"crossover_prob": 0.2, "depth": 2}
    records = [
        make_record("g00_i00", cfg=cfg_a, tokens=100, wall=10.0, score=1.5),
        make_record("g00_i01", cfg=cfg_b, tokens=200, wall=20.0, score=None),
        make_record("g01_i00", cfg=cfg_a, cached=True, tokens=100, wall=0.5, score=1.5),
        make_record("g01_i01", cfg=cfg_b, rejected=True),
    ]
    root = write_tree(tmp_path / "fb", records)
    write_run_dir(root, "g00_i00", [["draft"], ["debug", "analysis"]])
    write_run_dir(root, "g00_i01", [["draft"]])
    return root


class TestAuditFacts:
    def test_counts_distinguish_cached_rejected_actual(self, mini_tree: Path):
        facts = afb.audit(mini_tree)["facts"]
        assert facts["records"] == 4
        assert facts["unique_configs"] == 2  # duplicate cfg counted once
        assert facts["cached"] == 1
        assert facts["prerun_rejects"] == 1
        assert facts["actual_runs"] == 2
        assert facts["rejected"] == [
            {"tag": "g01_i01", "policy_safety": "empty_parent_selection_weights"}
        ]

    def test_cost_excludes_cache_hits(self, mini_tree: Path):
        facts = afb.audit(mini_tree)["facts"]
        # Only uncached rows count as cost; the cache hit's 100 tokens / 0.5 s
        # are re-counted provenance, not incremental spend.
        assert facts["uncached_tokens"] == 300
        assert facts["uncached_wall_s"] == 30.0
        assert facts["all_row_tokens"] == 400
        assert facts["cached_tokens_recounted"] == 100

    def test_score_counts_split_by_cache_origin(self, mini_tree: Path):
        facts = afb.audit(mini_tree)["facts"]
        assert facts["scored"] == 2
        assert facts["scored_uncached"] == 1
        assert facts["scored_cached"] == 1
        assert facts["status_count_totals"] == {"success": 2, "code_execution_error": 1}

    def test_operators_from_journals_only(self, mini_tree: Path):
        facts = afb.audit(mini_tree)["facts"]
        assert facts["operator_totals"] == {"draft": 2, "debug": 1, "analysis": 1}
        assert facts["journal_nodes"] == 3

    def test_improve_and_crossover_detected_when_present(self, tmp_path: Path):
        records = [make_record("g00_i00")]
        root = write_tree(tmp_path / "fb", records)
        write_run_dir(root, "g00_i00", [["draft"], ["improve"], ["crossover"]])
        report = afb.audit(root)
        assert report["facts"]["operator_totals"]["improve"] == 1
        assert report["facts"]["operator_totals"]["crossover"] == 1
        assert report["checks"]["op_improve"] is False
        assert report["checks"]["op_crossover"] is False


class TestFailClosed:
    def test_missing_required_field_is_problem(self, tmp_path: Path):
        records = [make_record("g00_i00", drop=("policy_safety",))]
        root = write_tree(tmp_path / "fb", records)
        write_run_dir(root, "g00_i00", [["draft"]])
        report = afb.audit(root)
        assert any("policy_safety" in p for p in report["problems"])

    def test_executed_record_missing_tokens_fails(self, tmp_path: Path):
        records = [make_record("g00_i00", drop=("total_tokens",))]
        root = write_tree(tmp_path / "fb", records)
        write_run_dir(root, "g00_i00", [["draft"]])
        report = afb.audit(root)
        assert any("total_tokens" in p for p in report["problems"])

    def test_rejected_record_without_execution_fields_is_clean(self, tmp_path: Path):
        records = [make_record("g00_i00", rejected=True)]
        root = write_tree(tmp_path / "fb", records)
        report = afb.audit(root)
        assert report["problems"] == []

    def test_missing_run_dir_is_problem(self, tmp_path: Path):
        records = [make_record("g00_i00")]
        root = write_tree(tmp_path / "fb", records)  # no run dir written
        report = afb.audit(root)
        assert any("g00_i00" in p and "missing runs/" in p for p in report["problems"])

    def test_orphan_run_dir_is_problem(self, tmp_path: Path):
        records = [make_record("g00_i00")]
        root = write_tree(tmp_path / "fb", records)
        write_run_dir(root, "g00_i00", [["draft"]])
        write_run_dir(root, "g09_i99", [["draft"]])
        report = afb.audit(root)
        assert any("g09_i99" in p for p in report["problems"])

    def test_summary_disagreement_is_problem(self, tmp_path: Path):
        records = [make_record("g00_i00")]
        root = write_tree(tmp_path / "fb", records, summary={
            "evaluations": 99, "unique_configs": 1, "scored": 1,
            "best": {"tag": "g00_i00", "fitness": 1.5},
        })
        write_run_dir(root, "g00_i00", [["draft"]])
        report = afb.audit(root)
        assert any("evaluations" in p for p in report["problems"])

    def test_missing_eval_log_fails_closed(self, tmp_path: Path):
        report = afb.audit(tmp_path / "nonexistent")
        assert report["problems"]
        assert report["facts"] == {}

    def test_check_fails_on_claim_mismatch(self, mini_tree: Path):
        report = afb.audit(mini_tree)
        # mini tree deliberately does not match the documented claims
        assert report["checks"]["records"] is False
        assert report["checks"]["op_draft"] is False


class TestReadOnlyAndReport:
    def test_audit_does_not_modify_source_tree(self, mini_tree: Path):
        before = {
            p: p.read_bytes()
            for p in sorted(mini_tree.rglob("*")) if p.is_file()
        }
        afb.audit(mini_tree)
        after = {
            p: p.read_bytes()
            for p in sorted(mini_tree.rglob("*")) if p.is_file()
        }
        assert before == after

    def test_report_refuses_to_write_inside_source(self, mini_tree: Path):
        report = afb.audit(mini_tree)
        with pytest.raises(ValueError):
            afb.write_report(report, mini_tree / "report.json", mini_tree)

    def test_cli_check_writes_report_outside_source(
        self, mini_tree: Path, tmp_path: Path
    ):
        out = tmp_path / "out" / "report.json"
        code = afb.main(["--root", str(mini_tree), "--check", "--json", str(out)])
        assert code == 1  # mini tree does not match documented claims
        written = json.loads(out.read_text())
        assert written["verdict"] == "FAIL"
        assert "records" in written["failed_checks"]
        assert not (mini_tree / "report.json").exists()


@pytest.mark.skipif(not LEGACY_ROOT.is_dir(), reason="legacy formal_baseline not present")
class TestLegacyIntegration:
    def test_documented_claims_hold_on_original_records(self):
        report = afb.audit(LEGACY_ROOT)
        assert report["problems"] == []
        assert all(report["checks"].values()), [
            k for k, v in report["checks"].items() if not v
        ]
        facts = report["facts"]
        assert facts["records"] == 60
        assert facts["unique_configs"] == 46
        assert facts["cached"] == 14
        assert facts["actual_runs"] == 44
        assert facts["prerun_rejects"] == 2
        assert facts["operator_totals"].get("draft") == 44
        assert facts["operator_totals"].get("debug") == 15
        assert "improve" not in facts["operator_totals"]
        assert "crossover" not in facts["operator_totals"]
        assert facts["uncached_tokens"] == 930962
        assert facts["uncached_wall_s"] == pytest.approx(15345.9, abs=0.05)
        assert facts["all_row_tokens"] == 1315148
        assert facts["scored"] == 54
        assert facts["run_dirs"] == 44

    def test_original_tree_untouched_by_audit(self):
        eval_log = LEGACY_ROOT / "eval_log.jsonl"
        summary = LEGACY_ROOT / "summary.json"
        before = (eval_log.read_bytes(), summary.read_bytes())
        afb.audit(LEGACY_ROOT)
        assert (eval_log.read_bytes(), summary.read_bytes()) == before
