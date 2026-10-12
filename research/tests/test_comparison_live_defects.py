"""Offline regressions for defects observed in the one supervised pilot.

All data and ledgers below are synthetic. The original stopped run is immutable.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from research.comparison.budget import ComparisonLedger, ReadOnlyComparisonLedger
from research.comparison.final import publish_final_data
from research.evaluator.metrics import accuracy
from research.tests import test_comparison_final as fixtures
from research.tests.test_comparison_budget import CELLS
from research.tests.test_comparison_orchestrate import default_scores, fake_child


def test_advancing_clock_final_phase_round_trips(tmp_path: Path) -> None:
    tick = 1000.0

    def clock() -> float:
        nonlocal tick
        tick += 0.000005
        return tick

    path = tmp_path / "ledger.json"
    with ComparisonLedger.open(path, cells=CELLS, now_fn=clock) as ledger:
        ledger.start_clock()
        ledger.enter_final_phase("a" * 64)
    raw = json.loads(path.read_text())
    event = next(e for e in raw["events"] if e["type"] == "final_phase")
    assert event["at"] == raw["final_phase"]["at"]
    reader = ReadOnlyComparisonLedger.open(path, cells=CELLS)
    assert reader.final_phase == raw["final_phase"]
    with ComparisonLedger.open(path, cells=CELLS, now_fn=clock) as reopened:
        reopened.enter_final_phase("a" * 64)
        assert reopened.final_phase == raw["final_phase"]


def test_trusted_six_decimal_accuracy_acceptance_and_resume(tmp_path: Path, monkeypatch) -> None:
    # The actual triggering ratio was 111/113, which cannot be represented
    # by the evaluator's six-decimal score within the old 1e-9 tolerance.
    labels = [0] * 111 + [1, 1]
    answers = ("id,label\n" + "".join(f"{i},{y}\n" for i, y in enumerate(labels))).encode()
    features = ("id,f1\n" + "".join(f"{i},{i / 200}\n" for i in range(113))).encode()
    monkeypatch.setattr(fixtures, "_FINAL_LABELS_CSV", answers)
    monkeypatch.setattr(fixtures, "_FINAL_FEATURES_CSV", features)
    orch = fixtures._make_repo_orch(tmp_path / "controller", monkeypatch)
    repo = fixtures._synthetic_repo(tmp_path / "repo")
    ledger_path = tmp_path / "ledger.json"

    class RoundedExecutor(fixtures.SyntheticExecutor):
        def execute(self, **kwargs):
            execution = super().execute(**kwargs)
            # Exercise the actual trusted metric implementation, not a
            # replacement fixture with a handpicked exact fraction.
            return replace(execution, score=accuracy(labels, [0] * len(labels)))

    with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        orch.run_schedule(ledger, fake_child(orch, default_scores(orch), [], ledger_path))
    # The scheduler releases its writer when child processes acquire it.
    with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        barrier = orch.verify_final_barrier(ledger)
        publication = publish_final_data(orch, ledger, barrier, now_fn=lambda: 1000.0,
                                         **fixtures._publish_kwargs(repo))
        executor = RoundedExecutor(repo["sealed"], registry_version=publication["registry_version"])
        fixtures._run_final(orch, ledger, barrier, executor, repo=repo,
                            now_fn=lambda: 1000.0)
        assert len(executor.calls) == 27
        for path in (orch.runtime_dir / "final").glob("*/final.json"):
            result = json.loads(path.read_text())
            assert result["raw_score"] == 0.982301
            assert result["recomputed_accuracy"] == 111 / 113
        # Resume revalidates saved results through the same rounded-score
        # contract and must not execute any program a second time.
        fixtures._run_final(orch, ledger, barrier, executor, repo=repo,
                            now_fn=lambda: 1000.0)
        assert len(executor.calls) == 27
