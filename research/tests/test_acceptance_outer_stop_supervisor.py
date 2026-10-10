"""Independent regression: a failed acceptance population must stop promptly."""
from __future__ import annotations

from research.contracts.budget import BudgetError, BudgetLedger
from research.search import acceptance_config as ac
from research.search.acceptance import AcceptanceError, run_acceptance
from research.tests.test_acceptance_controller import FakeAlgo, VECTORS, _fake_child, _repo

def test_failed_first_outer_evaluation_stops_without_telling_partial_fitness(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    algos = []
    calls = []
    def factory():
        algo = FakeAlgo(VECTORS)
        algos.append(algo)
        return algo
    try:
        result = run_acceptance(repo, runtime, algo_factory=factory,
            child_runner=_fake_child(runtime, calls, behaviors={"cfg1": {"status": "failed"}}))
        assert result["verdict"] == "incomplete"
    except AcceptanceError:
        pass
    assert calls == ["cfg0", "cfg1"]
    assert all(algo.told is None for algo in algos)
    assert (runtime / ac.CHECKPOINT_FILENAME).exists()
    with BudgetLedger.open(runtime / ac.LEDGER_FILENAME) as ledger:
        assert ledger.snapshot()["committed"]["requests"] == 2
        assert ledger.snapshot()["stopped"] is not None

def test_stopped_original_ledger_refuses_before_any_child_or_new_budget(tmp_path, monkeypatch):
    repo, runtime = _repo(tmp_path, monkeypatch)
    runtime.mkdir(parents=True, exist_ok=True)
    ledger_path = runtime / ac.LEDGER_FILENAME
    with BudgetLedger.open(ledger_path, limits=ac.acceptance_budget_limits()) as ledger:
        ledger.stop("user_budget_boundary")
    before = ledger_path.read_bytes()
    calls = []
    try:
        run_acceptance(repo, runtime, algo_factory=lambda: FakeAlgo(VECTORS),
            child_runner=_fake_child(runtime, calls))
    except (AcceptanceError, BudgetError):
        pass
    else:
        raise AssertionError("A stopped original acceptance cannot start again")
    assert calls == []
    assert ledger_path.read_bytes() == before
