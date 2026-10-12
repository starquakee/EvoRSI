"""Independent budget regression checks; offline fixtures only."""
from __future__ import annotations
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
import pytest
from research.comparison.budget import (
    ComparisonBudgetError, ComparisonLedger, ProviderUsage,
    guarded_comparison_attempt,
)
from research.comparison.specs import CellSpec, iter_cells
from research.tests.test_comparison_specs import make_bindings

def roster(commit: str = "a" * 40) -> tuple[CellSpec, ...]:
    return iter_cells(make_bindings(), commit)

def test_identity_failure_survives_correct_reopen(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    with ComparisonLedger.open(path, cells=roster()) as ledger:
        ledger.start_clock()
    with pytest.raises(ComparisonBudgetError):
        ComparisonLedger.open(path, cells=roster("b" * 40))
    with ComparisonLedger.open(path, cells=roster()) as ledger:
        assert ledger.stopped, "Source mismatch must persist global stop"

@pytest.mark.parametrize("field", ["source_commit", "protocol_identity", "cell_key", "run_key", "run_cell"])
def test_reopen_rejects_inconsistent_identity_metadata(tmp_path: Path, field: str) -> None:
    path = tmp_path / "budget.json"
    with ComparisonLedger.open(path, cells=roster()):
        pass
    data = json.loads(path.read_text())
    if field in ("source_commit", "protocol_identity"):
        data["experiment"][field] = "b" * (40 if field == "source_commit" else 64)
    elif field == "cell_key":
        data["cells"][next(iter(data["cells"]))][field] = "wdbc:999:FOREIGN"
    elif field == "run_key":
        data["runs"][next(iter(data["runs"]))][field] = "wdbc:999:FOREIGN:0"
    else:
        first = next(iter(data["runs"].values()))
        first["cell_identity"] = next(k for k in data["cells"] if k != first["cell_identity"])
    path.write_text(json.dumps(data))
    with pytest.raises(ComparisonBudgetError):
        with ComparisonLedger.open(path, cells=roster()):
            pass

def test_invalid_provider_identity_stops_after_transport(tmp_path: Path) -> None:
    cells = roster()
    with ComparisonLedger.open(tmp_path / "budget.json", cells=cells) as ledger:
        ledger.start_clock()
        with pytest.raises(ComparisonBudgetError):
            guarded_comparison_attempt(ledger, cells[0].inner_run(0), "main", lambda _: {}, None,
                estimated_input_tokens=100, usage_fn=lambda _: ProviderUsage(10, 5),
                provider_request_id_fn=lambda _: "")
        assert ledger.stopped, "Post-transport accounting failure must stop further calls"

def test_zero_reported_usage_can_be_reopened(tmp_path: Path) -> None:
    cells = roster()
    path = tmp_path / "budget.json"
    with ComparisonLedger.open(path, cells=cells) as ledger:
        ledger.start_clock()
        res = ledger.reserve(cells[0].inner_run(0), "main", estimated_input_tokens=100)
        ledger.commit(res.request_id, ProviderUsage(0, 0), provider_request_id="provider-zero")
    with ComparisonLedger.open(path, cells=cells) as ledger:
        assert not ledger.stopped

def test_recorded_overage_reopens_as_stopped_not_corrupt(tmp_path: Path) -> None:
    cells = roster()
    path = tmp_path / "budget.json"
    with ComparisonLedger.open(path, cells=cells) as ledger:
        ledger.start_clock()
        res = ledger.reserve(cells[0].inner_run(0), "main", estimated_input_tokens=191808)
        ledger.commit(res.request_id, ProviderUsage(191809, 8192), provider_request_id="provider-overage")
        assert ledger.stopped
    with ComparisonLedger.open(path, cells=cells) as ledger:
        assert ledger.stopped
    assert json.loads(path.read_text())["global_committed"]["tokens"] == 200001

def test_guard_serializes_whole_provider_attempts(tmp_path: Path) -> None:
    cells = roster()
    active = 0
    maximum = 0
    lock = threading.Lock()
    start = threading.Barrier(2)
    def transport(_: Any) -> dict[str, Any]:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.08)
        with lock:
            active -= 1
        return {}
    with ComparisonLedger.open(tmp_path / "budget.json", cells=cells) as ledger:
        ledger.start_clock()
        def attempt(index: int) -> Any:
            start.wait(timeout=5)
            return guarded_comparison_attempt(ledger, cells[index].inner_run(0), "main", transport, None,
                estimated_input_tokens=100, usage_fn=lambda _: ProviderUsage(10, 5))
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(attempt, [0, 1]))
    assert maximum == 1, "The protocol requires serial model execution, not just serialized reservations"

def test_separate_process_cannot_open_owned_ledger(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    child = """from pathlib import Path
import sys
from research.comparison.budget import ComparisonLedger, ComparisonLedgerLocked
from research.tests.test_comparison_budget_supervisor import roster
try:
    ledger = ComparisonLedger.open(Path(sys.argv[1]), cells=roster())
except ComparisonLedgerLocked:
    sys.exit(0)
else:
    ledger.close()
    sys.exit(7)
"""
    with ComparisonLedger.open(path, cells=roster()):
        result = subprocess.run([sys.executable, "-c", child, str(path)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_abrupt_process_exit_preserves_charged_reservation(tmp_path: Path) -> None:
    path = tmp_path / "crash.json"
    child = """import os, sys
from pathlib import Path
from research.comparison.budget import ComparisonLedger
from research.tests.test_comparison_budget_supervisor import roster
cells = roster()
ledger = ComparisonLedger.open(Path(sys.argv[1]), cells=cells)
ledger.start_clock()
ledger.reserve(cells[0].inner_run(0), "main", estimated_input_tokens=100)
os._exit(23)
"""
    result = subprocess.run([sys.executable, "-c", child, str(path)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 23, result.stderr
    before = json.loads(path.read_text())
    assert len(before["reservations"]) == 1
    assert before["reservations"][0]["tokens"] == 8292
    cells = roster()
    with ComparisonLedger.open(path, cells=cells) as ledger:
        assert ledger.stopped
        with pytest.raises(ComparisonBudgetError):
            ledger.reserve(cells[1].inner_run(0), "main", estimated_input_tokens=100)
    after = json.loads(path.read_text())
    assert after["reservations"] == before["reservations"]
    assert after["started_at"] == before["started_at"]

def test_atomic_replace_syncs_parent_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import stat
    original = os.fsync
    directory_syncs: list[bool] = []
    def observe(fd: int) -> None:
        directory_syncs.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        original(fd)
    monkeypatch.setattr(os, "fsync", observe)
    with ComparisonLedger.open(tmp_path / "durable.json", cells=roster()) as ledger:
        directory_syncs.clear()
        ledger.start_clock()
    assert any(directory_syncs), "Durable atomic replace must sync the parent directory after rename"


@pytest.mark.parametrize("outcome", ["overage", "unknown"])
def test_terminal_usage_evidence_cannot_resume_when_stop_flag_missing(tmp_path: Path, outcome: str) -> None:
    path = tmp_path / "terminal.json"
    cells = roster()
    with ComparisonLedger.open(path, cells=cells) as ledger:
        ledger.start_clock()
        res = ledger.reserve(cells[0].inner_run(0), "main", estimated_input_tokens=100)
        ledger.commit(res.request_id, ProviderUsage(1000000, 8192) if outcome == "overage" else None)
        assert ledger.stopped
    state = json.loads(path.read_text())
    state["stopped"] = None
    path.write_text(json.dumps(state))
    try:
        reopened = ComparisonLedger.open(path, cells=cells)
    except ComparisonBudgetError:
        return
    with reopened:
        assert reopened.stopped, "Unknown/overage records must require a global stop; do not trust only the flag"

def test_duplicate_provider_identity_stops_second_attempt(tmp_path: Path) -> None:
    cells = roster()
    with ComparisonLedger.open(tmp_path / "duplicate.json", cells=cells) as ledger:
        ledger.start_clock()
        for index in range(2):
            try:
                guarded_comparison_attempt(ledger, cells[index].inner_run(0), "main", lambda _: {}, None,
                    estimated_input_tokens=100, usage_fn=lambda _: ProviderUsage(10, 5),
                    provider_request_id_fn=lambda _: "same-provider-request")
            except ComparisonBudgetError:
                assert index == 1
        assert ledger.stopped, "Duplicate provider identity is a post-transport accounting failure"
