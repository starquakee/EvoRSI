"""A self-consistent replacement manifest cannot replace frozen task data."""
import hashlib
from pathlib import Path
import pytest
from research.comparison import data, deploy
from research.comparison.budget import ComparisonLedger
from research.tests.test_comparison_final import _synthetic_repo
from research.tests.test_comparison_final_supervisor import fake_barrier
from research.tests.test_comparison_orchestration_supervisor import make_controller


def test_publication_refuses_manifest_data_that_disagree_with_frozen_cell_bindings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _synthetic_repo(tmp_path)
    orch = make_controller(tmp_path, monkeypatch)
    barrier = fake_barrier(orch)
    verified = []
    # ONLY the search proof is synthetic. The real publication function,
    # real original ledger phase and frozen CellSpec hashes are exercised.
    def verified_barrier(ledger):
        verified.append(True)
        return barrier
    monkeypatch.setattr(orch, "verify_final_barrier", verified_barrier)
    monkeypatch.setattr(deploy, "git_head_commit", lambda root: orch.source_commit)
    manifest = data.load_manifest(repo["manifest_path"])
    task = orch.cells[0].task_id
    replacement_labels = (repo["sealed"] / task / "final_labels.csv").read_bytes()
    replacement_sha = hashlib.sha256(replacement_labels).hexdigest()
    assert manifest["tasks"][task]["files"][f"research/comparison/sealed/{task}/final_labels.csv"] == replacement_sha
    assert orch.cells[0].task_binding.final_labels_sha256 != replacement_sha
    writes = []
    original_write = data._write_frozen
    def observed_write(path, value):
        writes.append(path)
        return original_write(path, value)
    monkeypatch.setattr(data, "_write_frozen", observed_write)
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        ledger.enter_final_phase(barrier["barrier_identity"])
        with pytest.raises(data.ComparisonDataError, match="binding|manifest|frozen|hash|identity"):
            data.publish_final_refit(orchestrator=orch, ledger=ledger, barrier_evidence=barrier,
                repo_root=tmp_path, sealed_root=repo["sealed"], manifest_path=repo["manifest_path"],
                registry_path=repo["registry_path"], publication_root=tmp_path / "replacement-publication")
    assert verified, "test must reach the real post-barrier publication contract"
    assert not writes, "freeze mismatch must refuse BEFORE exposing any final bytes"
