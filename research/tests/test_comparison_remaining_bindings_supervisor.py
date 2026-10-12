"""Previously identified source gaps, isolated without models or real data."""
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from research.comparison import final
from research.comparison.budget import ComparisonLedger
from research.comparison.orchestrate import OrchestrationError
from research.comparison.runner import ComparisonRunSpec
from research.tests.test_comparison_final import _make_repo_orch, _publish_kwargs, _synthetic_repo
from research.tests.test_comparison_final_supervisor import fake_barrier
from research.tests.test_comparison_orchestrate import charge_main_requests, make_orch, open_ledger, write_fake_result


@pytest.mark.parametrize("mutation", ["none", "tree", "missing_code"])
def test_every_scored_job_binds_actual_code_and_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    orch = make_orch(tmp_path, monkeypatch)
    position = orch.positions[0]
    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        spec_path = orch.prepare_config(position, ledger)
        assert spec_path is not None
        spec = ComparisonRunSpec.from_dict(json.loads(spec_path.read_text()))
    request_ids = charge_main_requests(orch, spec, tmp_path / "ledger.json", 6)
    write_fake_result(orch, spec, 1.0, request_ids=request_ids)
    result_path = orch._result_path(spec.run)
    result = json.loads(result_path.read_text())
    original = result["jobs"][0]
    other = copy.deepcopy(original)
    other_id = "job-synthetic-nonincumbent"
    other["job_id"] = other_id
    other["score"] = other["evaluation"]["score"] = 0.5
    other["evaluation"]["prediction_snapshot"] = f"/mnt/rsi_storage/mlsandbox/jobs/{other_id}/prediction_snapshot.csv"
    predictions = orch._run_dir(spec.run) / "predictions"
    (predictions / f"{other_id}.csv").write_bytes((predictions / f"{original['job_id']}.csv").read_bytes())
    result["prediction_artifacts"][other_id] = other["evaluation"]["prediction_sha256"]
    if mutation == "tree":
        other["source_gate"]["source_sha256"] = "f" * 64
    elif mutation == "missing_code":
        source = b"print('absent from actual journal')\n"
        sha = hashlib.sha256(source).hexdigest()
        other["submitted_source_sha256"] = other["source_gate"]["entrypoint_sha256"] = sha
        other["source_gate"]["source_sha256"] = hashlib.sha256(b"main.py\0" + sha.encode() + b"\n").hexdigest()
    result["jobs"].append(other)
    result["job_ids"] = sorted(job["job_id"] for job in result["jobs"])
    result["job_counts"] = {"completed": 2}
    result_path.write_text(json.dumps(result))
    with open_ledger(tmp_path, orch) as ledger:
        if mutation == "none":
            orch.complete_config(position, ledger)
            assert not ledger.stopped
        else:
            with pytest.raises(OrchestrationError):
                orch.complete_config(position, ledger)
            assert ledger.stopped


def test_final_resume_refit_must_still_match_original_bound_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _synthetic_repo(tmp_path)
    orch = _make_repo_orch(tmp_path, monkeypatch)
    barrier = fake_barrier(orch)
    monkeypatch.setattr(orch, "verify_final_barrier", lambda ledger: barrier)
    class NeverExecute:
        def execute(self, **kwargs: Any) -> Any:
            raise AssertionError("no winning programs in this synthetic barrier")
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        final.publish_final_data(orch, ledger, barrier, now_fn=lambda: 1000.0, **_publish_kwargs(repo))
        first = final.run_final_phase(orch, ledger, barrier, NeverExecute(), now_fn=lambda: 1000.0, sealed_root=repo["sealed"])
        assert len(first) == 27 and not ledger.stopped
        record_path = orch.runtime_dir / "final" / "publication.json"
        record = json.loads(record_path.read_text())
        rel = "banknote/data/final/train.csv"
        published = Path(record["publication_root"]) / rel
        replaced = published.read_bytes().replace(b"0.1", b"9.9")
        assert replaced != published.read_bytes()
        published.write_bytes(replaced)
        (repo["sealed"] / "banknote" / "trainval.csv").write_bytes(replaced)
        record["files"][rel] = hashlib.sha256(replaced).hexdigest()
        record_path.write_text(json.dumps(record))
        # Original frozen cell hashes and search train/validation bytes are
        # unchanged. Only the mutable refit copies + self-record agree.
        with pytest.raises(final.FinalPhaseError):
            final.run_final_phase(orch, ledger, barrier, NeverExecute(), now_fn=lambda: 1000.0, sealed_root=repo["sealed"])
        assert ledger.stopped
