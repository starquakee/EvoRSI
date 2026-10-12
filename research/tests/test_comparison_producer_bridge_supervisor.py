"""Actual task-adapter output must compose with the outer verifier.

Only raw evaluator responses/provider usage are synthetic. The persisted
job dictionary is produced by the REAL ComparisonSandboxTask methods.
No model, network, real task data or live experiment is used.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pytest
from research.comparison import runner
from research.comparison.runner import ComparisonRunSpec
from research.comparison.task import ComparisonSandboxTask
from research.contracts.budget import CancellationToken
from research.contracts.results import MetricDirection
from research.contracts.source_gate import check_source_bytes, load_policy
from research.search.sandbox_task import SandboxExecutionResult, _single_file_tree_sha256
from research.tests.test_comparison_orchestrate import make_orch, open_ledger, charge_main_requests, write_fake_result


def task_adapter(spec: ComparisonRunSpec, ledger: Any) -> Any:
    # Inject synthetic trusted dependencies while retaining the actual
    # producer methods (including their field filtering and failure forms).
    task: Any = ComparisonSandboxTask.__new__(ComparisonSandboxTask)
    task.jobs = []
    task.stop_requested = False
    task._policy = load_policy()
    task._registry = SimpleNamespace(registry_version="evaluator-registry.v1")
    task._spec = SimpleNamespace(task_id=f"comparison_{spec.run.cell.task_id}", metric="accuracy",
                                 direction=MetricDirection.MAXIMIZE, split="validation",
                                 answer_sha256=spec.run.cell.task_binding.validation_labels_sha256)
    task._expected_rows = 1
    task._result_cls = SandboxExecutionResult
    task._comparison_ledger = ledger
    task._token = CancellationToken()
    task._name_prefix = "synthetic-producer-bridge"
    task._data_dir = f"/mnt/rsi_data/comparison/{spec.run.cell.task_id}/data/public"
    task._task_id = task._spec.task_id
    task._resource_type = "cpu"
    task._job_timeout = 600
    return task


@pytest.mark.parametrize("outcome", ["scored", "admission_denied"])
def test_actual_adapter_persisted_evidence_composes_with_config_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    orch = make_orch(tmp_path, monkeypatch)
    position = orch.positions[0]
    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        spec_path = orch.prepare_config(position, ledger)
        assert spec_path is not None
        spec = ComparisonRunSpec.from_dict(json.loads(spec_path.read_text()))
    request_ids = charge_main_requests(orch, spec, tmp_path / "ledger.json", 6)
    write_fake_result(orch, spec, 1.0 if outcome == "scored" else None, request_ids=request_ids)
    result_path = orch._result_path(spec.run)
    result = json.loads(result_path.read_text())
    with open_ledger(tmp_path, orch) as ledger:
        task = task_adapter(spec, ledger)
        if outcome == "scored":
            incumbent = result["incumbent"]
            nodes = [json.loads(line) for line in (orch._run_dir(spec.run) / "journal.jsonl").read_text().splitlines() if line]
            code = next(node["code"] for node in nodes if node["id"] == incumbent["node_id"])
            data = code.encode("utf-8", errors="surrogatepass")
            gate = check_source_bytes(data, task._policy).to_dict()
            gate.update(entrypoint="main.py", entrypoint_sha256=hashlib.sha256(data).hexdigest(),
                        source_sha256=_single_file_tree_sha256("main.py", data))
            evaluation = {"evaluator": "external_trusted", "task_id": task._spec.task_id,
                          "split": "validation", "metric": "accuracy", "direction": "maximize",
                          "score": 1.0, "row_count": 1,
                          "answer_sha256": task._spec.answer_sha256,
                          "registry_version": task._registry.registry_version,
                          "prediction_sha256": incumbent["prediction_sha256"],
                          "prediction_snapshot": f"/mnt/rsi_storage/mlsandbox/jobs/{incumbent['job_id']}/prediction_snapshot.csv"}
            raw = {"completed_at": "1970-01-01T00:16:40Z", "result": {
                "source_identity_verified": True, "worker_cleanup_verified": True,
                "source_gate": gate, "evaluation": evaluation}}
            task._translate(SimpleNamespace(status="completed", score=1.0,
                                             job_id=incumbent["job_id"], raw=raw), code, 0.01)
            assert task.jobs[-1]["evidence_verified"] is True, "the real adapter must first verify the synthetic raw response"
        else:
            class DeniedClient:
                def submit_and_wait(self, *args: Any, **kwargs: Any) -> Any:
                    raise ValueError("source_gate_denied_fixture")
            task._client = DeniedClient()
            task.step_task({}, "print('synthetic candidate')\n")
            assert task.jobs[-1]["status"] == "admission_denied"
            assert task.jobs[-1]["job_id"] is None
        assert not ledger.stopped
        result["jobs"] = task.jobs
        result["job_counts"] = runner.job_outcome_counts(task.jobs)
        result["job_ids"] = sorted({job["job_id"] for job in task.jobs if job.get("job_id")})
        result_path.write_text(json.dumps(result))
        orch.complete_config(position, ledger)
        assert not ledger.stopped
