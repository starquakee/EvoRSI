"""Independent comparison spec/fresh-run boundary tests; no auth or model."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pytest
import research.comparison.runner as runner
from research.comparison.data import load_task_bindings
from research.comparison.protocol import FIXED_THETA, decode_theta
from research.comparison.specs import iter_cells
from research.contracts.results import MetricDirection
from research.evaluator.service import EvaluatorRegistry
from research.tests.test_comparison_specs import make_bindings

ROOT = Path(__file__).resolve().parents[2]

def spec_fixture() -> runner.ComparisonRunSpec:
    inner = iter_cells(make_bindings(), "a"*40)[0].inner_run(0)
    return runner.ComparisonRunSpec(
        run=inner, decoded=decode_theta(FIXED_THETA),
        run_dir=str(runner.RUNTIME_DIR/"runs"/runner.run_dir_name(inner)),
        ledger_path=str(runner.RUNTIME_DIR/runner.LEDGER_FILENAME), runner_commit="a"*40)

def test_comparison_is_cpu_only() -> None:
    with pytest.raises(runner.ComparisonRunnerError):
        replace(spec_fixture(), resource_type="gpu").validate()

def test_direct_spec_source_commit_must_match_cell() -> None:
    with pytest.raises(runner.ComparisonRunnerError):
        replace(spec_fixture(), runner_commit="b"*40).validate()

def test_serialized_inner_seed_must_be_integer() -> None:
    value = spec_fixture().to_dict()
    value["inner_seed"] = float(value["inner_seed"])
    with pytest.raises(runner.ComparisonRunnerError):
        runner.ComparisonRunSpec.from_dict(value)

@pytest.mark.parametrize("case", ["existing_checkpoint", "symlinked_runs_parent", "symlinked_ledger_file"])
def test_entry_refuses_reuse_or_public_path_before_runtime_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    spec = spec_fixture()
    source = tmp_path/"spec.json"
    source.write_text(json.dumps(spec.to_dict()))
    ledger = tmp_path/spec.ledger_path
    ledger.parent.mkdir(parents=True)
    ledger.write_text("{}")
    run_dir = tmp_path/spec.run_dir
    if case == "existing_checkpoint":
        run_dir.mkdir(parents=True)
        (run_dir/"checkpoint").mkdir()
        (run_dir/"checkpoint"/"journal.jsonl").write_text("previous interrupted run")
    elif case == "symlinked_ledger_file":
        outside = tmp_path/"foreign-ledger.json"
        outside.write_text("{}")
        ledger.unlink()
        ledger.symlink_to(outside)
    else:
        public = tmp_path/"tasks"/"comparison"/"banknote"/"data"/"public"
        public.mkdir(parents=True)
        run_dir.parent.symlink_to(public, target_is_directory=True)
    monkeypatch.setattr(runner, "check_comparison_review", lambda _: None)
    monkeypatch.setattr(runner.ac, "git_head_commit", lambda _: "a"*40)
    monkeypatch.setattr(runner, "load_task_bindings", make_bindings)
    monkeypatch.setattr(runner, "verify_task_data_identity", lambda *a, **kw: {})
    monkeypatch.setattr(runner.EvaluatorRegistry, "load", lambda: SimpleNamespace())
    with pytest.raises(runner.ComparisonRunnerError):
        runner.run_inner(source, repo_root=tmp_path)

@pytest.mark.parametrize("field,bad", [("metric", "rmse"), ("direction", MetricDirection.MINIMIZE)])
def test_task_verification_pins_accuracy_maximization(field: str, bad: Any) -> None:
    binding = load_task_bindings()["banknote"]
    registry = EvaluatorRegistry.load()
    original = registry.task("comparison_"+binding.task_id)
    changed = replace(original, **{field: bad})
    altered = SimpleNamespace(registry_version=registry.registry_version, task=lambda _: changed)
    with pytest.raises(runner.ComparisonRunnerError):
        runner.verify_task_data_identity(binding, repo_root=ROOT, registry=altered)
