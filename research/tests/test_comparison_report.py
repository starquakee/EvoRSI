"""Offline tests for research/comparison/report.py (EXP-008).

Fixtures are GENUINE evidence chains built through the production
orchestrator/ledger/final machinery with the fake child runner and the
synthetic executor (no model, no network, no sandbox): a fully completed
+ published + final-scored experiment, a one-cell partial matrix and a
never-started matrix. The report must render all 27 cells truthfully in
every case, never infer final scores from validation, never zero-fill
unknown usage and never mutate or flock the original ledger.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.comparison import deploy, review
from research.comparison.budget import ComparisonLedger, ProviderUsage
from research.comparison.orchestrate import Orchestrator
from research.comparison.protocol import canonical_json
from research.comparison.report import (
    FULL_REPORT_SCHEMA,
    build_full_report,
    render_markdown,
    write_report_bundle,
)
from research.comparison.specs import iter_cells
from research.tests.test_comparison_final import (
    SyntheticExecutor,
    _publish_kwargs,
    _repo_bindings,
    _run_final,
    _synthetic_repo,
)
from research.tests.test_comparison_orchestrate import (
    ScriptedNative,
    default_scores,
    fake_child,
    step_positions,
)
from research.tests.test_comparison_specs import COMMIT


def _write_manifest(orch: Orchestrator) -> None:
    """Synthetic frozen manifest genuinely matching this orchestrator AND
    the full pinned schema (protocol/model/budget/data/registry/native/
    source-file pins) verified by the audit/report shared binding check."""
    from research.comparison.data import manifest_identity
    from research.comparison.protocol import ModelProfile, canonical_protocol
    from research.comparison.schedule import JOB_TIMEOUT_SECONDS

    repo_root = orch.repo_root
    bindings = {cell.task_id: cell.task_binding for cell in orch.cells}
    # Data manifest: reuse the synthetic repo's manifest.json bytes when
    # present; otherwise write a minimal genuine one at the pinned path.
    data_manifest_rel = "research/comparison/data-manifest.v1.json"
    data_manifest_path = repo_root / data_manifest_rel
    existing = repo_root / "manifest.json"
    if not data_manifest_path.exists():
        data_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if existing.exists():
            data_manifest_path.write_bytes(existing.read_bytes())
        else:
            data_manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "istratde-comparison-data.v1",
                        "tasks": {
                            task_id: {
                                "task_binding": binding.to_dict(),
                                "task_binding_identity": binding.identity(),
                            }
                            for task_id, binding in sorted(bindings.items())
                        },
                    }
                ),
                encoding="utf-8",
            )
    # Registry: ensure ALL three comparison search entries exist so the
    # required pins are genuinely present in the file.
    registry_rel = "research/evaluator/registry.v1.json"
    registry_path = repo_root / registry_rel
    if registry_path.exists():
        registry_payload = json.loads(registry_path.read_text())
    else:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_payload = {
            "registry_version": "evaluator-registry.v1",
            "schema_version": 1,
            "tasks": {},
        }
    for task_id, binding in sorted(bindings.items()):
        registry_payload["tasks"].setdefault(
            f"comparison_{task_id}",
            {
                "metric": "accuracy",
                "split": "validation",
                "answer_sha256": binding.validation_labels_sha256,
            },
        )
    registry_path.write_text(json.dumps(registry_payload), encoding="utf-8")
    # Native env pin at the canonical relative path.
    native_env_rel = "research/comparison/native-env.json"
    native_env_path = repo_root / native_env_rel
    if not native_env_path.exists():
        native_env_path.write_text(
            json.dumps({"jax_version": "0.4.33", "jax_platform": "cpu"}),
            encoding="utf-8",
        )
    # Every REQUIRED pinned source file exists with real bytes.
    for rel in review.PINNED_FILES:
        path = repo_root / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# synthetic pinned fixture: {rel}\n", encoding="utf-8")
    pinned = {rel: _sha256(repo_root / rel) for rel in review.PINNED_FILES}
    proto = canonical_protocol()
    profile = ModelProfile()
    manifest = {
        "schema": review.MANIFEST_SCHEMA,
        "source_commit": orch.source_commit,
        "protocol": proto.to_dict(),
        "protocol_identity": proto.identity(),
        "model_profile": profile.to_dict(),
        "model_profile_identity": profile.identity(),
        "data_manifest": data_manifest_rel,
        "data_manifest_sha256": pinned[data_manifest_rel],
        "data_manifest_identity": manifest_identity(
            json.loads(data_manifest_path.read_text())
        ),
        "task_bindings": {
            task_id: binding.identity() for task_id, binding in sorted(bindings.items())
        },
        "cells": {cell.cell_key(): cell.identity() for cell in orch.cells},
        "num_cells": 27,
        "num_inner_runs": 216,
        "schedule_identity": orch._schedule_identity,
        "num_positions": 216,
        "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
        "registry": {
            "path": registry_rel,
            "sha256": pinned[registry_rel],
            "registry_version": registry_payload["registry_version"],
            "tasks": {
                task_id: {
                    "split": entry["split"],
                    "metric": entry["metric"],
                    "answer_sha256": entry["answer_sha256"],
                }
                for task_id, entry in registry_payload["tasks"].items()
            },
        },
        "native": {
            "env_file": native_env_rel,
            "env_file_sha256": pinned[native_env_rel],
        },
        "stack_images": {"pins": dict(review.STACK_IMAGE_PINS), "image_ids": {}},
        "pinned_file_sha256": pinned,
    }
    (orch.runtime_dir / review.MANIFEST_FILENAME).write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_orch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Orchestrator:
    monkeypatch.setattr(deploy, "git_head_commit", lambda root: COMMIT)
    orch = Orchestrator.initialize(
        runtime_dir=tmp_path / "state",
        repo_root=tmp_path,
        cells=iter_cells(_repo_bindings(), COMMIT),
        provider=ScriptedNative(),
        now_fn=lambda: 1000.0,
    )
    _write_manifest(orch)
    return orch


@pytest.fixture(scope="module")
def completed(tmp_path_factory) -> dict:
    """ONE fully completed + published + final-scored synthetic experiment."""
    base = tmp_path_factory.mktemp("report-complete")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(deploy, "git_head_commit", lambda root: COMMIT)
        orch = Orchestrator.initialize(
            runtime_dir=base / "state",
            repo_root=base,
            cells=iter_cells(_repo_bindings(), COMMIT),
            provider=ScriptedNative(),
            now_fn=lambda: 1000.0,
        )
        ledger_path = orch.runtime_dir / review.LEDGER_FILENAME
        with ComparisonLedger.open(
            ledger_path, cells=orch.cells, now_fn=lambda: 1000.0
        ) as ledger:
            ledger.start_clock()
            orch.run_schedule(
                ledger, fake_child(orch, default_scores(orch), [], ledger_path)
            )
        # run_schedule relinquishes its handle; reopen for the barrier,
        # publication and final phase (the production handoff pattern).
        with ComparisonLedger.open(
            ledger_path, cells=orch.cells, now_fn=lambda: 1000.0
        ) as ledger:
            barrier = orch.verify_final_barrier(ledger)
            from research.comparison.final import publish_final_data

            repo = _synthetic_repo(base)
            publication = publish_final_data(
                orch, ledger, barrier, now_fn=lambda: 2000.0, **_publish_kwargs(repo)
            )
            executor = SyntheticExecutor(
                repo["sealed"], registry_version=publication["registry_version"]
            )
            results = _run_final(
                orch, ledger, barrier, executor, now_fn=lambda: 2000.0, repo=repo
            )
        _write_manifest(orch)
    assert len(results) == 27
    return {
        "base": base,
        "runtime_dir": base / "state",
        "ledger_path": ledger_path,
        "bindings": _repo_bindings(),
    }


@pytest.fixture()
def partial(tmp_path, monkeypatch) -> dict:
    """ONE fully completed cell; the other 26 never started. No final."""
    orch = _fresh_orch(tmp_path, monkeypatch)
    ledger_path = orch.runtime_dir / review.LEDGER_FILENAME
    cell_key = "banknote:11:FIXED"
    positions = [p for p in orch.positions if p.cell_key == cell_key]
    positions.sort(key=lambda p: p.position)
    with ComparisonLedger.open(
        ledger_path, cells=orch.cells, now_fn=lambda: 1000.0
    ) as ledger:
        ledger.start_clock()
        step_positions(
            orch,
            ledger,
            positions,
            fake_child(orch, default_scores(orch), [], ledger_path),
        )
    return {
        "base": tmp_path,
        "runtime_dir": tmp_path / "state",
        "ledger_path": ledger_path,
        "orch": orch,
        "completed_cell": cell_key,
    }


def _report(fixture: dict):
    base = fixture["base"]
    sealed = base / "research" / "comparison" / "sealed"
    return build_full_report(
        base,
        bindings=fixture["bindings"] if "bindings" in fixture else _repo_bindings(),
        provider=ScriptedNative(),
        runtime_dir=fixture["runtime_dir"],
        sealed_root=sealed if sealed.is_dir() else None,
        now_fn=lambda: 3000.0,
    )


class TestNotStarted:
    def test_all_cells_explicit_not_started(self, tmp_path, monkeypatch):
        orch = _fresh_orch(tmp_path, monkeypatch)
        report = _report({"base": tmp_path, "runtime_dir": orch.runtime_dir})
        assert report["schema"] == FULL_REPORT_SCHEMA
        assert report["ledger"] is None
        assert report["ledger_error"] is None
        assert report["orchestration_state_verified"] is True
        matrix = report["matrix"]
        assert matrix["complete"] is False
        assert matrix["terminal_cells"] == 0
        assert len(matrix["rows"]) == 27
        for row in matrix["rows"]:
            assert row["terminal_status"] == "not_started"
            assert row["final_status"] == "not_reached"
            assert row["raw_accuracy"] is None
            assert row["utility"] is None
            assert row["validation_incumbent"] is None
            assert row["incumbent_curve"] == []
        # No observed pairs -> no estimate, insufficient evidence.
        primary = report["analysis"]["contrasts"]["primary"]
        assert primary["observed_pairs"] == 0
        assert primary["estimate"] is None
        assert report["conclusion"]["conclusion"] == "insufficient-evidence"
        for stats in report["success_only_accuracy"].values():
            assert stats["observed_cells"] == 0
            assert stats["accuracy_mean"] is None


class TestCompletedMatrix:
    def test_full_report_over_actual_artifacts(self, completed):
        report = _report(completed)
        matrix = report["matrix"]
        assert matrix["complete"] is True
        assert matrix["terminal_cells"] == 27
        assert matrix["observed_final_cells"] == 27
        for row in matrix["rows"]:
            assert row["terminal_status"] == "frozen"
            assert row["final_status"] == "completed"
            assert row["raw_accuracy"] == pytest.approx(0.25)
            assert row["utility"] == pytest.approx(0.25)
            assert row["validation_incumbent"] == pytest.approx(0.57)
            assert row["winner_config_index"] == 7
            assert row["configs"]["completed"] == 8
            assert row["candidates"] == 8
            curve = row["incumbent_curve"]
            assert len(curve) == 8
            assert [p["cumulative_requests"] for p in curve] == [
                6 * i for i in range(1, 9)
            ]
            assert curve[-1]["validation_incumbent"] == pytest.approx(0.57)
            assert curve[-1]["cumulative_charged_tokens"] == 4800
        ledger = report["ledger"]
        assert ledger["requests"] == {
            "total": 1296,
            "main": 1296,
            "debug": 0,
            "unknown_usage": 0,
        }
        assert ledger["tokens"]["charged_total"] == 129600
        assert ledger["tokens"]["input_known"] == 77760
        assert ledger["tokens"]["output_known"] == 51840
        assert ledger["stopped"] is None
        assert ledger["final_phase"] is not None
        assert ledger["elapsed_seconds_observed"] == pytest.approx(0.0)
        assert len(ledger["cells"]) == 27
        assert ledger["cells"]["banknote:11:FIXED"]["requests"] == 48
        # Primary/secondary contrasts on the complete matrix.
        primary = report["analysis"]["contrasts"]["primary"]
        assert primary["partial"] is False
        assert primary["observed_pairs"] == 9
        assert primary["estimate"] == pytest.approx(0.0)
        assert primary["interval"]["replicates"] == 10000
        assert primary["interval"]["seed"] == 20261010
        assert report["conclusion"]["conclusion"] == "no-gain"
        for stats in report["success_only_accuracy"].values():
            assert stats["successful"] == 9
            assert stats["failed"] == 0
            assert stats["accuracy_mean"] == pytest.approx(0.25)

    def test_report_is_read_only(self, completed):
        ledger_path = completed["ledger_path"]
        before = _sha256(ledger_path)
        cell_state = (
            completed["runtime_dir"] / "cells" / "banknote-11-FIXED" / "cell.json"
        )
        cell_before = _sha256(cell_state)
        # The flock sidecar left by the fixture's own opens is removed so
        # the report alone proves it never locks.
        lock = ledger_path.with_name(ledger_path.name + ".lock")
        lock.unlink()
        _report(completed)
        assert _sha256(ledger_path) == before
        assert _sha256(cell_state) == cell_before
        # The read-only view never creates the flock sidecar.
        assert not lock.exists()

    def test_report_bundle_writes_json_and_markdown(self, completed, tmp_path):
        report = _report(completed)
        written = write_report_bundle(report, tmp_path / "out")
        assert set(written) == {"report.json", "report.md"}
        loaded = json.loads((tmp_path / "out" / "report.json").read_text())
        assert loaded["schema"] == FULL_REPORT_SCHEMA
        assert (
            hashlib.sha256((tmp_path / "out" / "report.json").read_bytes()).hexdigest()
            == written["report.json"]
        )
        markdown = (tmp_path / "out" / "report.md").read_text()
        assert "27-cell matrix" in markdown
        assert "banknote:11:ISTRATDE" in markdown
        assert "no-gain" in markdown
        assert "bootstrap" in markdown

    def test_markdown_renders_incomplete_honestly(self, partial):
        report = _report(partial)
        markdown = render_markdown(report)
        assert "insufficient-evidence" in markdown
        assert "not_started" in markdown
        assert "missing_final" in markdown


class TestPartialMatrix:
    def test_partial_states_explicit(self, partial):
        report = _report(partial)
        matrix = report["matrix"]
        assert matrix["complete"] is False
        assert matrix["terminal_cells"] == 1
        rows = {row["cell_key"]: row for row in matrix["rows"]}
        done = rows[partial["completed_cell"]]
        assert done["terminal_status"] == "frozen"
        # Frozen winner but NO final result: unobserved, never inferred
        # from the validation incumbent.
        assert done["final_status"] == "missing_final"
        assert done["validation_incumbent"] == pytest.approx(0.57)
        assert done["raw_accuracy"] is None
        assert done["utility"] is None
        assert rows["wdbc:22:RANDOM"]["terminal_status"] == "not_started"
        # Only the completed cell spent budget.
        ledger = report["ledger"]
        assert ledger["requests"]["total"] == 48
        assert ledger["cells"][partial["completed_cell"]]["charged_tokens"] == 4800
        assert ledger["cells"]["wdbc:22:RANDOM"]["requests"] == 0
        # No pair is observable -> no estimate, clearly partial.
        primary = report["analysis"]["contrasts"]["primary"]
        assert primary["observed_pairs"] == 0
        assert primary["partial"] is True
        assert report["conclusion"]["conclusion"] == "insufficient-evidence"


class TestUnknownUsage:
    def test_unknown_usage_stays_unknown_and_stopped(self, tmp_path, monkeypatch):
        orch = _fresh_orch(tmp_path, monkeypatch)
        ledger_path = orch.runtime_dir / review.LEDGER_FILENAME
        run = orch.cells[0].inner_run(0)
        with ComparisonLedger.open(
            ledger_path, cells=orch.cells, now_fn=lambda: 1000.0
        ) as ledger:
            ledger.start_clock()
            reservation = ledger.reserve(run, "main", estimated_input_tokens=100)
            ledger.commit(reservation.request_id, None)  # unknown -> stopped
        report = _report({"base": tmp_path, "runtime_dir": orch.runtime_dir})
        ledger_summary = report["ledger"]
        assert ledger_summary["requests"]["unknown_usage"] == 1
        assert ledger_summary["stopped"]["reason"].startswith("unknown_usage")
        # The FULL reservation stays charged; realized cost is unknown.
        assert ledger_summary["tokens"]["charged_total"] == 8292
        assert ledger_summary["tokens"]["input_known"] == 0
        assert "UNKNOWN" in ledger_summary["tokens"]["note"]


class TestCliBundle:
    def test_default_output_outside_experiment_runtime(
        self, tmp_path, monkeypatch, capsys
    ):
        """The default bundle dir must NOT land inside the experiment
        runtime (it would trip the pre-live re-preparation freshness gate)."""
        from research.comparison import audit as audit_mod
        from research.comparison import plots as plots_mod
        from research.comparison import report as report_mod
        from research.comparison.cli import main

        monkeypatch.setattr(
            report_mod,
            "build_full_report",
            lambda *a, **k: {
                "schema": FULL_REPORT_SCHEMA,
                "source_commit": None,
                "orchestration_state_verified": False,
                "orchestration_error": None,
                "ledger": None,
                "ledger_error": None,
                "matrix": {
                    "complete": False,
                    "rows": [],
                    "terminal_cells": 0,
                    "observed_final_cells": 0,
                    "num_cells": 27,
                },
                "success_only_accuracy": {},
                "analysis": {"contrasts": {}, "caveats": []},
                "conclusion": {"conclusion": "insufficient-evidence", "reason": "x"},
            },
        )
        monkeypatch.setattr(
            audit_mod,
            "audit_experiment",
            lambda *a, **k: {"schema": "test", "verdict": "INCOMPLETE"},
        )
        monkeypatch.setattr(
            plots_mod,
            "render_report_plots",
            lambda report, out: {},
        )
        assert main(["report", "--repo-root", str(tmp_path)]) == 0
        summary = json.loads(capsys.readouterr().out)
        expected = tmp_path.resolve() / ".runtime" / "comparison-reports"
        assert Path(summary["output_dir"]) == expected
        assert (expected / "report.json").is_file()
        assert (expected / "report.md").is_file()
        assert (expected / "audit.json").is_file()
        # Nothing written inside the experiment runtime dir.
        runtime = tmp_path / ".runtime" / "comparison-istratde"
        assert not runtime.exists()


class TestTamperedState:
    def test_tampered_cell_state_visible_not_crashing(self, tmp_path, monkeypatch):
        orch = _fresh_orch(tmp_path, monkeypatch)
        cell_dir = next((orch.runtime_dir / "cells").iterdir())
        payload = json.loads((cell_dir / "cell.json").read_text())
        payload["vectors"]["0"] = [0.9, 1.9, 0.9, 0.9, 1.0]
        (cell_dir / "cell.json").write_text(json.dumps(payload))
        report = _report({"base": tmp_path, "runtime_dir": orch.runtime_dir})
        assert report["orchestration_state_verified"] is False
        assert report["orchestration_error"]
        # Rows still produced from the raw persisted states, marked by the
        # verification flag above rather than hidden.
        assert len(report["matrix"]["rows"]) == 27
