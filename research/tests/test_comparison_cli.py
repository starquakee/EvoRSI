"""Offline tests for research/comparison/cli.py (EXP-001 surfaces, EXP-007 wiring).

prepare/preflight/report are offline commands; run/resume are review-gated
live commands. Without a prepared runtime and the ignored supervisor
review every command refuses safely (exit 2, machine-readable reason, no
artifacts). The live wiring test proves the review gate runs first and the
child runner/final executor receive the ORIGINAL ledger deadline. The
US-009 interfaces and both historic ledgers stay byte-identical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.comparison.cli import COMMANDS, COMMAND_STORIES, main

REPO_ROOT = Path(__file__).resolve().parents[2]

HISTORIC_LEDGERS = (
    REPO_ROOT / ".runtime" / "acceptance-us009" / "budget-ledger.json",
    REPO_ROOT / ".runtime" / "acceptance-us009-round2" / "budget-ledger.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestCommandSurfaces:
    def test_all_five_commands_exist(self):
        assert COMMANDS == ("prepare", "preflight", "run", "resume", "report")
        assert set(COMMAND_STORIES) == set(COMMANDS)

    def test_no_command_is_a_parse_error(self):
        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code == 2

    def test_unknown_command_is_a_parse_error(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["explode"])
        assert excinfo.value.code == 2

    def test_unprepared_commands_refuse_safely(self, tmp_path, capsys):
        """In an empty directory every command refuses and creates nothing."""
        before = set(tmp_path.rglob("*"))
        for command in COMMANDS:
            assert main([command, "--repo-root", str(tmp_path)]) == 2, command
        err = capsys.readouterr().err
        for reason in (
            "comparison_git_unavailable",  # prepare
            "comparison_manifest_missing",  # preflight
            "comparison_review_absent",  # run
            "comparison_review_absent",  # resume
            "comparison_manifest_missing",  # report
        ):
            assert reason in err
        assert set(tmp_path.rglob("*")) == before

    def test_run_and_resume_refuse_without_review(self, capsys):
        """The live gate is first: no ledger, no credentials, no model."""
        runtime = REPO_ROOT / ".runtime" / "comparison-istratde"
        ledger = runtime / "comparison-ledger.json"
        assert not ledger.exists()
        for command in ("run", "resume"):
            assert main([command, "--repo-root", str(REPO_ROOT)]) == 2
        err = capsys.readouterr().err
        assert err.count("comparison_review_absent") == 2
        assert not ledger.exists()


class _FakeLedger:
    def __init__(self, log: list):
        self._log = log
        self.started_at = 1000.0
        self.deadline_epoch = 1000.0 + 86400.0
        self.closed = False
        log.append("open")

    def start_clock(self) -> None:
        self._log.append("start_clock")

    def close(self) -> None:
        self.closed = True
        self._log.append("close")


class _FakeOrch:
    def __init__(self, log: list):
        self._log = log
        self.cells = ()
        self.runtime_dir = Path("unused")

    def run_schedule(self, ledger, child_runner) -> dict:
        self._log.append(("run_schedule", child_runner))
        ledger.close()
        return {"cell": "terminal"}

    def verify_final_barrier(self, ledger) -> dict:
        self._log.append("verify_final_barrier")
        return {"barrier_identity": "b" * 64}


class TestLiveWiring:
    def _install_fakes(self, monkeypatch) -> tuple[list, dict]:
        log: list = []
        holders: dict = {}

        import research.comparison.budget as budget_mod
        import research.comparison.data as data_mod
        import research.comparison.final as final_mod
        import research.comparison.native as native_mod
        import research.comparison.orchestrate as orch_mod
        import research.comparison.phase as phase_mod
        import research.comparison.preflight as preflight_mod
        import research.comparison.review as review_mod

        monkeypatch.setattr(
            review_mod, "check_review_acceptance", lambda root: log.append("review")
        )
        monkeypatch.setattr(data_mod, "load_task_bindings", lambda: {})
        monkeypatch.setattr(
            native_mod, "SubprocessNativeProvider", lambda root: object()
        )
        monkeypatch.setattr(
            orch_mod, "load_orchestrator", lambda **kw: _FakeOrch(log)
        )
        monkeypatch.setattr(
            budget_mod.ComparisonLedger, "open", lambda path, *, cells: _FakeLedger(log)
        )
        monkeypatch.setattr(
            preflight_mod,
            "inspect_runtime_stack",
            lambda root, **kw: {"phase": "search", "containers": {}},
        )
        import research.comparison.cli as cli_mod

        monkeypatch.setattr(cli_mod, "_manifest_image_baseline", lambda root: None)

        def _child(repo_root, *, deadline_fn=None):
            holders["child_deadline_fn"] = deadline_fn
            return lambda spec_path: 0

        monkeypatch.setattr(orch_mod, "evo_child_runner", _child)
        monkeypatch.setattr(
            final_mod,
            "publish_final_data",
            lambda orch, ledger, barrier: {
                "publication_root": "pub",
                "registry_file": "registry.final.v1.json",
                "files": {"registry.final.v1.json": "c" * 64},
            },
        )

        def _activate(repo_root, *, expected_registry_sha256, deadline_epoch, on_failure):
            holders["activate"] = {
                "sha256": expected_registry_sha256,
                "deadline_epoch": deadline_epoch,
                "on_failure": on_failure,
            }
            log.append("activate_final_phase")
            return "activated_final"

        monkeypatch.setattr(phase_mod, "activate_final_phase", _activate)

        def _executor(repo_root, publication, *, state_dir, deadline_fn=None):
            holders["executor_deadline_fn"] = deadline_fn
            return object()

        monkeypatch.setattr(final_mod, "sandbox_final_executor", _executor)
        monkeypatch.setattr(
            final_mod, "run_final_phase", lambda *a, **kw: {"cell": {"status": "ok"}}
        )
        return log, holders

    def test_run_wires_review_first_and_original_deadline(
        self, monkeypatch, tmp_path
    ):
        log, holders = self._install_fakes(monkeypatch)
        assert main(["run", "--repo-root", str(tmp_path)]) == 0
        assert log[0] == "review"
        assert log[1] == "open"
        assert log[2] == "start_clock"
        original = 1000.0 + 86400.0
        assert holders["child_deadline_fn"]() == original
        assert holders["executor_deadline_fn"]() == original
        # Server-side phase activation: hash-bound, original deadline, and a
        # durable failure sink, AFTER publication and BEFORE final execution.
        activation = holders["activate"]
        assert activation["sha256"] == "c" * 64
        assert activation["deadline_epoch"] == original
        assert callable(activation["on_failure"])
        # The schedule ran before the barrier/publication/activation/final
        # phase, all on the SAME original deadline.
        assert [entry for entry in log if isinstance(entry, str)] == [
            "review",
            "open",
            "start_clock",
            "close",
            "open",
            "verify_final_barrier",
            "activate_final_phase",
            "close",
        ]

    def test_resume_requires_existing_ledger(self, monkeypatch, tmp_path):
        log, _holders = self._install_fakes(monkeypatch)
        assert main(["resume", "--repo-root", str(tmp_path)]) == 2
        assert log == ["review"]  # no ledger open, no clock, no child

    def test_run_refuses_when_ledger_exists(self, monkeypatch, tmp_path):
        log, _holders = self._install_fakes(monkeypatch)
        ledger = tmp_path / ".runtime" / "comparison-istratde" / "comparison-ledger.json"
        ledger.parent.mkdir(parents=True)
        ledger.write_text("{}", encoding="utf-8")
        assert main(["run", "--repo-root", str(tmp_path)]) == 2
        assert log == ["review"]


class TestUS009Untouched:
    def test_historic_ledgers_exist_and_stay_byte_identical(self):
        for ledger in HISTORIC_LEDGERS:
            assert ledger.exists(), f"missing historic evidence: {ledger}"
        before = {path: _sha256(path) for path in HISTORIC_LEDGERS}
        for command in COMMANDS:
            main([command, "--repo-root", str(REPO_ROOT)])
        assert {path: _sha256(path) for path in HISTORIC_LEDGERS} == before

    def test_us009_acceptance_interface_unchanged(self):
        from research.contracts.budget import BudgetLimits
        from research.search import acceptance_config as ac

        assert ac.acceptance_budget_limits() == BudgetLimits(
            max_tokens=200_000, max_requests=30, max_elapsed_seconds=5_400.0
        )
        assert ac.TASK_ID == "hello_synth"
        assert ac.SEED == 42
        assert ac.RUNTIME_DIR == Path(".runtime") / "acceptance-us009"
        assert ac.DEFAULT_ROUND_ID == "round1"

    def test_comparison_package_does_not_reference_us009_runtime(self):
        package_dir = REPO_ROOT / "research" / "comparison"
        for source in package_dir.glob("*.py"):
            text = source.read_text(encoding="utf-8")
            # EXP-007: preflight.py performs the READ-ONLY historic ledger
            # audit (existence, committed totals, sha256 recording); every
            # other module stays free of US-009 runtime references.
            if source.name != "preflight.py":
                assert "acceptance-us009" not in text
            assert "acceptance_config" not in text
