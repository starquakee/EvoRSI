"""Final-phase resume must recheck the reviewed runtime identity, too."""
import pytest
from research.comparison import budget, cli, preflight, review
from research.tests.test_comparison_cli import TestLiveWiring as _Wiring, _FakeLedger

@pytest.mark.parametrize("runtime_phase", ("search", "final"))
def test_final_resume_refuses_changed_runtime_before_final_execution(monkeypatch, tmp_path, runtime_phase):
    log, holders = _Wiring()._install_fakes(monkeypatch)
    stops=[]
    class FinalLedger(_FakeLedger):
        final_phase={"barrier_identity":"b"*64}
        def stop(self, reason):
            stops.append(reason)
    monkeypatch.setattr(budget.ComparisonLedger, "open", lambda path, *, cells: FinalLedger(log))
    baseline={name:"sha256:"+"a"*64 for name in review.STACK_IMAGE_PINS}
    monkeypatch.setattr(cli, "_manifest_image_baseline", lambda root: baseline)
    ledger=tmp_path/review.RUNTIME_DIR/review.LEDGER_FILENAME
    ledger.parent.mkdir(parents=True)
    ledger.write_text("{}")
    changed=[False]
    inspected=[]
    def inspect(root, **kwargs):
        inspected.append(kwargs)
        assert kwargs["expected_images"]==baseline
        if changed[0]:
            raise review.ComparisonManifestError("stack_image_id_mismatch:rsi_worker")
        return {"phase":runtime_phase, "containers":{}}
    monkeypatch.setattr(preflight, "inspect_runtime_stack", inspect)
    assert cli.main(["resume", "--repo-root", str(tmp_path)])==0
    assert "activate_final_phase" in log and stops==[]
    log.clear()
    inspected.clear()
    changed[0]=True
    assert cli.main(["resume", "--repo-root", str(tmp_path)])==2
    assert len(inspected)==1
    assert not any(isinstance(item,tuple) and item[0]=="run_schedule" for item in log)
    assert "activate_final_phase" not in log
    assert stops
