"""Last-slot quota exhaustion must not invent a native feedback observation."""
import json
from pathlib import Path
import pytest
from research.comparison.budget import CellBudgetExhausted, ProviderUsage
from research.comparison.runner import ComparisonRunSpec
from research.tests.test_comparison_orchestrate import ScriptedNative, make_orch, open_ledger, write_fake_result


@pytest.mark.parametrize("last_slot_cap", [False, True])
def test_native_completion_requires_actual_last_slot_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, last_slot_cap: bool) -> None:
    provider = ScriptedNative()
    orch = make_orch(tmp_path, monkeypatch, provider=provider)
    cell = next(cell for cell in orch.cells if cell.method == "ISTRATDE")
    positions = sorted((p for p in orch.positions if p.cell_key == cell.cell_key()), key=lambda p: p.config_index)
    with open_ledger(tmp_path, orch) as ledger:
        ledger.start_clock()
        for pos in positions:
            spec_path = orch.prepare_config(pos, ledger)
            assert spec_path is not None
            spec = ComparisonRunSpec.from_dict(json.loads(spec_path.read_text()))
            request_ids: list[str] = []
            if last_slot_cap and pos.config_index == 7:
                # REAL cell quota: 42 settled requests at 4600 tokens =
                # 193200; next 8292-token reservation cannot fit in 200000.
                # This is a valid theta whose first MAIN is never sent.
                with pytest.raises(CellBudgetExhausted) as denied:
                    ledger.reserve(spec.run, "main", estimated_input_tokens=100)
                termination = str(denied.value)
                if not termination.startswith("CellBudgetExhausted"):
                    termination = "CellBudgetExhausted:" + termination
                write_fake_result(orch, spec, None, termination=termination,
                                  status="cell_terminated", request_ids=request_ids)
            else:
                for index in range(6):
                    reservation = ledger.reserve(spec.run, "main", estimated_input_tokens=100)
                    ledger.commit(reservation.request_id, ProviderUsage(4560 if last_slot_cap else 60, 40),
                                  provider_request_id=f"synthetic-{pos.config_index}-{index}")
                    request_ids.append(reservation.request_id)
                # All six synthetic model attempts genuinely fail to
                # produce an executable candidate: truthful null/utility0.
                write_fake_result(orch, spec, None, request_ids=request_ids)
                result_path = orch._run_dir(spec.run) / "run-result.json"
                result = json.loads(result_path.read_text())
                result["requests"]["charged_tokens"] = len(request_ids) * (4600 if last_slot_cap else 100)
                result_path.write_text(json.dumps(result))
            orch.complete_config(pos, ledger)
        state = orch._cell_state(cell.cell_key())
        assert state["terminal"]["status"] == "failed_no_valid_program"
        assert not ledger.stopped
        if last_slot_cap:
            assert provider.complete_calls == [], "zero-request final slot is UNEVALUATED, never a synthetic zero-fitness tell"
            assert state.get("native_completion") is None
        else:
            assert len(provider.complete_calls) == 1
            assert state.get("native_completion") is not None
