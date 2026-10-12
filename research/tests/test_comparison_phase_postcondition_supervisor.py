"""Final postcondition work must finish before the original deadline."""
from pathlib import Path
import pytest
from research.comparison import phase

@pytest.mark.parametrize("finish", (20.0, 21.0))
def test_phase_postcondition_cannot_return_after_deadline(finish, monkeypatch):
    def run(end):
        clock=[10.0]
        events=[]
        stopped=[]
        phases=iter(("search", "final"))
        def idle(*args, **kwargs):
            events.append("idle")
            if events.count("idle")==2:
                clock[0]=end
        monkeypatch.setattr(phase, "verify_idle", idle)
        monkeypatch.setattr(phase, "current_phase", lambda *a, **k: next(phases))
        monkeypatch.setattr(phase, "_compose_recreate_dispatcher", lambda *a, **k: events.append("recreate"))
        monkeypatch.setattr(phase, "_wait_dispatcher_running", lambda *a, **k: None)
        monkeypatch.setattr(phase, "_container_registry_sha256", lambda *a, **k: "a"*64)
        def activate():
            return phase.activate_final_phase(Path("/fixture"), expected_registry_sha256="a"*64,
                deadline_epoch=20.0, now_fn=lambda: clock[0], on_failure=stopped.append)
        return activate, events, stopped
    positive, reached, stopped=run(19.0)
    assert positive()=="activated_final"
    assert reached==["idle", "recreate", "idle"] and stopped==[]
    expired, reached, stopped=run(finish)
    with pytest.raises(phase.PhaseActivationError):
        expired()
    assert reached==["idle", "recreate", "idle"]
    assert stopped
