"""Exercise the actual CLI payload with its HTTP boundary replaced."""
import io
import json
from pathlib import Path
import runpy
import urllib.request

ROOT = Path(__file__).resolve().parents[2]

def invoke(monkeypatch, overrides):
    for name in ("SANDBOX_ENDPOINT", "SANDBOX_DATA_DIR", "SANDBOX_TASK_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SANDBOX_API_KEY", "dummy-offline")
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    captured = []
    def respond(request, **kwargs):
        captured.append(request)
        return io.BytesIO(b"{}")
    monkeypatch.setattr(urllib.request, "urlopen", respond)
    runpy.run_path(str(ROOT / "research/experiments/submit_e2e.py"), run_name="__main__")
    assert len(captured) == 1
    return captured[0].full_url, json.loads(captured[0].data)

def test_actual_script_defaults_to_registered_isolated_task(monkeypatch):
    url, payload = invoke(monkeypatch, {})
    assert url.startswith("http://127.0.0.1:6581/")
    assert payload["data_dir"] == "/mnt/rsi_data/hello_synth"
    assert payload["task_id"] == "hello_synth"

def test_actual_script_accepts_complete_explicit_rollback_configuration(monkeypatch):
    url, payload = invoke(monkeypatch, {"SANDBOX_ENDPOINT":"http://127.0.0.1:6580", "SANDBOX_DATA_DIR":"/mnt/pubdatasets2/tasks/hello_synth", "SANDBOX_TASK_ID":"hello_synth_e2e"})
    assert url.startswith("http://127.0.0.1:6580/")
    assert payload["data_dir"] == "/mnt/pubdatasets2/tasks/hello_synth"
    assert payload["task_id"] == "hello_synth_e2e"
