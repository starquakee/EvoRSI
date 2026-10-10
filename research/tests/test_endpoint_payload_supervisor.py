"""Default endpoint switch must also route a usable task/data payload."""
import json
import httpx
import pytest
from research.adapters.sandbox_eval_client import SandboxEvalClient

def capture_submit(monkeypatch, use_wrapper=False, **kwargs):
    captured = []
    original_client = httpx.Client
    def handler(request):
        captured.append(request)
        return httpx.Response(503, json={"detail": "offline capture only"})
    def factory(*args, **options):
        options["transport"] = httpx.MockTransport(handler)
        return original_client(*args, **options)
    monkeypatch.setattr(httpx, "Client", factory)
    endpoint = kwargs.pop("endpoint", "http://127.0.0.1:6581")
    client = SandboxEvalClient(endpoint=endpoint, api_key="dummy-offline")
    target = client.submit_and_wait
    if use_wrapper:
        from research.adapters import sandbox_eval_client as module
        monkeypatch.setattr(module, "_DEFAULT_CLIENT", client)
        target = module.submit_and_wait
    with pytest.raises(httpx.HTTPStatusError):
        target("print('synthetic payload probe')", **kwargs)
    assert len(captured) == 1
    return captured[0], json.loads(captured[0].content)

def test_new_gateway_default_submission_has_mounted_data_and_registered_task(monkeypatch):
    request, payload = capture_submit(monkeypatch)
    assert request.url.port == 6581
    assert payload["data_dir"] == "/mnt/rsi_data/hello_synth"
    assert payload.get("task_id") == "hello_synth"

def test_explicit_rollback_payload_is_preserved(monkeypatch):
    request, payload = capture_submit(monkeypatch, endpoint="http://127.0.0.1:6580", data_dir="/mnt/pubdatasets2/tasks/hello_synth", task_id="legacy-test")
    assert request.url.port == 6580
    assert payload["data_dir"] == "/mnt/pubdatasets2/tasks/hello_synth"
    assert payload["task_id"] == "legacy-test"

def test_public_wrapper_preserves_explicit_legacy_task(monkeypatch):
    request, payload = capture_submit(monkeypatch, use_wrapper=True, endpoint="http://127.0.0.1:6580", data_dir="/mnt/pubdatasets2/tasks/hello_synth", task_id="legacy-test")
    assert request.url.port == 6580
    assert payload["data_dir"] == "/mnt/pubdatasets2/tasks/hello_synth"
    assert payload["task_id"] == "legacy-test"
