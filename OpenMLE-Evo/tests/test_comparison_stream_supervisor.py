"""Independent actual-stream response-identity regression; no network."""
from __future__ import annotations
from types import SimpleNamespace
import pytest
from dojo.core.solvers.llm_helpers.backends.lite_llm import LiteLLMClient
from research.comparison.deploy import API_KEY_PLACEHOLDER


def test_actual_stream_preserves_provider_identity_inside_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LiteLLMClient(SimpleNamespace(
        model_id="k3", base_url="https://example.invalid", api_key=API_KEY_PLACEHOLDER,
        use_azure_client=False, provider="openai"))
    captured = []
    events = []

    class ObserveGuard:
        def run(self, thunk, **kwargs):
            events.append("reserve")
            response = thunk()
            captured.append(response)
            events.append("commit")
            return response

        def check_deadline(self):
            pass

    client.request_guard = ObserveGuard()

    def completion(**kwargs):
        assert kwargs["max_retries"] == kwargs["num_retries"] == 0
        assert kwargs["stream"] is True
        assert events == ["reserve"]
        events.append("transport")

        def chunks():
            yield {"id": "provider-stream-17", "choices": [{"delta": {"content": "ok"}}]}
            assert "commit" not in events
            events.append("usage")
            yield {"id": "provider-stream-17", "choices": [], "usage": {
                "prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26}}

        return chunks()

    monkeypatch.setattr("dojo.core.solvers.llm_helpers.backends.lite_llm.completion_fn", completion)
    output, usage = client._query_client([{"role": "user", "content": "offline"}], {"stream": True})
    assert output == "ok"
    assert events == ["reserve", "transport", "usage", "commit"]
    response = captured[0]
    reported_id = (response.get("id") or response.get("provider_request_id")
                   or response.get("usage", {}).get("provider_request_id"))
    assert reported_id == "provider-stream-17", "Actual stream must preserve its provider id through the accounting boundary"
    assert response["usage"]["total_tokens"] == 26
    assert usage["prompt_tokens"] == 19 and usage["completion_tokens"] == 7


@pytest.mark.parametrize("reported_ids", [("", ""), (17, 17), ("provider-A", "provider-B")])
def test_actual_stream_rejects_ambiguous_identity(tmp_path, monkeypatch, reported_ids):
    from research.comparison.budget import ComparisonLedger
    from research.comparison.protocol import ModelProfile
    from research.comparison.transport import comparison_request_profile, install_comparison_guard
    from research.contracts.budget import CancellationToken
    from test_comparison_inner import _cells

    profile = ModelProfile()
    client = LiteLLMClient(SimpleNamespace(
        model_id=profile.model_id, base_url=profile.base_url,
        api_key=API_KEY_PLACEHOLDER, use_azure_client=False, provider="openai"))
    cells = _cells()
    run = cells[0].inner_run(0)
    attempted = []
    def completion(**kwargs):
        attempted.append(True)
        yield {"id": reported_ids[0], "choices": [{"delta": {"content": "ok"}}]}
        yield {"id": reported_ids[1], "choices": [], "usage": {
            "prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26}}
    monkeypatch.setattr("dojo.core.solvers.llm_helpers.backends.lite_llm.completion_fn", completion)
    with ComparisonLedger.open(tmp_path / "ledger.json", cells=cells) as ledger:
        ledger.start_clock()
        token = CancellationToken(deadline_epoch=ledger.deadline_epoch)
        install_comparison_guard(client, ledger, token, run, operator="draft", debug_budget=2)
        with pytest.raises(Exception):
            client._query_client([{"role": "user", "content": "offline"}], comparison_request_profile())
        assert attempted == [True], "Failure must occur after consuming the malformed provider stream"
        snapshot = ledger.snapshot()
        assert snapshot["stopped"] is not None
        assert len(snapshot["requests"]) == 1
        assert snapshot["requests"][0]["usage_known"] is False
