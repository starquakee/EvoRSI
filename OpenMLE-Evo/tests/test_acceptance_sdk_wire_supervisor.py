"""Actual installed SDK transformation, with HTTP mocked and sockets denied."""
from __future__ import annotations

import json
import socket

import httpx
import litellm
import openai

from research.search import acceptance_config as ac
from research.search.acceptance_inner import build_acceptance_solver
from test_acceptance_inner import REPO_ROOT, _spec


def test_actual_acceptance_profile_survives_litellm_to_http_body(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Real sockets forbidden in SDK profile test")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(litellm, "drop_params", False)
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        chunks = [
            {"id":"offline-wire", "object":"chat.completion.chunk", "created":0, "model":"k3",
             "choices":[{"index":0,"delta":{"role":"assistant","content":"print(1)"},"finish_reason":None}]},
            {"id":"offline-wire", "object":"chat.completion.chunk", "created":0, "model":"k3",
             "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]},
            {"id":"offline-wire", "object":"chat.completion.chunk", "created":0, "model":"k3", "choices":[],
             "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}},
        ]
        text = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200,headers={"content-type":"text/event-stream"},text=text)
    solver = build_acceptance_solver(_spec(), repo_root=REPO_ROOT)
    kwargs = dict(solver.cfg.operators["draft"].llm.generation_kwargs)
    with openai.OpenAI(api_key="dummy-offline",base_url=ac.MODEL_BASE_URL,
            max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(handler),trust_env=False)) as client:
        stream = litellm.completion(**kwargs,model="openai/"+ac.MODEL_ID,base_url=ac.MODEL_BASE_URL,
            api_key="dummy-offline",client=client,messages=[{"role":"user","content":"offline fixture"}],
            num_retries=0,max_retries=0,stream_options={"include_usage":True})
        list(stream)
    assert len(captured) == 1
    body = captured[0]
    assert body["model"] == "k3"
    assert body["reasoning_effort"] == "low"
    assert body["max_tokens"] == 8192
    assert body["stream"] is True
    assert body["temperature"] == 1.0
    assert body["top_p"] == 0.95
