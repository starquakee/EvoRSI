"""Durable observation must cover submit/registration crash windows."""
import httpx
import pytest

from research.adapters.sandbox_eval_client import SandboxEvalClient
from research.contracts.budget import CancellationToken

JOB_ID = 'job_0123456789abcdef'


def completed_body():
    return {'job_id': JOB_ID, 'status': 'completed',
        'started_at': '2026-10-08T14:45:38Z', 'completed_at': '2026-10-08T14:45:40Z',
        'result': {'score': 0.55, 'worker_cleanup_verified': True}}


def install_http(monkeypatch, handler):
    original = httpx.Client
    def factory(*args, **kwargs):
        kwargs['transport'] = httpx.MockTransport(handler)
        return original(*args, **kwargs)
    monkeypatch.setattr(httpx, 'Client', factory)


def test_pending_intent_is_observable_before_post_then_job_before_poll(monkeypatch):
    events = []
    client = SandboxEvalClient('http://test.invalid', 'SYNTHETIC_KEY', require_cleanup=True,
        observer=lambda event: events.append(dict(event)))
    def service(request):
        if request.method == 'POST':
            trace_id = request.headers['x-trace-id']
            # A separate durable intent list is valid; the observer must
            # see the trace identity before the HTTP request can happen.
            assert any(e.get('trace_id') == trace_id for e in events)
            return httpx.Response(200, json={'job_id': JOB_ID})
        if request.method == 'GET':
            assert client.live_job_ids() == frozenset({JOB_ID})
            assert client.uncertain_submission_ids() == frozenset()
            assert any(e.get('job_id') == JOB_ID for e in events)
            return httpx.Response(200, json=completed_body())
        raise AssertionError(request.method)
    install_http(monkeypatch, service)
    result = client.submit_and_wait("print('offline')", cancellation_token=CancellationToken(), task_id='hello_synth')
    assert result.job_id == JOB_ID
    assert client.live_job_ids() == frozenset()
    assert client.uncertain_submission_ids() == frozenset()


def test_registry_write_failure_after_submit_cancels_known_job(monkeypatch):
    cancelled = []
    def observer(event):
        if event.get('event') == 'job_registered':
            raise OSError('synthetic persistent registry write failure')
    client = SandboxEvalClient('http://test.invalid', 'SYNTHETIC_KEY', require_cleanup=True, observer=observer)
    def service(request):
        if request.method == 'POST':
            return httpx.Response(200, json={'job_id': JOB_ID})
        if request.method == 'DELETE':
            cancelled.append(request.url.path.split('/')[-1])
            return httpx.Response(200, json={'status': 'cancelled'})
        if request.method == 'GET':
            return httpx.Response(200, json=completed_body())
        raise AssertionError(request.method)
    install_http(monkeypatch, service)
    with pytest.raises(OSError):
        client.submit_and_wait("print('offline')", cancellation_token=CancellationToken(), task_id='hello_synth')
    assert cancelled == [JOB_ID]
    assert client.live_job_ids() == frozenset()
