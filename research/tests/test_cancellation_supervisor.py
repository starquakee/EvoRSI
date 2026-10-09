"""Cancellation acknowledgements must survive missing or delayed cleanup."""
import httpx
import pytest

from research.adapters.sandbox_eval_client import SandboxEvalClient
from research.contracts.budget import CancellationToken, Cancelled
from research.contracts.sandbox_lifecycle import cleanup_confirmed
from test_adapter_cancellation import fake_http, FakeResponse, BENIGN_SOURCE, _client


def test_cancelled_status_waits_for_dispatcher_cleanup(fake_http):
    fake_http.get_payloads=[{'status':'cancelled'},
        {'status':'cancelled','completed_at':'2026-10-08T00:00:00Z',
         'result':{'worker_cleanup_verified':True}}]
    client=_client();client._register('job-a')
    evidence=client.cancel_job('job-a')
    assert evidence['cleanup_verified'] is True
    assert sum(method=='GET' for method,url in fake_http.calls)==2
    assert not client.live_job_ids()


def test_cancel_failure_does_not_discard_live_job(fake_http):
    fake_http.delete_raises=httpx.ConnectError('offline')
    client=_client();client._register('job-a')
    results=client.cancel_all_live()
    assert results[0]['cleanup_verified'] is False
    assert client.live_job_ids()==frozenset({'job-a'})


def test_unknown_submission_is_preserved_as_unverified(fake_http,monkeypatch):
    def lost(*args,**kwargs):
        raise httpx.ReadTimeout('response lost')
    monkeypatch.setattr(fake_http,'post',lost)
    client=_client()
    with pytest.raises(httpx.ReadTimeout):
        client.submit_and_wait(BENIGN_SOURCE)
    results=client.cancel_all_live()
    assert len(results)==1
    assert results[0]['cancel_error']=='submission_identity_unknown'
    assert results[0]['cleanup_verified'] is False


def test_token_stop_during_poll_cancels_before_returning(fake_http,monkeypatch):
    token=CancellationToken()
    original=fake_http.get
    def stop_on_poll(self,url,headers=None):
        if token.stopped_reason is None:
            token.stop('supervisor')
            return FakeResponse({'status':'running'})
        return original(self,url,headers=headers)
    monkeypatch.setattr(fake_http,'get',stop_on_poll)
    client=_client()
    with pytest.raises(Cancelled):
        client.submit_and_wait(BENIGN_SOURCE,cancellation_token=token,poll_interval=0)
    assert ('DELETE','/api/v1/jobs/job-1') in fake_http.calls
    assert not client.live_job_ids()


@pytest.mark.parametrize('body',[
    {'status':'cancelled'},
    {'status':'cancelled','result':{'worker_cleanup_verified':True}},
    {'status':'cancelled','completed_at':'done','result':{'worker_cleanup_verified':'true'}},
    {'status':'cancelled','completed_at':'done','started_at':'started',
     'result':{'execution_never_started':True}},
])
def test_status_or_malformed_proof_cannot_confirm_cleanup(body):
    assert not cleanup_confirmed(body)
