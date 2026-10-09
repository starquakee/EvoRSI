"""Owned auth resources close even when post-search evidence IO fails."""
import pytest

from research.search import acceptance_inner as ai
from test_acceptance_entry_supervisor import (
    test_production_inner_entry_preserves_request_job_and_budget_evidence as _exercise_entry,
)


def test_owned_auth_helper_closed_after_artifact_finalization_error(tmp_path, monkeypatch):
    closed = []
    class Helper:
        def refresh(self):
            raise AssertionError('Fresh fixture credentials must not refresh')
        def close(self):
            closed.append(True)
    monkeypatch.setattr(ai, 'KimiAuthHelper', Helper)
    def fail_binding(*args, **kwargs):
        raise ai.AcceptanceError('synthetic_prediction_read_failure')
    monkeypatch.setattr(ai, 'bind_prediction_snapshots', fail_binding)
    with pytest.raises(ai.AcceptanceError, match='synthetic_prediction_read_failure'):
        _exercise_entry(tmp_path, monkeypatch, False, False)
    assert closed == [True]
