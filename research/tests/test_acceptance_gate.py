"""US-009 live-gate tests: no ledger/auth/model before the review binding.

The gate exists to protect the one-shot real budget from unreviewed runner
code. Tests manufacture review payloads ONLY in tmp paths; the real
``.runtime/us009-runner-review.json`` is never created here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.search import acceptance as ap
from research.search import acceptance_config as ac
from research.search.acceptance_config import (
    RunnerReviewRefused,
    check_runner_review,
    evaluate_runner_review,
)

HEAD = "a" * 40
REPO_ROOT = Path(__file__).resolve().parents[2]


def _review(**overrides):
    payload = {
        "schema": ac.RUNNER_REVIEW_SCHEMA,
        "implementation_commit": HEAD,
        "approve_real_acceptance": True,
    }
    payload.update(overrides)
    return payload


# ------------------------------------------------------------- pure checks
def test_valid_review_passes():
    evaluate_runner_review(_review(), head_commit=HEAD)


@pytest.mark.parametrize(
    "overrides,rule",
    [
        ({"schema": "other"}, "runner_review_schema_mismatch"),
        ({"schema": None}, "runner_review_schema_mismatch"),
        ({"implementation_commit": "b" * 40}, "runner_review_commit_mismatch"),
        ({"implementation_commit": ""}, "runner_review_commit_mismatch"),
        ({"implementation_commit": None}, "runner_review_commit_mismatch"),
        ({"approve_real_acceptance": False}, "runner_review_not_approved"),
        ({"approve_real_acceptance": "true"}, "runner_review_not_approved"),
        ({"approve_real_acceptance": 1}, "runner_review_not_approved"),
    ],
)
def test_review_refusals(overrides, rule):
    with pytest.raises(RunnerReviewRefused, match=rule):
        evaluate_runner_review(_review(**overrides), head_commit=HEAD)


def test_review_not_a_mapping():
    with pytest.raises(RunnerReviewRefused, match="runner_review_not_an_object"):
        evaluate_runner_review(["not", "a", "mapping"], head_commit=HEAD)


# ------------------------------------------------------------- IO wrapper
def test_missing_review_file_refused(tmp_path):
    with pytest.raises(RunnerReviewRefused, match="runner_review_unavailable"):
        check_runner_review(REPO_ROOT, review_path=tmp_path / "absent.json")


def test_malformed_review_file_refused(tmp_path):
    path = tmp_path / "review.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RunnerReviewRefused, match="runner_review_unavailable"):
        check_runner_review(REPO_ROOT, review_path=path)


def test_binding_review_file_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "git_head_commit", lambda root: HEAD)
    monkeypatch.setattr(ac, "git_tree_dirty", lambda root: False)
    path = tmp_path / "review.json"
    path.write_text(json.dumps(_review()), encoding="utf-8")
    check_runner_review(REPO_ROOT, review_path=path)


def test_dirty_tree_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "git_head_commit", lambda root: HEAD)
    monkeypatch.setattr(ac, "git_tree_dirty", lambda root: True)
    path = tmp_path / "review.json"
    path.write_text(json.dumps(_review()), encoding="utf-8")
    with pytest.raises(RunnerReviewRefused, match="runner_review_dirty_tree"):
        check_runner_review(REPO_ROOT, review_path=path)


def test_git_helpers_real_repo():
    commit = ac.git_head_commit(REPO_ROOT)
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    assert isinstance(ac.git_tree_dirty(REPO_ROOT), bool)


# ------------------------------------------------- live entrypoint ordering
def test_live_refuses_before_ledger_auth_or_model(monkeypatch, tmp_path):
    """Without the binding review file, live mode must refuse BEFORE any
    ledger creation, credential read or child/model activity."""
    import research.search.credentials as creds

    def forbidden_token(self):  # pragma: no cover - must never run
        raise AssertionError("credential read attempted before review gate")

    def forbidden_child(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("child runner invoked before review gate")

    monkeypatch.setattr(creds.ManagedKimiCredentials, "access_token", forbidden_token)
    monkeypatch.setattr(ap, "_default_child_runner", forbidden_child)

    ledger = REPO_ROOT / ac.RUNTIME_DIR / ac.LEDGER_FILENAME
    # Post-run reality: the original stopped acceptance ledger EXISTS and is
    # immutable. The refusal must happen before touching it (or before
    # creating anything, in a fresh tree).
    ledger_bytes_before = ledger.read_bytes() if ledger.exists() else None
    with pytest.raises(RunnerReviewRefused, match="runner_review_"):
        ap.live(REPO_ROOT)
    if ledger_bytes_before is None:
        assert not ledger.exists()
        assert not (REPO_ROOT / ac.RUNTIME_DIR).exists()
    else:
        assert ledger.read_bytes() == ledger_bytes_before


def test_main_live_reports_machine_readable_refusal(capsys):
    code = ap.main(["live", "--repo-root", str(REPO_ROOT)])
    assert code == 2
    out = json.loads(capsys.readouterr().out)
    # Without a review file binding the CURRENT HEAD, live always refuses;
    # the exact rule depends on whether a stale binding exists on disk.
    assert out["refused"].startswith("runner_review_")
