"""Generation-outcome classification tests (offline, no model/sandbox).

Fixtures mirror the retained US-009 runtime evidence: 9 of 16 records were
finish_reason=length at the 4096 cap — some with empty visible content and
~4093 reasoning tokens, others with one unclosed code fence — and every one
extracted empty at the actual execution seam.
"""
from __future__ import annotations

import pytest

from research.search.generation_outcome import (
    OUTCOME_SCHEMA,
    build_outcome_record,
    classify_completion,
    classify_extraction_failure,
    extract_candidate_code,
    feedback_for_failure,
    numeric_or_none,
    strip_thinking,
    summarize_usage,
)

#: Minimized versions of the retained truncation patterns.
EMPTY_VISIBLE_LENGTH = {
    "text": "",
    "metrics": {
        "finish_reason": "length",
        "prompt_tokens": 530,
        "completion_tokens": 4096,
        "completion_tokens_details": {"reasoning_tokens": 4093},
        "usage_provenance": "provider",
    },
    "classification": "empty_visible_content",
}
UNCLOSED_FENCE_LENGTH = {
    "text": "Here is the improved program — with prose.\n```python\nimport pandas as pd\ndef train():\n    return 1\n",
    "metrics": {
        "finish_reason": "length",
        "prompt_tokens": 2536,
        "completion_tokens": 4096,
        "usage_provenance": "provider",
    },
    "classification": "unclosed_fence",
}
PROSE_BLOCK = {
    "text": "Plan first.\n```python\nThis — is not python at all\n```\n",
    "metrics": {"finish_reason": "stop", "usage_provenance": "provider"},
    "classification": "invalid_python",
}
VALID_PROGRAM = {
    "text": "A compact plan.\n```python\nprint('candidate')\n```\n",
    "metrics": {
        "finish_reason": "stop",
        "prompt_tokens": 659,
        "completion_tokens": 2669,
        "usage_provenance": "provider",
    },
    "classification": "ok",
}


@pytest.mark.parametrize(
    "fixture",
    [EMPTY_VISIBLE_LENGTH, UNCLOSED_FENCE_LENGTH, PROSE_BLOCK, VALID_PROGRAM],
    ids=lambda f: f["classification"],
)
def test_classify_completion(fixture):
    classification, extracted = classify_completion(fixture["text"])
    assert classification == fixture["classification"]
    assert bool(extracted) == (fixture["classification"] == "ok")


def test_length_truncated_records_are_never_collapsed_into_unexplained_empty():
    """The 9 retained length-truncated outcomes must keep their precise
    classification, usage and finish_reason instead of a bare
    ``empty_candidate_code``."""
    for fixture in (EMPTY_VISIBLE_LENGTH, UNCLOSED_FENCE_LENGTH):
        classification, extracted = classify_completion(fixture["text"])
        record = build_outcome_record(
            run_id="cfg0",
            sequence=1,
            operator="draft",
            visible_text=fixture["text"],
            extracted_code=extracted,
            classification=classification,
            metrics=fixture["metrics"],
            artifacts={},
        )
        assert record["schema"] == OUTCOME_SCHEMA
        assert record["truncated"] is True
        assert record["classification"] != "ok"
        assert record["usage"]["finish_reason"] == "length"
        feedback = feedback_for_failure(record)
        assert "finish_reason=length" in feedback
        assert fixture["classification"] in feedback


def test_empty_visible_keeps_reasoning_token_count():
    record = build_outcome_record(
        run_id="cfg0", sequence=1, operator="crossover",
        visible_text="", extracted_code="",
        classification="empty_visible_content",
        metrics=EMPTY_VISIBLE_LENGTH["metrics"], artifacts={},
    )
    assert record["usage"]["prompt_tokens"] == 530
    assert record["usage"]["completion_tokens"] == 4096
    assert record["usage"]["reasoning_tokens"] == 4093
    assert record["visible_chars"] == 0
    assert record["extracted_sha256"] is None
    assert "reasoning_tokens=4093" in feedback_for_failure(record)


def test_node_code_hash_matches_journal_code_on_both_paths():
    ok_classification, extracted = classify_completion(VALID_PROGRAM["text"])
    ok_record = build_outcome_record(
        run_id="r", sequence=1, operator="draft", visible_text=VALID_PROGRAM["text"],
        extracted_code=extracted, classification=ok_classification,
        metrics=VALID_PROGRAM["metrics"], artifacts={},
    )
    # On success the node stores the extracted code.
    from research.search.generation_outcome import sha256_text

    assert ok_record["node_code_sha256"] == sha256_text(extracted)
    fail_record = build_outcome_record(
        run_id="r", sequence=2, operator="draft", visible_text=UNCLOSED_FENCE_LENGTH["text"],
        extracted_code="", classification="unclosed_fence",
        metrics=UNCLOSED_FENCE_LENGTH["metrics"], artifacts={},
    )
    # On failure the node stores the raw visible text (production behavior).
    assert fail_record["node_code_sha256"] == sha256_text(UNCLOSED_FENCE_LENGTH["text"])


def test_extraction_never_salvages_incomplete_python():
    """No fabricated suffix: an unclosed fence yields nothing executable."""
    assert extract_candidate_code(UNCLOSED_FENCE_LENGTH["text"]) == ""
    assert extract_candidate_code("") == ""
    assert extract_candidate_code("no fences, just prose — not code") == ""


def test_extraction_fallback_accepts_plain_code_text():
    assert extract_candidate_code("print('plain code, no fences')") != ""


def test_invalid_python_block_rejected():
    assert extract_candidate_code(PROSE_BLOCK["text"]) == ""
    assert classify_extraction_failure(PROSE_BLOCK["text"]) == "invalid_python"


def test_no_complete_code_block_classification():
    assert classify_extraction_failure("just some prose without fences —") == (
        "no_complete_code_block"
    )


def test_strip_thinking_variants():
    assert strip_thinking("<think>secret plan</think>print(1)") == "print(1)"
    assert strip_thinking("hidden</think>print(2)") == "print(2)"
    assert strip_thinking("print(3)") == "print(3)"


@pytest.mark.parametrize(
    "value,expected",
    [(0, 0), (4096, 4096), (True, None), (False, None), ("530", None), (1.5, None), (-1, None), (None, None)],
)
def test_numeric_or_none_strict(value, expected):
    assert numeric_or_none(value) == expected


def test_usage_summary_rejects_non_numeric_and_unknown_provenance():
    usage = summarize_usage({
        "finish_reason": 42,
        "prompt_tokens": "many",
        "completion_tokens": True,
        "completion_tokens_details": {"reasoning_tokens": "lots"},
        "usage_provenance": "whitespace-estimate",
    })
    assert usage == {
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "usage_provenance": None,
    }


def test_unknown_classification_fails_closed():
    with pytest.raises(ValueError, match="outcome_classification_unknown"):
        build_outcome_record(
            run_id="r", sequence=1, operator="draft", visible_text="x",
            extracted_code="", classification="mystery", metrics={}, artifacts={},
        )
