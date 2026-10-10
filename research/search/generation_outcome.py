"""Precise per-generation outcome records for the acceptance run (US-009).

Stdlib-only on purpose: imported by the inner Evo child, the research test
suite and (indirectly) the dojo seam. The real US-009 run collapsed every
length-truncated, empty-visible or malformed completion into an unexplained
``empty_candidate_code``; this module classifies the ACTUAL outcome so the
evidence distinguishes:

- ``ok`` — a complete, compile-valid Python program was extracted;
- ``empty_visible_content`` — the provider returned no visible content
  (typically the whole output budget went to reasoning tokens);
- ``unclosed_fence`` — the visible output contains an odd number of code
  fences (truncated mid-block), so no complete block exists;
- ``invalid_python`` — fenced block(s) exist but none compiles (prose or
  non-Python characters inside the block);
- ``no_complete_code_block`` — no fenced block and the whole text is not a
  compile-valid program either.

Nothing here fabricates code: an incomplete program is NEVER salvaged by
appending a synthetic suffix, and invalid code is never executed. The
extraction mirror (:func:`extract_candidate_code`) replicates the semantics
of ``dojo.core.solvers.utils.response.extract_code`` (same regexes, same
compile-validity filter, minus the cosmetic black formatting) so the
classification matches what the production seam actually accepts; an Evo-venv
test pins that equivalence against the real function.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

OUTCOME_SCHEMA = "generation-outcome.v1"

#: Extraction classifications (``ok`` is the only executable outcome).
CLASSIFICATIONS = (
    "ok",
    "empty_visible_content",
    "unclosed_fence",
    "invalid_python",
    "no_complete_code_block",
)

_FENCED_RE = re.compile(r"```(python)?\n*(.*?)\n*```", re.DOTALL)
_WHOLE_TEXT_RE = re.compile(r"^(```(python)?)?\n?(.*?)\n?(```)?$", re.DOTALL)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compiles(script: str) -> bool:
    try:
        compile(script, "<string>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def strip_thinking(text: str) -> str:
    """Mirror dojo's parse_thinking_tags: return the visible (non-thinking)
    part of a completion, handling missing opening/closing think tags."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return text.replace(match.group(0), "").strip()
    match = re.search(r"(.*?)</think>", text, re.DOTALL)
    if match:
        return text.replace(match.group(0), "").strip()
    return text.strip()


def extract_candidate_code(text: str) -> str:
    """Compile-valid code blocks exactly as the production seam accepts them.

    Same block/fallback regexes and the same ``compile`` validity filter as
    ``dojo.core.solvers.utils.response.extract_code``; the only difference is
    that black reformatting is skipped (classification does not depend on
    formatting). Returns "" when no compile-valid code exists.
    """
    parsed = [match[1] for match in _FENCED_RE.findall(text)]
    if not parsed:
        matches = _WHOLE_TEXT_RE.findall(text)
        if matches:
            parsed.append(matches[0][2])
    valid = [block for block in parsed if _compiles(block)]
    return "\n\n".join(valid)


def classify_extraction_failure(visible_text: str) -> str:
    """Machine-readable reason a completion yielded no executable code."""
    if not visible_text.strip():
        return "empty_visible_content"
    if visible_text.count("```") % 2 == 1:
        # An odd fence count means the final block never closed (observed
        # truncation pattern); the fallback may still capture prose as
        # "code", but it cannot be a complete intended program.
        return "unclosed_fence"
    if _FENCED_RE.findall(visible_text):
        return "invalid_python"
    return "no_complete_code_block"


def classify_completion(visible_text: str) -> tuple[str, str]:
    """Classify one completion. Returns (classification, extracted_code)."""
    extracted = extract_candidate_code(visible_text)
    if extracted.strip():
        return "ok", extracted
    return classify_extraction_failure(visible_text), ""


def numeric_or_none(value: Any) -> int | None:
    """Token counts are numeric only; anything else (bool, str, float,
    missing) is unreported, never coerced."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def summarize_usage(metrics: Any) -> dict[str, Any]:
    """Provider usage subset for the outcome record: finish_reason plus
    integral prompt/completion/reasoning token counts (numeric only).

    Accepts both the flat usage-stats dict the operator seam carries and the
    GenericLLM envelope (``{"usage": {...}, ...}``). Reasoning CONTENT is
    never copied — only the numeric reasoning token count."""
    envelope: Mapping[str, Any] = metrics if isinstance(metrics, Mapping) else {}
    nested = envelope.get("usage")
    source: Mapping[str, Any] = nested if isinstance(nested, Mapping) else envelope
    finish = source.get("finish_reason")
    completion_details = source.get("completion_tokens_details")
    reasoning = None
    if isinstance(completion_details, Mapping):
        reasoning = numeric_or_none(completion_details.get("reasoning_tokens"))
    return {
        "finish_reason": finish if isinstance(finish, str) and finish else None,
        "prompt_tokens": numeric_or_none(source.get("prompt_tokens")),
        "completion_tokens": numeric_or_none(source.get("completion_tokens")),
        "reasoning_tokens": reasoning,
        "usage_provenance": (
            source.get("usage_provenance")
            if source.get("usage_provenance") in ("provider", "estimated")
            else None
        ),
    }


def build_outcome_record(
    *,
    run_id: str,
    sequence: int,
    operator: str,
    visible_text: str,
    extracted_code: str,
    classification: str,
    metrics: Any,
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """One generation-outcome record (hashes + counts, never credentials)."""
    if classification not in CLASSIFICATIONS:
        raise ValueError("outcome_classification_unknown")
    node_code = extracted_code if classification == "ok" else visible_text
    usage = summarize_usage(metrics)
    return {
        "schema": OUTCOME_SCHEMA,
        "run_id": run_id,
        "sequence": sequence,
        "operator": operator,
        "classification": classification,
        "truncated": usage["finish_reason"] == "length",
        "usage": usage,
        "visible_chars": len(visible_text),
        "visible_sha256": sha256_text(visible_text),
        "extracted_chars": len(extracted_code),
        "extracted_sha256": sha256_text(extracted_code) if extracted_code else None,
        # What the journal node will carry as its code (extracted on success,
        # the raw visible text on failure) — matches the operator trace's
        # child_code_sha256 for cross-binding.
        "node_code_sha256": sha256_text(node_code),
        "artifacts": dict(artifacts),
    }


def feedback_for_failure(record: Mapping[str, Any]) -> str:
    """Bounded, precise feedback carried into the debug prompt instead of an
    unexplained ``empty_candidate_code``."""
    parts = [str(record.get("classification") or "unknown")]
    usage_raw = record.get("usage")
    usage: Mapping[str, Any] = usage_raw if isinstance(usage_raw, Mapping) else {}
    finish = usage.get("finish_reason")
    if finish:
        parts.append(f"finish_reason={finish}")
    parts.append(f"visible_chars={record.get('visible_chars', 0)}")
    reasoning = usage.get("reasoning_tokens")
    if reasoning is not None:
        parts.append(f"reasoning_tokens={reasoning}")
    return "; ".join(parts)
