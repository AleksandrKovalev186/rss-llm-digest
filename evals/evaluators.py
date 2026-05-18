"""
Rule-based evaluators for the RSS summarizer.

Each evaluator receives three keyword arguments from langsmith.evaluate():
  - inputs          : dict — the Example inputs (contains "articles_text")
  - outputs         : dict — what the target function returned (contains "summary")
  - reference_outputs: dict — the Example outputs from the dataset (contains "description")

Each evaluator must return a dict with:
  - "key"   : str   — the metric name shown in LangSmith UI
  - "score" : float — 0.0 (bad) to 1.0 (good); binary 0/1 is fine too

No external LLM is used here — all checks are rule-based string analysis.
This makes the evals fast, cheap, and fully offline.
"""

import re


# ---------------------------------------------------------------------------
# Evaluator 1: format_compliance
# ---------------------------------------------------------------------------
# The system prompt requires each article to have these sections:
#   TITLE: / SUMMARY: / SOURCE: / --- (separator between articles)
# If the LLM skips any of these, it violated the output format rules.

REQUIRED_SECTIONS = ["TITLE:", "SUMMARY:", "SOURCE:"]
SEPARATOR = "---"


def format_compliance(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Check that the summary contains all required structural sections."""
    summary = outputs.get("summary", "")

    missing = [s for s in REQUIRED_SECTIONS if s not in summary]
    has_separator = SEPARATOR in summary

    # Score: 1.0 only if nothing is missing AND separator is present.
    # Partial credit: (found_sections / total_required) — penalises each missing section.
    sections_score = (len(REQUIRED_SECTIONS) - len(missing)) / len(REQUIRED_SECTIONS)
    separator_score = 1.0 if has_separator else 0.0
    score = (sections_score + separator_score) / 2

    return {"key": "format_compliance", "score": round(score, 2)}


# ---------------------------------------------------------------------------
# Evaluator 2: has_source_urls
# ---------------------------------------------------------------------------
# Every SOURCE: line must contain a real URL (http or https).
# This catches cases where the LLM writes "SOURCE: See article" instead of the link.

URL_PATTERN = re.compile(r"https?://\S+")


def has_source_urls(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Check that every SOURCE: line contains a valid http(s) URL."""
    summary = outputs.get("summary", "")

    source_lines = [line for line in summary.splitlines() if line.strip().startswith("SOURCE:")]

    if not source_lines:
        # No SOURCE: lines at all — format_compliance already penalises this,
        # but we also return 0 here.
        return {"key": "has_source_urls", "score": 0.0}

    lines_with_url = sum(1 for line in source_lines if URL_PATTERN.search(line))
    score = lines_with_url / len(source_lines)

    return {"key": "has_source_urls", "score": round(score, 2)}


# ---------------------------------------------------------------------------
# Evaluator 3: no_large_verbatim_copy
# ---------------------------------------------------------------------------
# System prompt rule: "Do not copy large fragments of the original text."
# We check: does the summary contain any substring of >150 chars that also
# appears verbatim in the input article text?
# Score 1.0 = no copying found. Score 0.0 = at least one large copy found.

VERBATIM_THRESHOLD = 150  # characters


def no_large_verbatim_copy(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Penalise verbatim chunks longer than VERBATIM_THRESHOLD chars copied from input."""
    article_text = inputs.get("articles_text", "")
    summary = outputs.get("summary", "")

    # Slide a window of VERBATIM_THRESHOLD chars over the summary and
    # check if each window also appears in the article text.
    violations = 0
    step = 50  # check every 50 chars to keep it fast

    for i in range(0, max(0, len(summary) - VERBATIM_THRESHOLD), step):
        chunk = summary[i:i + VERBATIM_THRESHOLD]
        if chunk in article_text:
            violations += 1
            break  # one violation is enough to fail

    score = 0.0 if violations > 0 else 1.0
    return {"key": "no_large_verbatim_copy", "score": score}
