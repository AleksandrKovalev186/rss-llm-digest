"""
LLM-as-a-judge evaluator for the RSS summarizer.

Uses a separate, larger model (Qwen/Qwen3.5-2B by default) to assess
whether the generated summary is grounded in the source article —
i.e., does not contain facts absent from the input.

Why a separate model (not get_chat_llm):
  - The judge must be independent of the model being evaluated.
  - It is loaded once via load_chat_llm's cache, keyed by model id,
    so it does not replace the main model.

Judge prompt design:
  - Binary YES/NO answer to keep parsing simple and reliable.
  - Chain-of-thought reasoning line before the verdict so the model
    "thinks" before answering (improves accuracy on small models).
  - Explicit instruction to ignore format issues — we only judge facts.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from settings.llm import load_chat_llm

JUDGE_MODEL_ID = "Qwen/Qwen3.5-2B"

JUDGE_SYSTEM_PROMPT = """You are a factual grounding evaluator.
Your task: decide whether a news summary contains ONLY facts that are present in the source article.

Rules:
- Ignore all formatting: field names (TITLE, SUMMARY, WHY_IT_MATTERS, SOURCE, TARGET_AUDIENCE),
  separators, and structure are NOT facts — do not flag them.
- Focus only on the actual content: sentences and bullet points inside each field.
- If any sentence states a fact, claim, or inference NOT present in the source — answer NO.
- If all content can be traced back to the source text — answer YES.
- "Limited information available" or "No content preview available" in summary — answer YES.
- Do NOT check whether the TITLE matches word-for-word — minor rephrasing is acceptable.

Respond in this exact format:
REASONING: <one sentence explaining your decision>
VERDICT: <YES or NO>"""

VERDICT_PATTERN = re.compile(r"VERDICT:\s*(YES|NO)", re.IGNORECASE)


def grounding_judge(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    Judge whether the summary is factually grounded in the source article.

    Score 1.0 = all facts in the summary come from the source (VERDICT: YES).
    Score 0.0 = summary contains hallucinated facts (VERDICT: NO).
    Score 0.5 = source too sparse to verify, or judge response could not be parsed.
    """
    article_text = inputs.get("articles_text", "")
    summary = outputs.get("summary", "")

    # Can't verify grounding when the source has no real content.
    if len(article_text.strip()) < 50 or "No content" in article_text:
        return {"key": "grounding_judge", "score": 0.5}

    judge = load_chat_llm(JUDGE_MODEL_ID, 120)
    response = judge.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=f"SOURCE ARTICLE:\n{article_text}\n\nSUMMARY:\n{summary}"),
    ])

    match = VERDICT_PATTERN.search(response.content)
    if not match:
        # Could not parse — return neutral score so it doesn't skew the mean.
        return {"key": "grounding_judge", "score": 0.5}

    verdict = match.group(1).upper()
    return {"key": "grounding_judge", "score": 1.0 if verdict == "YES" else 0.0}
