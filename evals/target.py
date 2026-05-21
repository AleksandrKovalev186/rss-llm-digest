"""
Target function for LangSmith evaluation.

This is the function that evaluate() calls for each example in the dataset.
It receives pre-fetched article text (simulating rss_feed tool output)
and calls the LLM directly with the system prompt — no RSS fetching, no ChromaDB.

Why bypass the full agent graph:
  - The graph fetches live RSS and queries ChromaDB. Both are non-deterministic
    and would make the eval unreproducible.
  - We want to measure LLM summarization quality in isolation.
  - The system prompt is the same — only tool calling is skipped.
"""

from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from settings.llm import get_chat_llm


def _load_system_prompt() -> str:
    data = yaml.safe_load(Path("system_prompt.yaml").read_text())
    messages_data = {msg["role"]: msg["content"] for msg in data["messages"]}
    return messages_data["system"]


def summarize(inputs: dict) -> dict:
    """Call the LLM with the system prompt and the pre-fetched article text."""
    llm = get_chat_llm()
    system_prompt = _load_system_prompt()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=inputs["articles_text"]),
    ]

    response = llm.invoke(messages)

    return {"summary": response.content}
