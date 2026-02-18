# rss-llm-digest

An autonomous AI agent pipeline that aggregates RSS feeds, summarizes them with an LLM, and delivers formatted digests to **Telegram** or **Email** — orchestrated with LangGraph and LangChain.

---

## AI Architecture

The pipeline is a **multi-node stateful graph** (LangGraph), not a single prompt call. Each stage is a specialized async node connected by a state machine with conditional routing.

```
START
  │
  ▼
summarizer_node   ← ReAct Agent (LangChain) + custom RSS tool + DeepSeek LLM
  │
  ▼
integration_router  ← Conditional edge: routes by CHANNEL_TO_SEND
  │
  ├── telegram_node  ← RAG formatting + LLM reformat + aiogram delivery
  │
  └── email_node     ← MIME message + Gmail SMTP SSL
```

---

## AI Patterns

**ReAct Agent with Tool Calling** — The summarizer node runs a LangChain agent equipped with a custom `@tool` that fetches RSS feeds. The agent autonomously decides what to read, reasons over the content (Thought → Action → Observation loop), and produces structured summaries.

**Graph-Based Orchestration (LangGraph)** — Nodes share typed state via a Pydantic `State` model. Conditional edges handle runtime routing between delivery channels. Adding a new channel means adding one node and one branch.

**Retrieval-Augmented Generation (RAG)** — Before formatting for Telegram, the pipeline fetches Telegram's official HTML spec, chunks and embeds it into a Chroma vector store, then retrieves the relevant formatting rules via semantic search and injects them into the LLM prompt.

**Prompt Engineering** — The LLM behavior is defined in a separate `system_prompt.yaml` with a strict output schema (`TITLE`, `SUMMARY`, `WHY_IT_MATTERS`, `SOURCE`, `TARGET_AUDIENCE`). Decoupled from code for independent versioning.

**Fully Async Pipeline** — RSS fetching, LLM calls, and message delivery all run with `asyncio` and LangChain's async APIs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | DeepSeek (`deepseek-chat`) |
| Agent Framework | LangChain — tool calling agent + AgentExecutor |
| Workflow Orchestration | LangGraph — StateGraph, conditional edges |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | Chroma |
| RSS Parsing | feedparser |
| Telegram Bot | aiogram 3.x (async) |
| Config | Pydantic BaseSettings |
| Logging | loguru |
| Tracing | LangSmith |
| Containerization | Docker + Poetry |

---

## Running

### 1. Configure environment

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=...
LANGSMITH_API_KEY=...      # optional, for LangSmith tracing

CHANNEL_TO_SEND=TELEGRAM   # TELEGRAM | EMAIL

# Telegram
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Email
EMAIL_FROM=...
EMAIL_TO=...
EMAIL_PASSWORD=...         # Gmail app password
```

### 2. Docker (recommended)

```bash
docker build -t rss-llm-digest .
docker run --env-file .env rss-llm-digest
```

### 3. Local

```bash
poetry install
poetry run python agent.py
```
