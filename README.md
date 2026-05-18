# rss-llm-digest

An autonomous AI agent pipeline that aggregates RSS feeds, summarizes them with an LLM, and delivers formatted digests to **Telegram** or **Email** — orchestrated with LangGraph and LangChain.

---

## AI Architecture

The pipeline is a **multi-node stateful graph** (LangGraph), not a single prompt call. Each stage is a specialized async node connected by a state machine with conditional routing.

```
START
  │
  ▼
summarizer_node   ← ReAct Agent (LangChain) + rss_feed tool + search_rss_history tool + HuggingFace LLM (Qwen3.5)
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

**ReAct Agent with Tool Calling** — The summarizer node runs a LangChain agent equipped with two custom tools: `rss_feed` (fetches and stores RSS entries) and `search_rss_history` (searches past entries by semantic similarity). The agent autonomously decides what to read and what context to retrieve before summarizing. The HuggingFace pipeline is sync-only, so it runs inside `loop.run_in_executor`.

**Graph-Based Orchestration (LangGraph)** — Nodes share typed state via a Pydantic `State` model. Conditional edges handle runtime routing between delivery channels. Adding a new channel means adding one node and one branch.

**Retrieval-Augmented Generation (RAG)** — Single Chroma collection `news_collection` accumulates all fetched RSS entries across runs. The agent uses `search_rss_history` to find related past articles and enrich summaries with historical links.

**Prompt Engineering** — The LLM behavior is defined in a separate `system_prompt.yaml` with a strict output schema (`TITLE`, `SUMMARY`, `WHY_IT_MATTERS`, `SOURCE`, `TARGET_AUDIENCE`). Decoupled from code for independent versioning.

**Offline Evaluation (LangSmith Evals)** — The summarizer LLM is evaluated independently from the full pipeline using a static dataset of pre-fetched articles stored in LangSmith. Three rule-based evaluators measure format compliance, URL presence in `SOURCE:` fields, and absence of verbatim copy-paste from input. Each `poetry run python -m evals.run_eval` creates a new named experiment in LangSmith — experiments accumulate over time so prompt changes can be compared across runs.

**Scheduled Execution** — The pipeline runs on a daily cron schedule via APScheduler (`AsyncIOScheduler` + `CronTrigger`). Send time is configured via `SCHEDULE_TIME` env var. The process runs continuously and waits for the next trigger.

**Fully Async Pipeline** — RSS fetching, LLM calls, and message delivery all run with `asyncio` and LangChain's async APIs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | HuggingFace Transformers — `Qwen/Qwen3.5-0.8B` (local, configurable via `HF_MODEL_ID`) |
| Agent Framework | LangChain Classic — tool calling agent + AgentExecutor |
| Workflow Orchestration | LangGraph — StateGraph, conditional edges |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | Chroma (separate container, two persistent collections) |
| RSS Parsing | feedparser |
| Telegram Bot | aiogram 3.x (async) |
| Config | Pydantic BaseSettings |
| Logging | loguru |
| Tracing & Evals | LangSmith |
| Scheduling | APScheduler (`AsyncIOScheduler` + `CronTrigger`) |
| Containerization | Docker + Docker Compose + Poetry |

---

## Running

### 1. Configure environment

Create a `.env` file in the project root:

```env
CHANNEL_TO_SEND=TELEGRAM   # TELEGRAM | EMAIL
SCHEDULE_TIME=08:00        # daily digest time (HH:MM, system timezone)

# Telegram
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Email
EMAIL_FROM=...
EMAIL_TO=...
EMAIL_PASSWORD=...         # Gmail app password

# Optional
LANGSMITH_API_KEY=...      # LangSmith tracing
HF_MODEL_ID=Qwen/Qwen3.5-0.8B  # override default model (~1.6 GB)
```

### 2. Docker Compose (recommended)

```bash
docker-compose up --build
```

### 3. Local development

Run ChromaDB in Docker, app locally:

```bash
# Start ChromaDB
docker-compose up chroma

# Run the app
poetry install
poetry run python agent.py
```

### 4. Run evaluations

Requires `LANGSMITH_API_KEY` in `.env`.

```bash
# Install eval dependencies
poetry install --with evals

# Create the dataset in LangSmith (run once)
poetry run python -m evals.dataset

# Run eval — results appear in terminal and in LangSmith UI
poetry run python -m evals.run_eval
```

The eval runs the summarizer LLM against 4 static article examples and scores each output with three metrics: `format_compliance`, `has_source_urls`, `no_large_verbatim_copy`. Results are stored as a named experiment in LangSmith so you can compare runs after changing `system_prompt.yaml` or the model.
