# rss-llm-digest

An AI engineering lab for **evaluating and iterating on local LLMs** — built around a real-world task: autonomous RSS feed summarization and delivery. The RSS pipeline is the test domain; the evaluation framework is the core engineering work.

---

## Evaluation Framework

Offline evaluation of the summarizer node using LangSmith. Runs independently of the full pipeline — no RSS fetching, no ChromaDB required. Each run creates a named experiment in LangSmith so prompt and model changes can be compared over time.

```
system_prompt.yaml  →  evals.run_eval  →  LangSmith experiment
     ▲                                           │
     └───────────── compare & iterate ───────────┘
```

**Evaluators (`evals/evaluators.py`)** — 5 rule-based, no external LLM:

| Evaluator | What it checks | Score |
|---|---|---|
| `format_compliance` | TITLE: / SUMMARY: / SOURCE: / `---` sections present | 0–1 |
| `has_source_urls` | each SOURCE: line contains a valid http(s) URL | 0–1 |
| `no_large_verbatim_copy` | no >150-char chunks copied verbatim from input | 0 or 1 |
| `correct_field_names` | no markdown bold field names (`**TITLE:**`) | 0 or 1 |
| `no_repeated_summary` | no repeated sentences across article sections | 0 or 1 |

**LLM-as-a-judge (`evals/judge.py`)** — `grounding_judge` uses `Qwen/Qwen3.5-2B` to check whether the summary contains only facts present in the source article. Returns `1.0` (grounded), `0.0` (hallucinated), `0.5` (unparseable response).

**Iteration loop:**

```bash
# Install eval dependencies
poetry install --with evals

# Create the dataset in LangSmith (run once)
poetry run python -m evals.dataset

# Run eval — results in terminal and LangSmith UI
poetry run python -m evals.run_eval
```

Change `system_prompt.yaml` or `HF_MODEL_ID`, re-run, compare experiments in LangSmith.

---

## AI Patterns

**ReAct Agent with Tool Calling** — The summarizer runs a LangChain agent with two custom tools: `rss_feed` (fetches and stores RSS entries) and `search_rss_history` (semantic search over past entries). The agent autonomously decides what to read and what context to retrieve before summarizing.

**Retrieval-Augmented Generation (RAG)** — A single Chroma collection (`news_collection`) accumulates all fetched RSS entries across runs. The agent uses `search_rss_history` to find related past articles and enrich summaries with historical links.

**Graph-Based Orchestration (LangGraph)** — Nodes share typed state via a Pydantic model. Conditional edges handle runtime routing between delivery channels. The graph is the unit of composition — adding a channel means one node and one edge.

**Prompt Engineering with Versioned Evaluation** — LLM behavior is defined in `system_prompt.yaml`, decoupled from code. Each prompt change is evaluated with the full evaluator suite and logged as a named experiment in LangSmith. Changes are comparable across time.

**LLM-as-a-Judge** — A second, larger local model (`Qwen3.5-2B`) evaluates the output of the smaller model (`Qwen3.5-0.8B`) for factual grounding. No external API calls — the judge runs fully locally.

**Scheduled Async Pipeline** — RSS fetching, LLM inference, and message delivery are fully async. The pipeline runs on a daily cron schedule via APScheduler with configurable trigger time.

---

## Architecture (Lab Environment)

The RSS pipeline is a realistic, end-to-end AI task used as the evaluation substrate. It is not the product — it is what the LLMs are being tested against.

```
START
  │
  ▼
summarizer_node   ← ReAct Agent + rss_feed tool + search_rss_history tool + HuggingFace LLM
  │
  ▼
integration_router  ← Conditional edge: routes by CHANNEL_TO_SEND
  │
  ├── telegram_node  ← LLM reformat + aiogram delivery
  │
  └── email_node     ← MIME message + Gmail SMTP SSL
```

**Key components:**

- **Summarizer node** (`agent.py`) — LangChain ReAct agent. HuggingFace pipeline is sync-only; runs inside `loop.run_in_executor`.
- **Vector store** (`memory/vector_store.py`) — ChromaDB over HTTP. `fetched_at` stored as Unix timestamp for Chroma's `$lt` numeric filter. `retrieve_news()` excludes entries from the last 20 minutes to prevent self-retrieval.
- **LLM helper** (`settings/llm.py`) — `get_chat_llm()`: snapshot download → `AutoModelForCausalLM` with `device_map="auto"` → `HuggingFacePipeline` + `ChatHuggingFace`. `@lru_cache` — model loads once, shared across nodes.
- **State** (`integrations/message_state.py`) — Pydantic model: `messages` (LangChain message list) + `summaries` (aggregated text string).

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | HuggingFace Transformers — `Qwen/Qwen3.5-0.8B` (local, configurable via `HF_MODEL_ID`) |
| LLM Judge | `Qwen/Qwen3.5-2B` (local, no external API) |
| Agent Framework | LangChain — ReAct agent + AgentExecutor |
| Workflow Orchestration | LangGraph — StateGraph, conditional edges |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (separate container, persistent) |
| Tracing & Evals | LangSmith |
| RSS Parsing | feedparser |
| Telegram Bot | aiogram 3.x (async) |
| Config | Pydantic BaseSettings |
| Logging | loguru |
| Scheduling | APScheduler (`AsyncIOScheduler` + `CronTrigger`) |
| Containerization | Docker + Docker Compose + Poetry |

---

## Running

### Evaluations (main workflow)

```bash
poetry install --with evals

# Create LangSmith dataset (run once)
poetry run python -m evals.dataset

# Run evaluations
poetry run python -m evals.run_eval
```

Requires `LANGSMITH_API_KEY` in `.env`.

### Full pipeline (local)

```bash
# Start ChromaDB
docker-compose up chroma

# Run the agent
poetry install
poetry run python agent.py
```

### Full pipeline (Docker)

```bash
docker-compose up --build
```

### Environment variables

```env
CHANNEL_TO_SEND=TELEGRAM       # TELEGRAM | EMAIL
SCHEDULE_TIME=08:00            # daily trigger time (HH:MM)

# Telegram
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Email
EMAIL_FROM=...
EMAIL_TO=...
EMAIL_PASSWORD=...             # Gmail app password

# Optional
LANGSMITH_API_KEY=...          # LangSmith tracing and evals
HF_MODEL_ID=Qwen/Qwen3.5-0.8B  # override model (~1.6 GB on first run)
```