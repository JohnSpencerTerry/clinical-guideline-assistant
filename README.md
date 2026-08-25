# Clinical Guideline Assistant

Grounded Q&A over Type 2 Diabetes clinical guidelines (ADA, NICE, IDF) — a LangChain/LangGraph learning project. See [mvp.md](mvp.md) for the full design.

## Setup

```bash
cp .env.example .env   # fill in LANGSMITH_API_KEY and OPENROUTER_API_KEY
uv sync
```

Model access is via [OpenRouter](https://openrouter.ai) (OpenAI-compatible API) to keep costs low — `OPENROUTER_MODEL` in `.env` defaults to a free-tier model, but OpenRouter's free lineup rotates, so check [openrouter.ai/models](https://openrouter.ai/models) (filter by `:free`) before relying on the default.

## Run the chat UI

```bash
uv run streamlit run app/streamlit_app.py
```

## Run tests

```bash
uv run pytest
```

## Project layout

- `src/cga/ingestion/` — loaders, chunking, and per-source vector indices (ADA / NICE / IDF)
- `src/cga/graph/` — LangGraph nodes: guardrails, retrieval, extraction, comparison, synthesis
- `src/cga/memory/` — conversation checkpointer
- `app/` — Streamlit chat UI
- `eval/` — LangSmith eval dataset and evaluators
- `data/` — raw and processed guideline documents (gitignored)
