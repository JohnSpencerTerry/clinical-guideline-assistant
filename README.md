# Clinical Guideline Assistant

Grounded Q&A over Type 2 Diabetes clinical guidelines (ADA, NICE) — a LangChain/LangGraph learning project. See [mvp.md](mvp.md) for the full design.

## Setup

```bash
cp .env.example .env   # fill in OPENROUTER_API_KEY (required); LANGSMITH_API_KEY is optional
uv sync
```

Model access is via [OpenRouter](https://openrouter.ai) (OpenAI-compatible API) to keep costs low. Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys). **To make sure it can't cost money**: set the key's own credit limit to $0 at [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys), and leave Auto Top-Up off at [openrouter.ai/settings/credits](https://openrouter.ai/settings/credits) — with both, a request can fail but never bill.

`OPENROUTER_MODEL` in `.env` defaults to a free-tier model, but OpenRouter's free lineup rotates and individual free models get temporarily rate-limited or overloaded upstream (see Known Limitations below). Check current options at [openrouter.ai/models](https://openrouter.ai/models) (filter by `:free`) or query `GET https://openrouter.ai/api/v1/models` directly and filter for `pricing.prompt == "0"`.

## Getting the source documents

Loaders read from local files (not live scraping — see mvp.md Phase 1 notes) that you download once:

**ADA** — save each of the 17 *Standards of Care in Diabetes—2026* sections as `data/raw/ada/section_NN.pdf`, from `https://doi.org/10.2337/dc26-s00N` (section 13 is `dc26-S013`, capital S).

**NICE** — save the NG28 PDF as `data/raw/nice/ng28.pdf` from [nice.org.uk/guidance/ng28](https://www.nice.org.uk/guidance/ng28) (resources tab).

Then build the vector indices:

```bash
uv run python -m cga.ingestion.index
```

This embeds locally via `sentence-transformers` (free, no API key, but CPU-bound — can take several minutes depending on hardware).

## Run the chat UI

```bash
uv run streamlit run app/streamlit_app.py
```

## Deploying a free demo (Streamlit Community Cloud)

The built Chroma indices under `data/processed/chroma/` (~47MB) are committed to
this repo, so the deployed app doesn't need the raw PDFs or an ingestion step —
it just loads the index that's already there. (`data/raw/` stays gitignored;
the ADA/NICE source PDFs aren't redistributed.)

1. Push this repo to GitHub (already set up if you're reading this from a clone).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and create a new app:
   - Repository: this repo
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
3. Under **Advanced settings -> Secrets**, paste the contents of
   [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example) with your
   own `OPENROUTER_API_KEY` filled in (same $0-credit-limit key from Setup above).
4. Deploy. First build installs CPU-only `torch` (see `requirements.txt`'s
   `--extra-index-url`) plus the rest of the deps — takes a few minutes.

`requirements.txt` is generated from `uv.lock` via `uv export`, with the
CUDA/nvidia-\* packages stripped and a CPU-only PyTorch index added, since
Streamlit Cloud's free tier has no GPU and would otherwise pull several GB of
unused CUDA wheels. Regenerate it after changing dependencies:

```bash
uv export --no-dev --no-hashes -o requirements.txt
# then strip nvidia-*/cuda-*/triton==* lines and re-add the --extra-index-url line
```

## Run tests

```bash
uv run pytest
```

## Run the eval suite

```bash
uv run python -m eval.run
```

`eval/dataset.py` has a starter set of hand-checked questions (not the full ~70-90 mvp.md targets — see mvp.md's Evaluation section for the full plan). Guardrail categories (`scope_guardrail`, `urgent_symptom_detection`) are graded pass/fail deterministically; grounded-recall/comparison/adversarial categories are run-and-reported for manual review until there's a reference answer to grade against.

## Project layout

- `src/cga/ingestion/` — loaders, chunking, and per-source vector indices (ADA / NICE)
- `src/cga/graph/` — LangGraph nodes: guardrails, retrieval, extraction, comparison, synthesis, `build_graph.py` wiring
- `src/cga/memory/` — conversation checkpointer (in-memory, per-process)
- `app/` — Streamlit chat UI
- `eval/` — eval dataset, evaluators, and runner
- `data/` — raw and processed guideline documents (gitignored)

## Known limitations

- **This is a learning project, not a clinical tool.** The guardrails (emergency/scope detection) are not a substitute for real regulatory, legal, and clinical review.
- **Free-tier model flakiness.** OpenRouter's `:free` models sometimes stall or return transient upstream errors (observed: `502 Service temporarily overloaded`) under back-to-back requests. `get_chat_model()` sets a 60s timeout + 2 retries, and `eval/run.py` adds inter-case delay + call-level retry with backoff — but a fully unattended run can still hit a case that needs a manual re-run. This is upstream provider behavior, not a bug in this codebase (individual isolated calls consistently succeed).
- **Retrieval quality is v1.** With `k=4` and a small local embedding model (`all-MiniLM-L6-v2`), retrieval sometimes misses the most relevant passage (e.g. pulling an unrelated section instead of the specific recommendation). Worth tuning `k`, trying a larger embedding model, or adding a reranking step.
- **Eval set is a starter, not the full plan.** mvp.md targets ~70-90 hand-verified questions across 6 categories; the current set is a smaller, hand-checked sample to prove the framework works — expand it by reading the source PDFs directly.
- **Streamlit UI hasn't been browser-verified.** No headless-browser tooling (`chromium-cli`/Playwright) was available in the dev environment used to build this — the server boots and the identical backend call was verified via CLI, but the chat UI itself (history rendering, badges) should get a manual once-over.
- **Retrieval/extraction/comparison run sequentially per source**, not fanned out in parallel — simpler and correct, but not as fast as it could be (see `build_graph.py`'s docstring).
