# Deploying the demo to Streamlit Community Cloud

Free hosting for the chat UI, using the Chroma indices already committed at
`data/processed/chroma/` — no PDF ingestion step runs on the host.

## Prerequisites

- This repo pushed to GitHub (`origin` is already `github.com/JohnSpencerTerry/clinical-guideline-assistant`).
- An OpenRouter API key with its credit limit set to $0 (see [README.md](README.md#setup)) — the free tier can fail a request but never bill you.

## Steps

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app** and fill in:
   - **Repository**: `JohnSpencerTerry/clinical-guideline-assistant`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`
3. Open **Advanced settings -> Secrets** and paste in the contents of
   [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example), filling
   in your own `OPENROUTER_API_KEY`:

   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-..."
   OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
   OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
   ```

   Streamlit exposes these as both `st.secrets` and `os.environ`, so
   `pydantic-settings` (`src/cga/config.py`) picks them up with no code changes.
4. Click **Deploy**. First build installs dependencies from `requirements.txt`
   — expect several minutes, mostly `torch` + `sentence-transformers`.
5. Once it's up, ask a question in the chat to confirm retrieval + the
   OpenRouter call both work end to end.

## Why this works without re-ingesting PDFs

- `data/processed/chroma/{ada,nice}/` (the built vector indices, ~47MB) is
  committed to the repo — `src/cga/graph/retrieval.py` loads them directly.
- `data/raw/` (the source PDFs) stays gitignored and is **not** needed at
  runtime — it's only read by `uv run python -m cga.ingestion.index` when
  rebuilding the index locally.
- Embeddings are computed locally via `sentence-transformers`
  (`all-MiniLM-L6-v2`), so no embedding API key is needed on the host either.

## `requirements.txt` notes

`requirements.txt` is generated from `uv.lock`, not written by hand. Streamlit
Cloud has no GPU, so the default `uv export` output (which resolves the CUDA
build of `torch` on Linux, pulling in ~15 `nvidia-*`/`cuda-*` packages) is
trimmed down to the CPU-only wheel:

```bash
uv export --no-dev --no-hashes -o requirements.txt
# then: remove every nvidia-*/cuda-*/triton==* line and its "# via" comment
# block, and add this line near the top:
#   --extra-index-url https://download.pytorch.org/whl/cpu
```

Regenerate and re-trim it any time `pyproject.toml`'s dependencies change.

## Updating the demo after re-ingesting

If you re-run `uv run python -m cga.ingestion.index` (new source PDFs, chunking
changes, different embedding model, etc.), the rebuilt files under
`data/processed/chroma/` need to be committed and pushed like any other
change — Streamlit Cloud redeploys automatically on push to `main`.

## Known limitations on the free tier

- **Cold starts.** Apps that go idle spin down; the next visitor waits ~30s
  for a restart.
- **1 GB RAM / shared CPU.** `torch` + `sentence-transformers` +
  `chromadb` fit, but there's little headroom — avoid loading a second model
  or a much larger embedding model without checking memory use.
- **In-memory conversation checkpointer.** `src/cga/memory/checkpointer.py`
  uses `MemorySaver`, so any restart (deploy, reboot, idle spin-down) drops
  all conversation history — expected for a demo, not a concern to fix here.
- **Free-tier OpenRouter model flakiness** carries over from local dev — see
  [README.md's Known Limitations](README.md#known-limitations).
