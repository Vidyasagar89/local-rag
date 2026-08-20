# Local RAG (fully offline)

A minimal, offline RAG system built for a small machine (tested against an
M1 Mac mini, 8GB RAM). Uses Ollama for embeddings + generation and Chroma
as a local vector store. No data ever leaves your machine.

## 1. Pull the models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

If 3B still feels heavy on 8GB, swap `LLM_MODEL` in `rag_core.py` for
`phi3.5` or `gemma2:2b` — both are lighter.

## 2. Install Python deps

```bash
cd local-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Index your documents

```bash
python ingest.py /path/to/your/documents
```

Supports PDF, DOCX, and most text/code file types (.py, .js, .md, .json, etc).
Re-running on the same folder updates existing chunks rather than duplicating.

## 4. Ask questions

CLI, one-off:

```bash
python query.py "What does the auth module do?"
```

CLI, interactive chat:

```bash
python query.py
```

Or the web UI:

```bash
python app.py
# open http://localhost:5050
```

Upload files directly in the browser and chat with them.

### Open it from your phone

Same approach as [local-llm](../local-llm): `app.py` binds to `0.0.0.0`
and prints a LAN URL on startup, so any device on the same WiFi network
can reach it:

```
Local RAG is starting...
Open on your phone: http://<your-pc-lan-ip>:5050
Other LAN addresses: ...
Local browser: http://localhost:5050
```

Open that first URL on your phone's browser (same WiFi, not guest WiFi —
that usually isolates devices from each other). If it won't connect, check
your PC's firewall allows inbound connections on port 5050.

## How retrieval works now

Three changes in `rag_core.py` fix the "asked one way, got 'not in the
document', reworded it, got the right answer" problem:

- **Small collections skip retrieval entirely.** If your whole index has
  `SMALL_COLLECTION_THRESHOLD` (25) chunks or fewer — e.g. you just uploaded
  one offer letter — every chunk gets sent to the model as context instead of
  a top-k search trying to guess which few are relevant. Nothing can get
  "missed" if nothing gets filtered out.
- **Query expansion.** For larger collections, before searching, the LLM is
  asked to generate a single alternate phrasing of your question, and both the
  original and alternate are searched. This is the automatic version of
  rewording the question yourself when it fixed a missed result.
- **Hybrid vector + keyword search.** Embedding search alone can miss exact
  terms (names, numbers, jargon) if the wording doesn't line up. A BM25
  keyword pass runs alongside the vector search and the two are merged, so an
  exact term match can surface a chunk even if its embedding similarity was
  weak.

Chunking is also now paragraph-aware (`chunk_text` in `rag_core.py`) instead
of a blind character sliding window, so a labeled field like "Joining Bonus:
₹X" is much less likely to get split across two chunks.

## Web search (optional, off by default)

For questions your local documents can't answer, you can search the web
instead — on a per-question, manual basis only, never automatically. This
uses the [Tavily](https://tavily.com) API (free tier available), which
returns clean text snippets built for feeding to an LLM.

1. Get a free API key at https://tavily.com
2. Copy `.env.example` to `.env` and fill in your key:
   ```bash
   cp .env.example .env
   # then edit .env and paste your key in place of tvly-your-key-here
   ```
3. In the web UI, check the **Web** box before asking a question.
   On the CLI: `python query.py --web "your question"`, or prefix a
   question with `web:` in interactive mode.

`.env` is already in `.gitignore`, so your key won't get committed if this
folder is ever put under version control. The key loads via `python-dotenv`
in `web_search.py`, independent of your shell's startup files — it works
the same way whether you launch the app from a terminal, an IDE, or a script.

This is intentionally a separate code path (`web_search.py`) from local RAG
(`rag_core.py`) — local-document answers and web answers never blend into
one context, so the answer bubble always tells you which source it came
from. Nothing goes over the network unless you explicitly check the box or
pass `--web`.

## Performance

The web UI (`app.py`) used to block until the entire answer was generated,
and re-scanned + re-indexed the whole document collection on every single
question. Both are fixed now:

- **Streaming answers.** `/ask` streams newline-delimited JSON and the
  browser renders tokens as they arrive (same idea as local-llm's
  token-by-token chat), instead of waiting for the full response. This is
  the single biggest improvement in *perceived* speed — generation itself
  takes the same time, but you're not staring at a blank bubble for it.
- **Cached corpus + BM25 index.** `retrieve()` used to call
  `collection.get()` (every id/document/metadata in the index) and rebuild
  a BM25 index from scratch on every question. Both are now cached in
  `rag_core.py` and only rebuilt when the chunk count actually changes —
  cheap on every other question.
- **Models kept warm.** Every question round-trips through two different
  Ollama models (`nomic-embed-text` for embedding, `LLM_MODEL` for
  generation). By default Ollama can evict one to load the other between
  those two calls, which shows up as multi-second stalls. All Ollama calls
  in this project now pass `keep_alive="30m"` (see `KEEP_ALIVE` in
  `rag_core.py`) so a model isn't dropped right after use. For this to let
  *both* models sit in RAM at once rather than still taking turns, also
  set `OLLAMA_MAX_LOADED_MODELS=2` in the environment `ollama serve` runs
  in, e.g.:
  ```bash
  OLLAMA_MAX_LOADED_MODELS=2 ollama serve
  ```
  On an 8GB machine, `nomic-embed-text` (~270MB) plus a 3B model in Q4
  (~2GB) fit in memory together fine; if you're on something even tighter,
  leave this unset and accept the occasional reload instead.

## Notes for your hardware

- Chunk size (1200 chars) and top_k (8) in `rag_core.py` are tuned to keep
  prompts reasonably small while giving retrieval more to work with —
  increase them once you confirm speed is acceptable.
- The Chroma DB persists to `./chroma_db` — delete that folder to reset
  the index from scratch.
- Query expansion adds one extra LLM call per question. If it feels slow on
  your machine, set `QUERY_EXPANSION = False` in `rag_core.py` — the
  small-collection full-context path and hybrid search still apply.
