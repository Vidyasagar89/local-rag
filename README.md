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

## How retrieval works now

Three changes in `rag_core.py` fix the "asked one way, got 'not in the
document', reworded it, got the right answer" problem:

- **Small collections skip retrieval entirely.** If your whole index has
  `SMALL_COLLECTION_THRESHOLD` (25) chunks or fewer — e.g. you just uploaded
  one offer letter — every chunk gets sent to the model as context instead of
  a top-k search trying to guess which few are relevant. Nothing can get
  "missed" if nothing gets filtered out.
- **Query expansion.** For larger collections, before searching, the LLM is
  asked to generate 1-2 alternate phrasings of your question, and all of them
  are searched. This is the automatic version of what you were doing by hand
  when rewording the question fixed it.
- **Hybrid vector + keyword search.** Embedding search alone can miss exact
  terms (names, numbers, jargon) if the wording doesn't line up. A BM25
  keyword pass runs alongside the vector search and the two are merged, so an
  exact term match can surface a chunk even if its embedding similarity was
  weak.

Chunking is also now paragraph-aware (`chunk_text` in `rag_core.py`) instead
of a blind character sliding window, so a labeled field like "Joining Bonus:
₹X" is much less likely to get split across two chunks.

## Notes for your hardware

- Chunk size (1200 chars) and top_k (8) in `rag_core.py` are tuned to keep
  prompts reasonably small while giving retrieval more to work with —
  increase them once you confirm speed is acceptable.
- Ollama loads/unloads models automatically, so embedding + generation
  models don't both have to sit in memory at once, which helps a lot at 8GB.
- The Chroma DB persists to `./chroma_db` — delete that folder to reset
  the index from scratch.
- Query expansion adds one extra LLM call per question. If it feels slow on
  your machine, set `QUERY_EXPANSION = False` in `rag_core.py` — the
  small-collection full-context path and hybrid search still apply.
