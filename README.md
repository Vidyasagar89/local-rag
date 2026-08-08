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

## Notes for your hardware

- Chunk size (800 chars) and top_k (4) in `rag_core.py` are tuned to keep
  prompts small — increase them once you confirm speed is acceptable.
- Ollama loads/unloads models automatically, so embedding + generation
  models don't both have to sit in memory at once, which helps a lot at 8GB.
- The Chroma DB persists to `./chroma_db` — delete that folder to reset
  the index from scratch.
