"""
Shared config + helpers for the local RAG system.
Everything here runs offline: Ollama for embeddings/generation, Chroma for storage.
"""
import os
import re
import chromadb
import ollama
from rank_bm25 import BM25Okapi

# ---- Config (tune these for your 8GB M1 Mac mini) ----
EMBED_MODEL = "nomic-embed-text"   # ~270MB, light on RAM
# swap for phi3.5 / gemma2:2b if you want it lighter
LLM_MODEL = "granite4.1:3b"
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "local_docs"

# How long Ollama keeps each model resident in RAM after a request. We
# alternate between EMBED_MODEL and LLM_MODEL on every question, so without
# this Ollama's default (5m, and it can be even more aggressive under memory
# pressure) can evict one to make room for the other between the embedding
# call and the generation call -- that reload is most of the "why is this
# slow" latency. 30m keeps both warm across a normal chat session. For this
# to actually let both sit in RAM at once (not just take turns), also set
# OLLAMA_MAX_LOADED_MODELS=2 (or higher) in the environment `ollama serve`
# runs in -- see README.
KEEP_ALIVE = "30m"

# characters per chunk (was 800 — bigger keeps labels + values together)
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250    # overlap so context isn't cut mid-thought
TOP_K = 8              # how many chunks to retrieve per question (was 4)

# If the WHOLE index has this many chunks or fewer, skip retrieval entirely and
# hand the model every chunk. This is the fix for "single offer letter" style
# use: with only a few chunks total, there's no reason to risk the retriever
# missing the right one — just give the model everything.
SMALL_COLLECTION_THRESHOLD = 25

# Before retrieving, ask the LLM for a single alternate phrasing of the
# question and search with both the original and alternate. This is the fix
# for "rephrasing gave me the right answer" — instead of relying on you to
# rephrase, the system does it automatically and merges the results.
QUERY_EXPANSION = True

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go",
    ".rs", ".c", ".cpp", ".h", ".json", ".yaml", ".yml", ".css", ".html",
    ".pdf", ".docx",
}


def get_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection():
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def extract_text(filepath: str) -> str:
    """Extract raw text from pdf, docx, or plain text/code files."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext == ".docx":
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        with open(filepath, "r", errors="ignore") as f:
            return f.read()


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Paragraph-aware chunker. Packs whole paragraphs together up to chunk_size
    instead of blindly cutting every N characters — this keeps a labeled field
    (e.g. "Joining Bonus: X") in the same chunk as its value, which a raw
    sliding window can accidentally split apart. Falls back to a sliding
    window only for a single paragraph that's longer than chunk_size on its
    own (e.g. a dense table dumped as one block by PDF extraction).
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    raw_chunks = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            raw_chunks.append(current)
            current = ""

        if len(para) <= chunk_size:
            current = para
        else:
            # Single paragraph too big on its own — sliding window over it.
            start = 0
            while start < len(para):
                end = start + chunk_size
                piece = para[start:end].strip()
                if piece:
                    raw_chunks.append(piece)
                start += chunk_size - overlap

    if current:
        raw_chunks.append(current)

    # Stitch a small tail of overlap from the previous chunk onto each chunk
    # so a fact split right at a paragraph boundary still has context nearby.
    chunks = []
    for i, c in enumerate(raw_chunks):
        if i == 0 or overlap <= 0:
            chunks.append(c)
        else:
            tail = raw_chunks[i - 1][-overlap:]
            chunks.append(f"{tail}\n\n{c}")
    return chunks


def embed(text: str):
    response = ollama.embeddings(
        model=EMBED_MODEL, prompt=text, keep_alive=KEEP_ALIVE)
    return response["embedding"]


def add_document(filepath: str):
    """Extract, chunk, embed, and store one file. Returns number of chunks added."""
    text = extract_text(filepath)
    if not text.strip():
        return 0
    chunks = chunk_text(text)
    collection = get_collection()
    filename = os.path.basename(filepath)

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        ids.append(f"{filename}::{i}")
        embeddings.append(embed(chunk))
        documents.append(chunk)
        metadatas.append({"source": filename, "chunk": i})

    if ids:
        collection.upsert(ids=ids, embeddings=embeddings,
                          documents=documents, metadatas=metadatas)
        _invalidate_corpus_cache()
    return len(ids)


def delete_document(filename: str) -> int:
    """Delete all chunks belonging to a given source filename. Returns count removed."""
    collection = get_collection()
    existing = collection.get(where={"source": filename})
    count = len(existing.get("ids", []))
    if count:
        collection.delete(where={"source": filename})
        _invalidate_corpus_cache()
    return count


# ---- Corpus + BM25 caching ----
# retrieve() used to call collection.get() (a full read of every id,
# document, and metadata in the index) and rebuild a BM25 index over the
# whole corpus from scratch on *every question*. For anything but a tiny
# collection that's the dominant cost before the LLM even starts generating.
# We cache both here and only refresh when the collection has actually
# changed. collection.count() is cheap (no document payload), so checking it
# every call is fine even though it can't be skipped -- it's what lets us
# notice edits made by another process (ingest.py, manage.py) too.
_corpus_cache = {"count": None, "ids": [],
                  "documents": [], "metadatas": [], "bm25": None}


def _invalidate_corpus_cache():
    _corpus_cache["count"] = None


def _get_corpus():
    """Return (ids, documents, metadatas) for the whole collection, cached
    until the chunk count changes."""
    collection = get_collection()
    count = collection.count()
    if _corpus_cache["count"] != count:
        all_docs = collection.get()
        _corpus_cache["ids"] = all_docs.get("ids", [])
        _corpus_cache["documents"] = all_docs.get("documents", [])
        _corpus_cache["metadatas"] = all_docs.get("metadatas", [])
        _corpus_cache["bm25"] = None  # stale, rebuild lazily below
        _corpus_cache["count"] = count
    return _corpus_cache["ids"], _corpus_cache["documents"], _corpus_cache["metadatas"]


def list_sources():
    """Return the distinct set of filenames currently indexed."""
    _, _, all_metas = _get_corpus()
    return sorted({meta["source"] for meta in all_metas})


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def _get_bm25(ids, docs):
    """Build (or reuse) the BM25 index for the current corpus. Tokenizing
    every document is the expensive part, so this only happens once per
    corpus version instead of once per question."""
    if _corpus_cache["bm25"] is None and docs:
        tokenized_docs = [_tokenize(d) for d in docs]
        _corpus_cache["bm25"] = BM25Okapi(tokenized_docs)
    return _corpus_cache["bm25"]


def _bm25_scores(question: str, ids, docs):
    """Keyword-overlap ranking, as a fallback for terms the embedding missed."""
    bm25 = _get_bm25(ids, docs)
    if bm25 is None:
        return {}
    scores = bm25.get_scores(_tokenize(question))
    return dict(zip(ids, scores))


def expand_queries(question: str):
    """Ask the LLM for a single alternate phrasing to widen recall automatically."""
    if not QUERY_EXPANSION:
        return [question]
    prompt = (
        "Rewrite the question below as a single alternate phrasing that a document "
        "might use for the same underlying fact (different wording, synonyms, "
        "more formal or more literal phrasing). Return ONLY the alternate, "
        "on a single line, with no numbering and no extra commentary.\n\n"
        f"Question: {question}"
    )
    try:
        response = ollama.chat(model=LLM_MODEL, messages=[
                               {"role": "user", "content": prompt}], keep_alive=KEEP_ALIVE)
        alt_lines = [l.strip("-• ").strip() for l in response["message"]
                     ["content"].splitlines() if l.strip()]
        return [question] + alt_lines[:1]
    except Exception:
        return [question]


def retrieve(question: str, top_k=TOP_K):
    collection = get_collection()
    all_ids, all_documents, all_metas = _get_corpus()

    # Small index: don't risk retrieval missing anything, just hand it all over.
    if len(all_ids) <= SMALL_COLLECTION_THRESHOLD:
        return list(zip(all_documents, all_metas))

    # Larger index: hybrid vector + keyword search, across expanded queries.
    combined = {}  # id -> (doc, meta, score)

    for q in expand_queries(question):
        q_embedding = embed(q)
        results = collection.query(
            query_embeddings=[q_embedding], n_results=top_k)
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for i, d, m, dist in zip(ids, docs, metas, distances):
            score = 1 - dist  # cosine distance -> similarity
            if i not in combined or score > combined[i][2]:
                combined[i] = (d, m, score)

    # Keyword pass catches exact terms (names, numbers, jargon) the embedding missed.
    bm25 = _bm25_scores(question, all_ids, all_documents)
    if bm25:
        max_bm25 = max(bm25.values()) or 1.0
        id_to_doc_meta = {i: (d, m) for i, d, m in zip(
            all_ids, all_documents, all_metas)}
        top_bm25_ids = sorted(bm25, key=bm25.get, reverse=True)[:top_k]
        for i in top_bm25_ids:
            # normalize onto ~0-1 to sit alongside similarity
            norm_score = bm25[i] / max_bm25
            if norm_score <= 0:
                continue
            if i not in combined or norm_score > combined[i][2]:
                d, m = id_to_doc_meta[i]
                combined[i] = (d, m, norm_score)

    ranked = sorted(combined.values(), key=lambda x: x[2], reverse=True)
    return [(d, m) for d, m, _ in ranked[:top_k]]


def _build_messages(question: str, hits):
    """Shared prompt construction for both the streaming and one-shot paths."""
    if not hits:
        context = "(no documents indexed yet)"
    else:
        context = "\n\n---\n\n".join(
            f"[Source: {meta['source']}]\n{doc}" for doc, meta in hits
        )

    system_prompt = (
        "You are a helpful assistant answering questions using ONLY the provided "
        "context from the user's local documents. If the answer isn't in the "
        "context, say so clearly instead of guessing. Cite the source filename "
        "when relevant."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def answer_question(question: str, top_k=TOP_K):
    """Full RAG pipeline: retrieve relevant chunks, then generate a grounded
    answer in one shot. Used by the CLI tools (query.py); the web app uses
    answer_question_stream instead so it can show tokens as they arrive."""
    hits = retrieve(question, top_k=top_k)
    messages = _build_messages(question, hits)
    response = ollama.chat(model=LLM_MODEL, messages=messages,
                            keep_alive=KEEP_ALIVE)
    return response["message"]["content"], hits


def answer_question_stream(question: str, top_k=TOP_K):
    """Same pipeline as answer_question, but retrieves first (fast, no LLM),
    yields the hits immediately, then yields answer text incrementally as
    Ollama generates it instead of blocking until the full answer is done.

    Yields:
        ("hits", hits)              -- once, right after retrieval
        ("token", text_delta)       -- repeatedly, as generation streams in
    """
    hits = retrieve(question, top_k=top_k)
    yield "hits", hits

    messages = _build_messages(question, hits)
    stream = ollama.chat(model=LLM_MODEL, messages=messages,
                          keep_alive=KEEP_ALIVE, stream=True)
    for chunk in stream:
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            yield "token", delta
