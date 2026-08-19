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
LLM_MODEL = "llama3.2:3b"
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "local_docs"

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
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
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
    return len(ids)


def delete_document(filename: str) -> int:
    """Delete all chunks belonging to a given source filename. Returns count removed."""
    collection = get_collection()
    existing = collection.get(where={"source": filename})
    count = len(existing.get("ids", []))
    if count:
        collection.delete(where={"source": filename})
    return count


def list_sources():
    """Return the distinct set of filenames currently indexed."""
    collection = get_collection()
    all_docs = collection.get()
    return sorted({meta["source"] for meta in all_docs.get("metadatas", [])})


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def _bm25_scores(question: str, ids, docs):
    """Keyword-overlap ranking, as a fallback for terms the embedding misses."""
    if not docs:
        return {}
    tokenized_docs = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)
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
                               {"role": "user", "content": prompt}])
        alt_lines = [l.strip("-• ").strip() for l in response["message"]
                     ["content"].splitlines() if l.strip()]
        return [question] + alt_lines[:1]
    except Exception:
        return [question]


def retrieve(question: str, top_k=TOP_K):
    collection = get_collection()
    all_docs = collection.get()
    all_ids = all_docs.get("ids", [])
    all_documents = all_docs.get("documents", [])
    all_metas = all_docs.get("metadatas", [])

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


def answer_question(question: str, top_k=TOP_K):
    """Full RAG pipeline: retrieve relevant chunks, then generate a grounded answer."""
    hits = retrieve(question, top_k=top_k)
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

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"], hits
