"""
Shared config + helpers for the local RAG system.
Everything here runs offline: Ollama for embeddings/generation, Chroma for storage.
"""
import os
import chromadb
import ollama

# ---- Config (tune these for your 8GB M1 Mac mini) ----
EMBED_MODEL = "nomic-embed-text"   # ~270MB, light on RAM
LLM_MODEL = "llama3.2:3b"          # swap for phi3.5 / gemma2:2b if you want it lighter
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "local_docs"
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap so context isn't cut mid-thought
TOP_K = 4               # how many chunks to retrieve per question

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
    """Simple sliding-window chunker. Good enough for most RAG use cases."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
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
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
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


def retrieve(question: str, top_k=TOP_K):
    collection = get_collection()
    q_embedding = embed(question)
    results = collection.query(query_embeddings=[q_embedding], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return list(zip(docs, metas))


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
