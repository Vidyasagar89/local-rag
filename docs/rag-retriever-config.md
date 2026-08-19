Retriever configuration recommendations (conversation-aware)

Goal: pass a rewritten standalone query plus short conversation summary and recent turns as metadata to bias retrieval scoring.

Generic JSON input (send with each retrieval request):
{
  "query": "<rewritten_query>",
  "conversation_summary": "<1-2 sentence summary>",
  "recent_turns": [
    {"role":"user","text":"..."},
    {"role":"assistant","text":"..."}
  ],
  "top_k": 10,
  "use_metadata_bias": true
}

Examples for popular stacks

1) Pinecone + hybrid re-ranking (pseudo):
- Store metadata per vector: {"source":"Offer Letter Mphasis.pdf","page":12}
- When querying, send the conversation_summary as filterless metadata hint in the reranker / cross-encoder stage.
- Retrieval flow: ANN search (top_k=50) -> cross-encoder re-rank using [rewritten_query + conversation_summary] -> return top 10.

2) FAISS (local) + cross-encoder:
- ANN top_k 100 -> cross-encoder rerank with conversation_summary -> return top_k 10.

3) Weaviate / Milvus with metadata-aware scoring:
- Pass conversation_summary in the vector search "query" or use Weaviate's hybrid (BM25 + vector) with metadata_expressions that boost matches containing keywords from the summary.

Notes:
- If no reranker available, still pass the conversation_summary as part of the query string (e.g., "<conversation_summary>. Question: <rewritten_query>") to bias lexical/semantic matches.
- Always include source metadata (filename, page/paragraph) when indexing so snippets can be cited.
- Increase top_k at the ANN stage to 50–100 for re-ranking if using a cross-encoder; final return top_k 10.

Operational tips:
- Rate limit re-ranker calls; keep ANN small then re-rank.
- Cache conversation_summary for short TTL (e.g., 60s) to avoid recompute.
- Log rewritten_query + retrieved ids for debugging follow-up failures.
