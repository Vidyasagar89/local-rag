"""
Web search fallback for the RAG app. Off by default — only used when you
explicitly ask for it (checkbox in the web UI, or --web / "web:" prefix on
the CLI). Deliberately kept separate from rag_core.py: local-document
answers and web answers should never silently blend into one context, so
you always know which source produced an answer.

Uses the Tavily API, which is built for feeding LLMs — it returns clean
text snippets instead of raw HTML you'd have to scrape and parse yourself.

Setup:
    Get a free API key at https://tavily.com
    Create a .env file in the project root containing:
        TAVILY_API_KEY=tvly-...

Swapping to a different provider (Google Custom Search, Bing, etc.) only
requires rewriting search_web() below — everything else (answer_from_web,
the app.py/query.py wiring) stays the same as long as it still returns a
list of {title, url, content} dicts.
"""
import os
import requests
import ollama
from dotenv import load_dotenv

from rag_core import LLM_MODEL, KEEP_ALIVE

# Load variables from a .env file in the project root, if present. This is
# independent of your shell's startup files (.zshrc, .bashrc, etc.), so it
# works the same way whether you launch app.py from a terminal, an IDE run
# button, or a script — no dependency on how/whether that shell sourced
# your profile.
load_dotenv()

TAVILY_URL = "https://api.tavily.com/search"
WEB_TOP_K = 5


class WebSearchUnavailable(Exception):
    """Raised when web search can't run (e.g. missing API key)."""
    pass


def search_web(query: str, max_results: int = WEB_TOP_K):
    """Return a list of {title, url, content} dicts from a live web search."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise WebSearchUnavailable(
            "TAVILY_API_KEY is not set. Add it to a .env file in the project "
            "root (TAVILY_API_KEY=your_key) — get a free key at https://tavily.com"
        )

    response = requests.post(
        TAVILY_URL,
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in data.get("results", [])
    ]


def _build_messages(question: str, results):
    if not results:
        context = "(no web results found)"
    else:
        context = "\n\n---\n\n".join(
            f"[Source: {r['title']} ({r['url']})]\n{r['content']}" for r in results
        )

    system_prompt = (
        "You are a helpful assistant answering questions using ONLY the "
        "provided web search results below. If the answer isn't in the "
        "results, say so clearly instead of guessing. Cite source URLs "
        "when relevant."
    )
    user_prompt = f"Web search results:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def answer_from_web(question: str, max_results: int = WEB_TOP_K):
    """
    Search the web and generate an answer grounded in those results.
    Mirrors rag_core.answer_question()'s shape (answer, sources) so
    query.py can treat the two paths interchangeably at the call site.
    """
    results = search_web(question, max_results=max_results)
    messages = _build_messages(question, results)

    response = ollama.chat(model=LLM_MODEL, messages=messages,
                            keep_alive=KEEP_ALIVE)

    sources = [r["url"] for r in results]
    return response["message"]["content"], sources


def answer_from_web_stream(question: str, max_results: int = WEB_TOP_K):
    """Streaming counterpart to answer_from_web, mirroring
    rag_core.answer_question_stream()'s (kind, payload) shape:

        ("sources", [url, ...])   -- once, right after the search
        ("token", text_delta)     -- repeatedly, as generation streams in
    """
    results = search_web(question, max_results=max_results)
    yield "sources", [r["url"] for r in results]

    messages = _build_messages(question, results)
    stream = ollama.chat(model=LLM_MODEL, messages=messages,
                          keep_alive=KEEP_ALIVE, stream=True)
    for chunk in stream:
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            yield "token", delta
