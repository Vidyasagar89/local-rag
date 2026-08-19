#!/usr/bin/env python3
"""Simple benchmark/eval for local RAG.

Usage:
  python tests/benchmarks/eval_rag.py --simulate
  python tests/benchmarks/eval_rag.py --real

--simulate: fast, no Ollama calls; uses simple substring-matching retriever for
measuring relative latency and recall across configurations.

--real: calls into rag_core.expand_queries and retrieve; requires Ollama + models
and can be slow. Use when ready to run a real evaluation.
"""
import argparse
import json
import time
from collections import defaultdict

SIM_TOP_K = 8


def simple_retrieve_sim(query, docs, top_k=SIM_TOP_K):
    """Naive substring-based retriever for simulation mode.
    Returns list of (doc_text, meta) where meta={'source': filename}.
    """
    qtokens = set(query.lower().split())
    scores = []
    for filename, text in docs.items():
        words = set(text.lower().split())
        overlap = len(qtokens & words)
        if overlap > 0:
            scores.append((overlap, filename, text))
    scores.sort(reverse=True)
    return [(text, {"source": filename}) for _, filename, text in scores[:top_k]]


def load_sample_docs():
    files = ["README.md", "rag_core.py"]
    docs = {}
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                docs[f] = fh.read()
        except FileNotFoundError:
            docs[f] = ""
    return docs


def run_simulation(sample_queries_path, top_k=SIM_TOP_K):
    docs = load_sample_docs()
    with open(sample_queries_path) as fh:
        queries = json.load(fh)

    timings = defaultdict(list)
    recall_counts = []

    for q in queries:
        question = q["query"]
        expected = set(q.get("expected_sources", []))

        t0 = time.perf_counter()
        # simulate expand_queries cost: inexpensive synthetic alternate
        alt = question + " (alternate phrasing)"
        t1 = time.perf_counter()
        timings["expand_queries"].append(t1 - t0)

        t2 = time.perf_counter()
        hits = simple_retrieve_sim(question + " " + alt, docs, top_k=top_k)
        t3 = time.perf_counter()
        timings["retrieve"].append(t3 - t2)

        sources = {meta["source"] for _, meta in hits}
        hit = bool(expected & sources)
        recall_counts.append(1 if hit else 0)

    n = len(queries)
    print("Simulation results (n=%d):" % n)
    print("  expand_queries avg: %.4f ms" % (1000 * (sum(timings["expand_queries"]) / n)))
    print("  retrieve avg: %.4f ms" % (1000 * (sum(timings["retrieve"]) / n)))
    print("  recall@%d: %d/%d = %.2f%%" % (top_k, sum(recall_counts), n, 100 * sum(recall_counts) / n))


def run_real(sample_queries_path, top_k=SIM_TOP_K):
    try:
        from rag_core import expand_queries, retrieve
    except Exception as e:
        print("Failed to import rag_core: ", e)
        return

    with open(sample_queries_path) as fh:
        queries = json.load(fh)

    timings = defaultdict(list)
    recall_counts = []

    for q in queries:
        question = q["query"]
        expected = set(q.get("expected_sources", []))

        t0 = time.perf_counter()
        exps = expand_queries(question)
        t1 = time.perf_counter()
        timings["expand_queries"].append(t1 - t0)

        t2 = time.perf_counter()
        hits = retrieve(question, top_k=top_k)
        t3 = time.perf_counter()
        timings["retrieve"].append(t3 - t2)

        sources = {meta["source"] for _, meta in hits}
        hit = bool(expected & sources)
        recall_counts.append(1 if hit else 0)

        print(f"Query: {question}\n  expanded: {exps}\n  hits: {list(sources)}\n")

    n = len(queries)
    print("Real run results (n=%d):" % n)
    print("  expand_queries avg: %.4f s" % (sum(timings["expand_queries"]) / n))
    print("  retrieve avg: %.4f s" % (sum(timings["retrieve"]) / n))
    print("  recall@%d: %d/%d = %.2f%%" % (top_k, sum(recall_counts), n, 100 * sum(recall_counts) / n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--simulate", action="store_true", help="Run fast simulated benchmark")
    p.add_argument("--real", action="store_true", help="Run against rag_core (may call Ollama)")
    p.add_argument("--sample", default="tests/benchmarks/sample_queries.json")
    args = p.parse_args()

    if not (args.simulate or args.real):
        print("Please pass --simulate or --real")
        return

    if args.simulate:
        run_simulation(args.sample)
    if args.real:
        run_real(args.sample)


if __name__ == "__main__":
    main()
