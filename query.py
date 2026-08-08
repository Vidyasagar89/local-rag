"""
Ask a question against your indexed documents.

Usage:
    python query.py "What does the payment module do?"

Or run with no args for an interactive chat loop.
"""
import sys
from rag_core import answer_question


def ask(question: str):
    answer, hits = answer_question(question)
    print("\n--- Answer ---")
    print(answer)
    print("\n--- Sources ---")
    seen = set()
    for _, meta in hits:
        if meta["source"] not in seen:
            print(f"- {meta['source']}")
            seen.add(meta["source"])


def main():
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
        return

    print("Local RAG chat. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        ask(question)
        print()


if __name__ == "__main__":
    main()
