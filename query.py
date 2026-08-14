"""
Ask a question against your indexed documents.

Usage:
    python query.py "What does the payment module do?"
    python query.py --web "What's the latest Python release?"

Or run with no args for an interactive chat loop. In that loop, prefix a
question with "web:" to search the web instead of your local documents.
"""
import sys
from rag_core import answer_question
from web_search import answer_from_web, WebSearchUnavailable


def ask(question: str, use_web: bool = False):
    if use_web:
        try:
            answer, sources = answer_from_web(question)
        except WebSearchUnavailable as e:
            print(f"\n{e}")
            return
    else:
        answer, hits = answer_question(question)
        sources = []
        seen = set()
        for _, meta in hits:
            if meta["source"] not in seen:
                sources.append(meta["source"])
                seen.add(meta["source"])

    print("\n--- Answer ---")
    print(answer)
    print("\n--- Sources ---" if not use_web else "\n--- Web sources ---")
    for s in sources:
        print(f"- {s}")


def main():
    args = sys.argv[1:]
    use_web = "--web" in args
    if use_web:
        args.remove("--web")

    if args:
        ask(" ".join(args), use_web=use_web)
        return

    print("Local RAG chat. Type 'exit' to quit. Prefix a question with 'web:' to search the web instead of local docs.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        web = False
        if question.lower().startswith("web:"):
            web = True
            question = question[4:].strip()
        ask(question, use_web=web)
        print()


if __name__ == "__main__":
    main()
