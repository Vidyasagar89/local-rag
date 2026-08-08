"""
List indexed files, or delete one by filename.

Usage:
    python manage.py list
    python manage.py delete somefile.pdf
"""
import sys
from rag_core import list_sources, delete_document


def main():
    if len(sys.argv) < 2:
        print("Usage:\n  python manage.py list\n  python manage.py delete <filename>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "list":
        sources = list_sources()
        if not sources:
            print("No documents indexed yet.")
        for s in sources:
            print(s)

    elif cmd == "delete":
        if len(sys.argv) != 3:
            print("Usage: python manage.py delete <filename>")
            sys.exit(1)
        filename = sys.argv[2]
        count = delete_document(filename)
        if count:
            print(f"Deleted {count} chunk(s) for '{filename}'.")
        else:
            print(f"No chunks found for '{filename}'. Run 'python manage.py list' to see exact names.")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
