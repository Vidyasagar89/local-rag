"""
Index a folder of documents into the local vector store.

Usage:
    python ingest.py /path/to/your/documents
"""
import sys
import os
from rag_core import add_document, SUPPORTED_EXTENSIONS


def main():
    if len(sys.argv) != 2:
        print("Usage: python ingest.py /path/to/documents")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Not a folder: {folder}")
        sys.exit(1)

    total_files = 0
    total_chunks = 0
    for root, _, files in os.walk(folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            path = os.path.join(root, fname)
            try:
                n = add_document(path)
                print(f"Indexed {fname}: {n} chunks")
                total_files += 1
                total_chunks += n
            except Exception as e:
                print(f"Skipped {fname}: {e}")

    print(f"\nDone. Indexed {total_files} files, {total_chunks} chunks total.")


if __name__ == "__main__":
    main()
