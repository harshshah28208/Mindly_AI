"""CLI script for ingesting knowledge documents into Mindly VectorStore.

Usage:
    python ingest.py [--force]
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from rag.ingestion import ingest_knowledge


def main():
    force = "--force" in sys.argv or "-f" in sys.argv
    print("=" * 60)
    print("MINDLY LAB 3 — KNOWLEDGE INGESTION PIPELINE")
    print("=" * 60)
    print(f"Target PDF Path : {settings.gale_pdf_path}")
    print(f"Vector DB Dir   : {settings.vector_db_dir}")
    print(f"Force Re-index  : {force}")
    print("-" * 60)

    result = ingest_knowledge(force=force)

    if result.get("success"):
        print(f"Status          : SUCCESS ({result.get('status')})")
        print(f"Chunks Indexed  : {result.get('chunks_indexed')}")
        print(f"Message         : {result.get('message')}")
    else:
        print(f"Status          : FAILED ({result.get('status')})")
        print(f"Error Message   : {result.get('message')}")

    print("=" * 60)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
