import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
from rag.retriever import retrieve_relevant_chunks

queries = [
    "how to handle skin acnes",
    "What are the symptoms of migraine?",
    "Causes of acid reflux"
]

for q in queries:
    chunks = retrieve_relevant_chunks(q, top_k=1)
    if chunks:
        p = chunks[0].get("page", 1)
        text = chunks[0].get("text", "")
        print(f"\n=== Query: {q} (Page {p}) ===")
        print(text[:300].replace("\n", " "))
