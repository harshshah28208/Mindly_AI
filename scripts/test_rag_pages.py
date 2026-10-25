import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.vectorstore import vector_store

tests = [
    ("i am having pain in my knee", 6),
    ("I am having acidity what to do", 5),
    ("kidney stones and flank pain", 64),
    ("appendicitis and lower right abdominal pain", 31),
    ("epilepsy and seizures", 45),
    ("gout attack in big toe", 50),
    ("pneumonia and lung consolidation", 85),
    ("sciatica and herniated disc pain", 93),
]

for query, expected_page in tests:
    res = vector_store.search(query, top_k=1)
    page = res[0]["page"]
    status = "EXACT MATCH" if page == expected_page else f"GOT Page {page} (expected {expected_page})"
    print(f"[{status}] '{query}' -> Page {page}: {res[0]['text'][:55]}...")
