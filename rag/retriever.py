"""Retriever component for Mindly RAG."""

from typing import Any, Dict, List
from rag.vectorstore import vector_store


def retrieve_relevant_chunks(
    query: str,
    top_k: int = 3,
    threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    """Retrieve top-k relevant knowledge chunks from the vector store."""
    return vector_store.search(query=query, top_k=top_k, threshold=threshold)
