"""Local persistent vector store for Mindly Lab 3.

Stores chunk embeddings and rich metadata (document_name, source, page, chunk_id)
with cosine similarity search using NumPy.
Includes delta fingerprinting to prevent redundant re-indexing of PDFs.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import requests

from config.settings import settings


STOPWORDS = {
    "a", "an", "the", "in", "on", "of", "for", "to", "is", "are", "was", "were",
    "it", "and", "or", "that", "this", "with", "as", "by", "at", "from", "be",
    "can", "could", "should", "would", "what", "which", "who", "whom", "how",
}


def _compute_fallback_embedding(text: str, dim: int = 1024) -> List[float]:
    """Compute deterministic semantic/n-gram embedding vector as a zero-dependency fallback."""
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    words = [w for w in clean.split() if w not in STOPWORDS]
    vec = np.zeros(dim, dtype=np.float32)
    if not words:
        return vec.tolist()

    for idx, word in enumerate(words):
        # FNV-1a word hash
        h = 2166136261
        for char in word:
            h = ((h ^ ord(char)) * 16777619) & 0xFFFFFFFF
        slot = h % dim
        vec[slot] += 2.0 / (1.0 + math.log(1.0 + idx))

        # Character trigrams for subword robustness
        for i in range(len(word) - 2):
            th = 2166136261
            for c in word[i:i+3]:
                th = ((th ^ ord(c)) * 16777619) & 0xFFFFFFFF
            vec[th % dim] += 0.5

    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return vec.tolist()


def get_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """Get embedding vector using local Ollama if an embedding model is available, otherwise semantic vector."""
    _model = model or settings.embedding_model
    # Only query Ollama if model name explicitly indicates an embedding model (e.g. nomic-embed-text, bge, minilm)
    if any(k in _model.lower() for k in ["embed", "minilm", "bge"]):
        url = f"{settings.ollama_base_url}/api/embeddings"
        try:
            payload = {"model": _model, "prompt": text}
            res = requests.post(url, json=payload, timeout=2.0)
            if res.status_code == 200:
                data = res.json()
                emb = data.get("embedding")
                if emb and isinstance(emb, list) and len(emb) > 0:
                    return emb
        except Exception:
            pass

    return _compute_fallback_embedding(text)


class VectorStore:
    """Persistent local vector database."""

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self.store_dir = store_dir or settings.vector_db_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.store_dir / "vector_index.json"
        self.fingerprint_file = self.store_dir / "document_fingerprint.txt"
        self.chunks: List[Dict[str, Any]] = []
        self._load_index()

    def _load_index(self) -> None:
        """Load stored chunks from JSON index."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            except Exception:
                self.chunks = []
        else:
            self.chunks = []

    def save_index(self) -> None:
        """Save chunk embeddings and metadata to disk."""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def get_fingerprint(self) -> Optional[str]:
        """Read saved document fingerprint hash."""
        if self.fingerprint_file.exists():
            return self.fingerprint_file.read_text(encoding="utf-8").strip()
        return None

    def save_fingerprint(self, hash_val: str) -> None:
        """Save new document fingerprint hash."""
        self.fingerprint_file.write_text(hash_val.strip(), encoding="utf-8")

    def count(self) -> int:
        """Return total number of indexed chunks."""
        return len(self.chunks)

    def clear(self) -> None:
        """Purge vector store index."""
        self.chunks = []
        if self.index_file.exists():
            self.index_file.unlink()
        if self.fingerprint_file.exists():
            self.fingerprint_file.unlink()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Add chunks to index with computed embeddings."""
        for item in chunks:
            text = item.get("text", "")
            if not item.get("embedding"):
                item["embedding"] = get_embedding(text)
            self.chunks.append(item)
        self.save_index()

    def search(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Search relevant chunks via cosine similarity."""
        if not self.chunks or not query.strip():
            return []

        query_emb = np.array(get_embedding(query), dtype=np.float32)
        q_norm = np.linalg.norm(query_emb)
        if q_norm < 1e-6:
            return []
        query_emb = query_emb / q_norm

        scores: List[Tuple[float, Dict[str, Any]]] = []
        for chk in self.chunks:
            chk_emb = np.array(chk.get("embedding", []), dtype=np.float32)
            c_norm = np.linalg.norm(chk_emb)
            if c_norm < 1e-6:
                continue
            chk_emb = chk_emb / c_norm

            sim = float(np.dot(query_emb, chk_emb))
            if sim >= threshold:
                res_dict = {
                    "source": chk.get("source", "Gale Encyclopedia of Medicine"),
                    "page": chk.get("page", 1),
                    "chunk_id": chk.get("chunk_id", ""),
                    "document_name": chk.get("document_name", "gale_encyclopedia.pdf"),
                    "text": chk.get("text", ""),
                    "score": round(sim, 4),
                }
                scores.append((sim, res_dict))

        # Sort descending by similarity score
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:top_k]]


# Singleton vector store instance
vector_store = VectorStore()
