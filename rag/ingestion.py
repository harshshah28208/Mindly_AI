"""Document ingestion pipeline for Mindly Lab 3 RAG.

Extracts text from the Gale Encyclopedia of Medicine PDF,
splits into semantically overlapping chunks with rich metadata,
computes document SHA256 fingerprints, and indexes into the VectorStore.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pypdf import PdfReader

from config.settings import settings
from rag.vectorstore import vector_store


def calculate_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for delta change detection."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[str]:
    """Split text into overlapping character chunks while preserving paragraph/sentence boundaries."""
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    chunks = []
    start = 0
    while start < len(clean):
        end = start + chunk_size
        if end >= len(clean):
            chunks.append(clean[start:].strip())
            break

        # Look for sentence boundary near end
        boundary = clean.rfind(". ", start, end)
        if boundary != -1 and boundary > start + (chunk_size // 2):
            chunk = clean[start:boundary + 1].strip()
            start = boundary + 2
        else:
            # Fallback to word space
            space = clean.rfind(" ", start, end)
            if space != -1 and space > start + (chunk_size // 2):
                chunk = clean[start:space].strip()
                start = space + 1
            else:
                chunk = clean[start:end].strip()
                start = end - chunk_overlap

        if chunk:
            chunks.append(chunk)

    return chunks


def extract_chunks_from_pdf(
    pdf_path: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[Dict[str, Any]]:
    """Extract and chunk text from PDF with source and page metadata."""
    if not pdf_path or not Path(pdf_path).exists():
        return []

    reader = PdfReader(str(pdf_path))
    chunks: List[Dict[str, Any]] = []
    doc_name = pdf_path.name

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = page.extract_text() or ""
        page_chunks = split_text_into_chunks(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk_idx, chk_text in enumerate(page_chunks):
            chunk_id = f"{pdf_path.stem}_p{page_num}_c{chunk_idx + 1}"
            chunks.append({
                "chunk_id": chunk_id,
                "document_name": doc_name,
                "source": "Gale Encyclopedia of Medicine",
                "page": page_num,
                "text": chk_text,
            })

    return chunks


def ingest_knowledge(
    pdf_path: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Run RAG ingestion pipeline with delta hashing."""
    target_pdf = pdf_path or settings.gale_pdf_path
    if not target_pdf.exists():
        return {
            "success": False,
            "status": "missing_file",
            "message": f"Knowledge PDF not found at {target_pdf}. Place 'gale_encyclopedia.pdf' in data/knowledge/.",
            "chunks_indexed": 0,
        }

    current_hash = calculate_file_hash(target_pdf)
    existing_hash = vector_store.get_fingerprint()

    # Delta detection: skip if unchanged unless forced
    if existing_hash == current_hash and not force and vector_store.count() > 0:
        return {
            "success": True,
            "status": "unchanged",
            "message": "Vector index is already up to date with existing document fingerprint.",
            "chunks_indexed": vector_store.count(),
        }

    try:
        raw_chunks = extract_chunks_from_pdf(target_pdf)
        if not raw_chunks:
            return {
                "success": False,
                "status": "empty_pdf",
                "message": "Could not extract readable text from PDF.",
                "chunks_indexed": 0,
            }

        vector_store.clear()
        vector_store.add_chunks(raw_chunks)
        vector_store.save_fingerprint(current_hash)

        return {
            "success": True,
            "status": "indexed",
            "message": f"Successfully ingested {len(raw_chunks)} chunks from {target_pdf.name}.",
            "chunks_indexed": len(raw_chunks),
        }
    except Exception as exc:
        return {
            "success": False,
            "status": "error",
            "message": f"Ingestion failed: {str(exc)}",
            "chunks_indexed": 0,
        }
