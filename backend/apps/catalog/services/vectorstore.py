"""Thin accessor for the persistent ChromaDB collection."""
from __future__ import annotations

from pathlib import Path


def get_collection(path: str | Path, name: str):
    """Open (or create) the persistent collection configured for cosine search."""
    import chromadb

    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
