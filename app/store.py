"""Vector store backed by ChromaDB."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import chromadb
from chromadb.api.types import EmbeddingFunction

from app.chunking import split_text


@dataclass(frozen=True)
class Hit:
    text: str
    document_id: str
    chunk_index: int
    distance: float
    metadata: dict


class Retriever(Protocol):
    def query(self, question: str, top_k: int) -> list[Hit]: ...


class VectorStore:
    """Thin wrapper around a Chroma collection with document-level ids.

    Chunk ids are `<document_id>:<chunk_index>` so re-ingesting the same
    document replaces its chunks instead of duplicating them.
    """

    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection_name: str = "documents",
        embedding_function: EmbeddingFunction | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        kwargs = {"name": collection_name, "metadata": {"hnsw:space": "cosine"}}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self.collection = client.get_or_create_collection(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @classmethod
    def persistent(cls, path: str, **kwargs) -> VectorStore:
        return cls(chromadb.PersistentClient(path=path), **kwargs)

    @classmethod
    def in_memory(cls, **kwargs) -> VectorStore:
        return cls(chromadb.EphemeralClient(), **kwargs)

    # ── Write ────────────────────────────────────────────────────────────

    def add_document(self, document_id: str, text: str, metadata: dict | None = None) -> int:
        """Chunk, embed and upsert a document. Returns the number of chunks."""
        self.delete_document(document_id)
        chunks = split_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return 0
        base_meta = dict(metadata or {})
        base_meta["document_id"] = document_id
        self.collection.upsert(
            ids=[f"{document_id}:{c.index}" for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{**base_meta, "chunk_index": c.index, "start": c.start, "end": c.end} for c in chunks],
        )
        return len(chunks)

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    # ── Read ─────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 4) -> list[Hit]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(
            query_texts=[question],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True):
            hits.append(
                Hit(
                    text=text,
                    document_id=str(meta.get("document_id", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    distance=float(dist),
                    metadata={k: v for k, v in meta.items() if k not in {"document_id", "chunk_index", "start", "end"}},
                )
            )
        return hits

    def count_documents(self) -> int:
        got = self.collection.get(include=["metadatas"])
        return len({m.get("document_id") for m in got["metadatas"]})


def document_id_for(name: str) -> str:
    """Stable id derived from a file name so re-uploads replace, not duplicate."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
