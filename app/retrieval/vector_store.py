import logging
from dataclasses import dataclass

import chromadb

from app.core.config import Config
from app.ingestion.chunking import Chunk

logger = logging.getLogger("vector_store")


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    score: float   # cosine similarity, 0–1


class VectorStore:
    def __init__(self, config: Config) -> None:
        vs = config.vector_store
        self._collection_name = vs.collection_name
        self._client = chromadb.HttpClient(host=vs.host, port=vs.port)
        self._collection = self._client.get_or_create_collection(
            name=vs.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore connected: {vs.host}:{vs.port}, "
            f"collection={vs.collection_name}"
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[RetrievedChunk] = []
        for cid, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance → similarity: 1 - distance
            chunks.append(RetrievedChunk(
                chunk_id=cid,
                text=text,
                metadata=meta,
                score=max(0.0, 1.0 - dist),
            ))
        return chunks

    def get_all_texts(self) -> tuple[list[str], list[str]]:
        """Returns (ids, texts) for BM25 index rebuilding."""
        result = self._collection.get(include=["documents"])
        return result["ids"], result["documents"]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
