import logging

from sentence_transformers import CrossEncoder

from app.retrieval.vector_store import RetrievedChunk

logger = logging.getLogger("reranker")


class Reranker:
    def __init__(self, model_name: str) -> None:
        self._model = CrossEncoder(model_name)
        logger.info(f"Reranker loaded: {model_name}")

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_k]]
