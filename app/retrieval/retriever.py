import logging

from rank_bm25 import BM25Okapi

from app.core.config import Config
from app.ingestion.embeddings import BaseEmbedder
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger("retriever")


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: BaseEmbedder,
        config: Config,
    ) -> None:
        self._vs = vector_store
        self._embedder = embedder
        self._cfg = config.retrieval
        self._reranker = (
            Reranker(self._cfg.reranker_model) if self._cfg.use_reranker else None
        )
        self._bm25_index: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._bm25_texts: list[str] = []
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        ids, texts = self._vs.get_all_texts()
        if not texts:
            self._bm25_index = None
            self._bm25_ids = []
            self._bm25_texts = []
            return
        tokenized = [t.lower().split() for t in texts]
        self._bm25_index = BM25Okapi(tokenized)
        self._bm25_ids = list(ids)
        self._bm25_texts = list(texts)
        logger.info(f"BM25 index built with {len(texts)} documents")

    def rebuild_index(self) -> None:
        self._build_bm25_index()

    def search(self, query: str, query_embedding: list[float]) -> list[RetrievedChunk]:
        """
        1. Vector search → top_k results (scores normalized 0–1)
        2. BM25 search → top_k results (scores normalized 0–1)
        3. Fuse: score = alpha * vector_score + (1-alpha) * bm25_score
        4. Take top_k fused results
        5. If use_reranker: reorder by cross-encoder, keep rerank_top_k
        6. Return final ranked list
        """
        top_k = self._cfg.top_k
        alpha = self._cfg.hybrid_alpha

        vector_results = self._vs.query(query_embedding, top_k)
        vector_map: dict[str, RetrievedChunk] = {r.chunk_id: r for r in vector_results}

        if not self._cfg.use_hybrid or self._bm25_index is None:
            candidates = vector_results
        else:
            tokens = query.lower().split()
            bm25_raw = self._bm25_index.get_scores(tokens)
            max_bm25 = max(bm25_raw) if max(bm25_raw) > 0 else 1.0
            bm25_norm = {
                self._bm25_ids[i]: bm25_raw[i] / max_bm25
                for i in range(len(self._bm25_ids))
            }

            # Top-k BM25 ids
            top_bm25_ids = {
                self._bm25_ids[i]
                for i in sorted(
                    range(len(bm25_raw)), key=lambda x: bm25_raw[x], reverse=True
                )[:top_k]
            }
            all_ids = set(vector_map.keys()) | top_bm25_ids

            fused: list[tuple[str, float]] = []
            for cid in all_ids:
                v = vector_map[cid].score if cid in vector_map else 0.0
                b = bm25_norm.get(cid, 0.0)
                fused.append((cid, alpha * v + (1 - alpha) * b))

            fused.sort(key=lambda x: x[1], reverse=True)
            bm25_text_map = dict(zip(self._bm25_ids, self._bm25_texts))
            candidates = []
            for cid, score in fused[:top_k]:
                if cid in vector_map:
                    src = vector_map[cid]
                    candidates.append(RetrievedChunk(
                        chunk_id=cid,
                        text=src.text,
                        metadata=src.metadata,
                        score=score,
                    ))
                else:
                    candidates.append(RetrievedChunk(
                        chunk_id=cid,
                        text=bm25_text_map.get(cid, ""),
                        metadata={},
                        score=score,
                    ))

        if self._reranker:
            final = self._reranker.rerank(query, candidates, self._cfg.rerank_top_k)
        else:
            final = candidates[: self._cfg.rerank_top_k]

        logger.info(
            f'Query: "{query}" → {len(final)} chunks retrieved '
            f"(hybrid={self._cfg.use_hybrid}, reranked={self._reranker is not None})"
        )
        return final
