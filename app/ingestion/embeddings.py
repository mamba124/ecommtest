from abc import ABC, abstractmethod

import httpx

from app.core.config import EmbeddingsConfig


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder(BaseEmbedder):
    _BATCH_SIZE = 32

    def __init__(self, config: EmbeddingsConfig) -> None:
        self._model = config.model
        self._base_url = config.base_url

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            for text in texts[i : i + self._BATCH_SIZE]:
                embeddings.append(self.embed_query(text))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]


class SentenceTransformerEmbedder(BaseEmbedder):
    _BATCH_SIZE = 32

    def __init__(self, config: EmbeddingsConfig) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(config.model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            vecs = self._model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(vecs.tolist())
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()


class EmbedderFactory:
    @staticmethod
    def create(config) -> BaseEmbedder:
        provider = config.embeddings.provider
        if provider == "ollama":
            return OllamaEmbedder(config.embeddings)
        elif provider == "sentence_transformers":
            return SentenceTransformerEmbedder(config.embeddings)
        else:
            raise ValueError(f"Unknown embedding provider: {provider!r}")
