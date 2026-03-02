import functools
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    provider: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int


class EmbeddingsConfig(BaseModel):
    provider: str
    model: str
    base_url: str
    dimension: int


class ChunkingConfig(BaseModel):
    strategy: str
    chunk_size: int
    chunk_overlap: int
    separators: list[str]


class RetrievalConfig(BaseModel):
    top_k: int
    rerank_top_k: int
    use_hybrid: bool
    hybrid_alpha: float
    use_reranker: bool
    reranker_model: str


class VectorStoreConfig(BaseModel):
    provider: str
    host: str
    port: int
    collection_name: str


class CacheConfig(BaseModel):
    enabled: bool
    ttl: int = 3600            # Redis key TTL in seconds


class AgentConfig(BaseModel):
    max_sub_questions: int
    synthesis_model: Optional[str] = None


class Config(BaseModel):
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    vector_store: VectorStoreConfig
    cache: CacheConfig
    agent: AgentConfig


@functools.lru_cache(maxsize=1)
def get_config() -> Config:
    config_path = Path(os.getenv("CONFIG_PATH", "config.yaml"))
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return Config(**data)
