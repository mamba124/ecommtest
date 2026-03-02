from fastapi import Request

from app.core.config import Config
from app.core.cache import QueryCache
from app.ingestion.embeddings import BaseEmbedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.generation.llm import BaseLLM


def get_config(request: Request) -> Config:
    return request.app.state.config


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_embedder(request: Request) -> BaseEmbedder:
    return request.app.state.embedder


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_llm(request: Request) -> BaseLLM:
    return request.app.state.llm


def get_cache(request: Request) -> QueryCache:
    return request.app.state.cache
