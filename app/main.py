"""
POST /ingest     — triggers ingestion of a document folder
POST /query      — accepts a natural language question, returns answer with citations
GET  /health     — returns service status
POST /research   — agentic multi-step research over the knowledge base
POST /config/chunking — update live chunking parameters (no restart required)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_config
from app.core.logging import setup_logging
from app.core.cache import QueryCache
from app.api.config import ChunkingConfigBody
from app.ingestion.embeddings import EmbedderFactory
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.generation.llm import LLMFactory
from app.api.ingest import router as ingest_router
from app.api.query import router as query_router
from app.api.health import router as health_router
from app.api.research import router as research_router
from app.api.config import router as config_router

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    app.state.config = config

    # Seed live chunking config from config.yaml defaults
    app.state.chunking_config = ChunkingConfigBody(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        separators=config.chunking.separators,
    )

    app.state.vector_store = VectorStore(config)
    app.state.embedder = EmbedderFactory.create(config)
    app.state.retriever = Retriever(app.state.vector_store, app.state.embedder, config)
    app.state.llm = LLMFactory.create(config)

    # Async Redis cache
    app.state.cache = QueryCache(ttl=config.cache.ttl)

    yield

    await app.state.cache.close()


app = FastAPI(title="RAG Service", lifespan=lifespan)

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(health_router)
app.include_router(research_router)
app.include_router(config_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error", "error": str(exc)},
    )


app.mount("/ui", StaticFiles(directory="ui"), name="ui")
