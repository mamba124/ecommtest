"""backend for handle /ingest"""
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.api.models import IngestRequest, IngestResponse
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter()
logger = logging.getLogger("ingest")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, request: Request) -> IngestResponse:
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.path}")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {req.path}")

    # Always read the live chunking config from app state so that any update
    # made via POST /config/chunking is immediately reflected here.
    base_config = request.app.state.config
    chunking_config = request.app.state.chunking_config

    pipeline = IngestionPipeline(base_config, chunking_config)
    docs, chunks, failed = pipeline.run(str(path))

    if not docs:
        raise HTTPException(status_code=400, detail="No supported files found in path")

    embedder = request.app.state.embedder
    vector_store = request.app.state.vector_store
    retriever = request.app.state.retriever

    try:
        embeddings = embedder.embed([c.text for c in chunks])
        vector_store.upsert(chunks, embeddings)
    except Exception as exc:
        logger.error(f"Vector store upsert failed: {exc}")
        raise HTTPException(status_code=503, detail="Vector store unavailable")

    retriever.rebuild_index()
    logger.info(
        f"Ingested {len(docs)} documents, {len(chunks)} chunks "
        f"into collection={base_config.vector_store.collection_name} "
        f"(chunk_size={chunking_config.chunk_size}, "
        f"overlap={chunking_config.chunk_overlap})"
    )

    return IngestResponse(
        status="ok",
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        collection=base_config.vector_store.collection_name,
    )
