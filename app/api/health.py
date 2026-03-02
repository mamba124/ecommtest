"""modules that check health of the service"""
import logging

from fastapi import APIRouter, Request

from app.api.models import HealthResponse

router = APIRouter()
logger = logging.getLogger("health")


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    config = request.app.state.config
    components: dict[str, str] = {}

    # Vector DB
    try:
        count = request.app.state.vector_store.count()
        components["vector_db"] = "ok"
    except Exception as exc:
        logger.warning(f"Vector DB health check failed: {exc}")
        components["vector_db"] = f"error: {exc}"
        count = 0

    # Embedder
    try:
        request.app.state.embedder.embed_query("health")
        components["embedding"] = "ok"
    except Exception as exc:
        logger.warning(f"Embedder health check failed: {exc}")
        components["embedding"] = f"error: {exc}"

    # LLM
    try:
        request.app.state.llm.generate("Reply with ok.", system="")
        components["llm"] = "ok"
    except Exception as exc:
        logger.warning(f"LLM health check failed: {exc}")
        components["llm"] = f"error: {exc}"

    overall = "ok" if all(v == "ok" for v in components.values()) else "degraded"

    return HealthResponse(
        status=overall,
        vector_db=components["vector_db"],
        embedding_model=config.embeddings.model,
        llm=f"{config.llm.provider}/{config.llm.model}",
        collection_count=count,
    )
