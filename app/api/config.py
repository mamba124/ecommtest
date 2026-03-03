from fastapi import APIRouter, Request
from pydantic import BaseModel, Field


class ChunkingConfigBody(BaseModel):
    """
    Live chunking parameters. Accepted by POST /config/chunking and stored in
    app.state.chunking_config. Changes take effect on the next /ingest call.
    No persistence between restarts — in-memory only.

    Defaults are tuned for LLM API documentation: sections are dense and
    self-contained (endpoint description + parameters + code example), so
    chunks must be large enough to hold a complete concept without splitting
    it mid-explanation.
    """
    chunk_size: int = Field(
        default=1200,
        gt=0,
        description=(
            "Maximum characters per chunk. 1200 fits a typical API section "
            "(endpoint + parameters + one code example) without mid-concept splits."
        ),
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description=(
            "Characters of overlap between consecutive chunks (~17% of chunk_size). "
            "Preserves sentences that straddle a chunk boundary so retrieval "
            "does not miss bridging context."
        ),
    )
    separators: list[str] = Field(
        default=["\n\n## ", "\n\n### ", "\n\n", "\n", ". ", " ", ""],
        description=(
            "Ordered split hierarchy. Markdown heading boundaries are tried first "
            "so splits happen at section edges rather than mid-paragraph."
        ),
    )


router = APIRouter(prefix="/config", tags=["config"])


@router.post("/chunking", response_model=ChunkingConfigBody, summary="Update chunking config")
async def update_chunking(body: ChunkingConfigBody, request: Request) -> ChunkingConfigBody:
    """
    Update the live chunking configuration.  Changes are reflected immediately
    on the next `/ingest` call.  Visible and testable from Swagger UI at `/docs`.
    """
    request.app.state.chunking_config = body
    return body


@router.get("/chunking", response_model=ChunkingConfigBody, summary="Get current chunking config")
async def get_chunking(request: Request) -> ChunkingConfigBody:
    return request.app.state.chunking_config
