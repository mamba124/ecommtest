from fastapi import APIRouter, Request
from pydantic import BaseModel, Field


class ChunkingConfigBody(BaseModel):
    """
    Live chunking parameters. Accepted by POST /config/chunking and stored in
    app.state.chunking_config. Changes take effect on the next /ingest call.
    No persistence between restarts — in-memory only.
    """
    chunk_size: int = Field(default=500, gt=0, description="Max tokens per chunk")
    chunk_overlap: int = Field(default=50, ge=0, description="Overlap between consecutive chunks")
    separators: list[str] = Field(
        default=["\n\n", "\n", "."],
        description="Ordered list of split separators",
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
