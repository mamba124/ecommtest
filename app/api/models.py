from typing import Optional

from pydantic import BaseModel, Field


# ── Ingest ──────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    path: str = Field(..., description="Absolute or relative path to folder of documents")


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    collection: str


# ── Query ────────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    chunk_id: str
    source: str
    page: Optional[int] = None
    score: float


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None    # overrides config if set
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    from_cache: bool = False


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str                    # "ok" | "degraded" | "error"
    vector_db: str
    embedding_model: str
    llm: str
    collection_count: int


# ── Research Agent ───────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str
    max_sub_questions: Optional[int] = None


class GapItem(BaseModel):
    sub_question: str
    reason: str


class ResearchResponse(BaseModel):
    topic: str
    sub_questions: list[str]
    report: str
    gaps: list[GapItem]
    all_citations: list[Citation]
