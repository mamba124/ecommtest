import logging

from fastapi import APIRouter, Request

from app.api.models import ResearchRequest, ResearchResponse
from app.core.agent import run_research

router = APIRouter()
logger = logging.getLogger("research")


@router.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest, request: Request) -> ResearchResponse:
    config = request.app.state.config
    return await run_research(
        topic=req.topic,
        llm=request.app.state.llm,
        retriever=request.app.state.retriever,
        embedder=request.app.state.embedder,
        max_sub_questions=req.max_sub_questions or config.agent.max_sub_questions,
    )
