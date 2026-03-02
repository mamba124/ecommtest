import logging

from fastapi import APIRouter, Request

from app.api.models import ResearchRequest, ResearchResponse
from app.core.agent import ResearchAgent

router = APIRouter()
logger = logging.getLogger("research")


@router.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest, request: Request) -> ResearchResponse:
    agent = ResearchAgent(
        llm=request.app.state.llm,
        retriever=request.app.state.retriever,
        embedder=request.app.state.embedder,
        config=request.app.state.config,
    )
    return await agent.run(req.topic, req.max_sub_questions)
