"""backend for the handle /query"""
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.models import Citation, QueryRequest, QueryResponse
from app.generation.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, VERIFICATION_PROMPT, format_context

router = APIRouter()
logger = logging.getLogger("query")


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request):
    config = request.app.state.config
    cache = request.app.state.cache
    embedder = request.app.state.embedder
    retriever = request.app.state.retriever
    llm = request.app.state.llm

    # ── Redis cache check ────────────────────────────────────────────────────
    if config.cache.enabled:
        cached = await cache.get(req.question)
        if cached is not None:
            # cached is a plain dict — reconstruct the model
            return QueryResponse(**{**cached, "from_cache": True})

    # ── Embed + retrieve ─────────────────────────────────────────────────────
    try:
        query_embedding = embedder.embed_query(req.question)
    except Exception as exc:
        logger.error(f"Embedding failed: {exc}")
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    chunks = retriever.search(req.question, query_embedding)

    citations = [
        Citation(
            chunk_id=c.chunk_id,
            source=c.metadata.get("source", "unknown"),
            page=c.metadata.get("page"),
            score=c.score,
            text=c.text,
        )
        for c in chunks
    ]

    context = format_context(chunks)
    prompt = USER_PROMPT_TEMPLATE.format(context=context, question=req.question)

    # ── Streaming path (not cached) ───────────────────────────────────────────
    if req.stream:
        async def event_stream():
            try:
                for token in llm.stream(prompt, SYSTEM_PROMPT):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                yield f"data: {json.dumps({'done': True, 'citations': [c.model_dump() for c in citations]})}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ── Normal path ───────────────────────────────────────────────────────────
    try:
        answer = llm.generate(prompt, SYSTEM_PROMPT)
    except Exception as exc:
        logger.error(f"LLM generation failed: {exc}")
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    # ── Verification pass ─────────────────────────────────────────────────────
    if answer.strip().upper() != "INSUFFICIENT_CONTEXT":
        try:
            verification_prompt = VERIFICATION_PROMPT.format(context=context, answer=answer)
            verdict = llm.generate(verification_prompt).strip()
            if not verdict.upper().startswith("YES"):
                logger.info(f"Verification rejected answer: {verdict[:120]}")
                answer = "INSUFFICIENT_CONTEXT"
        except Exception as exc:
            logger.warning(f"Verification pass failed: {exc}")

    response = QueryResponse(answer=answer, citations=citations)

    # ── Store in Redis ────────────────────────────────────────────────────────
    if config.cache.enabled:
        await cache.set(req.question, response)

    return response
