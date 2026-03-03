import asyncio
import json
import logging

from app.api.models import Citation, GapItem, ResearchResponse
from app.generation.llm import BaseLLM
from app.generation.prompts import (
    SUB_QUESTION_PROMPT,
    SYNTHESIS_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    format_context,
)
from app.ingestion.embeddings import BaseEmbedder
from app.retrieval.retriever import Retriever

logger = logging.getLogger("agent")

_REFUSAL_PHRASES = [
    "i don't have enough information",
    "not in the context",
    "cannot find",
    "no information",
]

# Module-level: genuinely shared across all concurrent /research requests.
_llm_sem = asyncio.Semaphore(3)


async def run_research(
    topic: str,
    llm: BaseLLM,
    retriever: Retriever,
    embedder: BaseEmbedder,
    max_sub_questions: int,
) -> ResearchResponse:

    # Step 1 — generate sub-questions
    async with _llm_sem:
        raw = await asyncio.to_thread(
            llm.generate, SUB_QUESTION_PROMPT.format(topic=topic, n=max_sub_questions)
        )
    try:
        start, end = raw.index("["), raw.rindex("]") + 1
        sub_questions: list[str] = json.loads(raw[start:end])[:max_sub_questions]
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"Sub-question JSON parse failed: {exc}; splitting by line")
        sub_questions = [
            ln.strip().lstrip("-•0123456789. ")
            for ln in raw.splitlines() if ln.strip()
        ][:max_sub_questions]
    logger.info(f"Agent: {len(sub_questions)} sub-questions for {topic!r}")

    # Step 2 — parallel retrieval (no semaphore: no LLM involved)
    async def retrieve(q: str):
        emb = await asyncio.to_thread(embedder.embed_query, q)
        return await asyncio.to_thread(retriever.search, q, emb)

    chunk_lists = await asyncio.gather(*[retrieve(q) for q in sub_questions])

    # Step 3 — parallel answer generation, capped by semaphore
    async def answer(q: str, chunks) -> str:
        prompt = USER_PROMPT_TEMPLATE.format(context=format_context(chunks), question=q)
        async with _llm_sem:
            return await asyncio.to_thread(llm.generate, prompt, SYSTEM_PROMPT)

    answers: list[str] = list(await asyncio.gather(*[
        answer(q, c) for q, c in zip(sub_questions, chunk_lists)
    ]))

    # Step 4 — gap detection
    gaps = [
        GapItem(sub_question=q, reason=a)
        for q, a in zip(sub_questions, answers)
        if any(p in a.lower() for p in _REFUSAL_PHRASES)
    ]

    # Step 5 — synthesis
    sub_qa = "\n\n".join(f"Sub-question: {q}\nAnswer: {a}" for q, a in zip(sub_questions, answers))
    async with _llm_sem:
        report = await asyncio.to_thread(llm.generate, SYNTHESIS_PROMPT.format(topic=topic, sub_qa=sub_qa))

    # Deduplicated citations
    seen: set[str] = set()
    all_citations: list[Citation] = []
    for chunk in (c for chunks in chunk_lists for c in chunks):
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            all_citations.append(Citation(
                chunk_id=chunk.chunk_id,
                source=chunk.metadata.get("source", "unknown"),
                page=chunk.metadata.get("page"),
                score=chunk.score,
            ))

    return ResearchResponse(
        topic=topic,
        sub_questions=sub_questions,
        report=report,
        gaps=gaps,
        all_citations=all_citations,
    )
