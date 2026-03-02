import asyncio
import json
import logging

from app.api.models import Citation, GapItem, ResearchResponse
from app.core.config import Config
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
_SEMAPHORE = 3


class ResearchAgent:
    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        embedder: BaseEmbedder,
        config: Config,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._embedder = embedder
        self._max_sub_questions = config.agent.max_sub_questions

    async def run(
        self, topic: str, max_sub_questions: int | None = None
    ) -> ResearchResponse:
        n = max_sub_questions or self._max_sub_questions
        sem = asyncio.Semaphore(_SEMAPHORE)

        # Step 1 — Sub-question generation
        sub_questions = await self._generate_sub_questions(topic, n)
        logger.info(f"Agent: {len(sub_questions)} sub-questions for topic {topic!r}")

        # Step 2 — Parallel retrieval
        async def retrieve(q: str):
            async with sem:
                emb = await asyncio.to_thread(self._embedder.embed_query, q)
                return await asyncio.to_thread(self._retriever.search, q, emb)

        chunk_lists = await asyncio.gather(*[retrieve(q) for q in sub_questions])

        # Step 3 — Parallel answer generation
        async def answer(q: str, chunks):
            ctx = format_context(chunks)
            prompt = USER_PROMPT_TEMPLATE.format(context=ctx, question=q)
            async with sem:
                return await asyncio.to_thread(self._llm.generate, prompt, SYSTEM_PROMPT)

        answers = await asyncio.gather(
            *[answer(q, c) for q, c in zip(sub_questions, chunk_lists)]
        )

        # Step 4 — Gap detection
        gaps: list[GapItem] = [
            GapItem(sub_question=q, reason=a)
            for q, a in zip(sub_questions, answers)
            if any(phrase in a.lower() for phrase in _REFUSAL_PHRASES)
        ]

        # Step 5 — Synthesis
        sub_qa = "\n\n".join(
            f"Sub-question: {q}\nAnswer: {a}"
            for q, a in zip(sub_questions, answers)
        )
        synthesis = SYNTHESIS_PROMPT.format(topic=topic, sub_qa=sub_qa)
        report = await asyncio.to_thread(self._llm.generate, synthesis)

        # Collect deduplicated citations
        seen: set[str] = set()
        all_citations: list[Citation] = []
        for chunks in chunk_lists:
            for chunk in chunks:
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
            sub_questions=list(sub_questions),
            report=report,
            gaps=gaps,
            all_citations=all_citations,
        )

    async def _generate_sub_questions(self, topic: str, n: int) -> list[str]:
        prompt = SUB_QUESTION_PROMPT.format(topic=topic, n=n)
        raw = await asyncio.to_thread(self._llm.generate, prompt)
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            return json.loads(raw[start:end])[:n]
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning(f"Failed to parse sub-questions JSON: {exc}; falling back to line split")
            lines = [
                ln.strip().lstrip("-•0123456789. ")
                for ln in raw.splitlines()
                if ln.strip()
            ]
            return lines[:n]
