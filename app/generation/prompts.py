"""key prompts necessary for all tasks presented in the assignment"""

SYSTEM_PROMPT = """You are a precise technical assistant. Answer questions using ONLY the provided context chunks.

Rules:
- If the answer is not in the context, say: "I don't have enough information in the provided documents to answer this question."
- Never fabricate facts or infer beyond what the context states.
- Always cite the chunk IDs you used in your answer using the format [chunk_id].
- Be concise. Prefer bullet points for lists of items.
"""

USER_PROMPT_TEMPLATE = """Context chunks:
{context}

Question: {question}

Answer (cite chunk IDs inline):"""

FAITHFULNESS_PROMPT = """Context:
{context}

Answer: {answer}

Does the answer contain ONLY claims supported by the context above?
Reply with a single number between 0.0 (completely unsupported) and 1.0 (fully supported). Reply with only the number."""

RELEVANCY_PROMPT = """Given this answer: {answer}

Generate {n} different questions that this answer addresses.
Return as a JSON array of strings only."""

SUB_QUESTION_PROMPT = """Given the topic '{topic}', generate {n} specific sub-questions \
that together would give a comprehensive understanding.
Return as a JSON array of strings only."""

SYNTHESIS_PROMPT = """Synthesize these findings into a structured report on '{topic}'.
Use markdown with sections. Be factual.

{sub_qa}"""

SAMPLE_GENERATION_PROMPT = """Given the following document excerpt, generate {n} evaluation questions.
Include a mix of: factual (clear unambiguous answers), multi_hop (require synthesizing info from \
multiple chunks), unanswerable (no answer in the corpus, to test hallucination resistance), \
and paraphrased (same question asked differently from the source text).

Document:
{text}

Return a JSON array of objects with keys: id, type, question, ground_truth_answer, ground_truth_context.
For unanswerable questions set ground_truth_answer to "UNANSWERABLE" and ground_truth_context to null."""


def format_context(chunks) -> str:
    """Format retrieved chunks into the context block injected into the prompt."""
    lines: list[str] = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")
        page_str = f", page: {page}" if page is not None else ""
        lines.append(f"[chunk_id: {chunk.chunk_id}] (source: {source}{page_str})")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines)
