"""key prompts necessary for all tasks presented in the assignment"""

SYSTEM_PROMPT = """You are a precise technical assistant. You must answer EXCLUSIVELY from the numbered context chunks provided below.

Rules (strictly enforced):
- Every sentence in your answer MUST include an inline citation in the format [Chunk N].
- If the explicit answer is not present in the provided chunks, respond with exactly: INSUFFICIENT_CONTEXT
- Never infer, extrapolate, or add information beyond what the chunks state verbatim.
- Never fabricate citations or reference chunks that do not contain the information cited.
- Be concise. Prefer bullet points for lists of items.
"""

USER_PROMPT_TEMPLATE = """{context}

Question: {question}

Answer (cite every sentence with [Chunk N]):"""

VERIFICATION_PROMPT = """You are a fact-checking assistant. Given the context chunks and an answer, determine whether every claim in the answer is explicitly supported by the context.

{context}

Answer to verify:
{answer}

Is every sentence in the answer fully supported by the context chunks above?
Reply with YES or NO on the first line, then one sentence explaining why."""

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

SAMPLE_GENERATION_PROMPT = """You are a dataset builder. Read the document excerpt below and generate exactly {n} evaluation questions of the type specified at the end of this prompt.

Document:
{text}

Output ONLY a JSON array — no explanation, no prose, no markdown fences. Each element must have exactly these keys:
  "id"                  – leave as empty string ""
  "type"                – the question type
  "question"            – the question text
  "ground_truth_answer" – short, accurate answer drawn solely from the document; "UNANSWERABLE" for unanswerable type
  "ground_truth_context"– verbatim sentence(s) from the document that support the answer; null for unanswerable type

Example of the required output format (do not copy these questions, use the document above):
[
  {{"id": "", "type": "factual", "question": "What is the API endpoint for sending messages?", "ground_truth_answer": "The endpoint is POST /v1/messages.", "ground_truth_context": "Send a structured list of input messages to the POST /v1/messages endpoint."}},
  {{"id": "", "type": "unanswerable", "question": "What is Claude's favourite colour?", "ground_truth_answer": "UNANSWERABLE", "ground_truth_context": null}}
]"""


def format_context(chunks) -> str:
    """Format retrieved chunks into numbered [Chunk N] blocks for the prompt."""
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")
        page_str = f", page {page}" if page is not None else ""
        lines.append(f"[Chunk {i}] (source: {source}{page_str})")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines)
