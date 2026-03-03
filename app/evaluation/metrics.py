"""
Evaluation Metrics used to evaluate the output comparing to ground truth.

Metric          What It Measures
Context Precision   Are the retrieved chunks actually relevant to the question?
Context Recall      Does retrieval find all the chunks needed to answer?
Faithfulness        Is the generated answer grounded in the retrieved context (no hallucination)?
Answer Relevancy    Does the generated answer actually address the question?
Answer Correctness  Does the generated answer match the ground truth?
"""
import json
import math

from app.generation.llm import BaseLLM
from app.generation.prompts import FAITHFULNESS_PROMPT, RELEVANCY_PROMPT
from app.ingestion.embeddings import BaseEmbedder

REFUSAL_PHRASES = [
    "i don't have enough information",
    "not in the context",
    "cannot find",
    "no information",
    "unanswerable",
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def context_precision(
    question: str,
    retrieved_chunks: list[str],
    ground_truth_context: str,
) -> float:
    """Fraction of retrieved chunks that are relevant to the ground truth."""
    if not retrieved_chunks or not ground_truth_context:
        return 0.0
    gt_lower = ground_truth_context.lower()
    relevant = sum(
        1
        for chunk in retrieved_chunks
        if any(w in gt_lower for w in chunk.lower().split() if len(w) > 4)
    )
    return relevant / len(retrieved_chunks)


def context_recall(
    retrieved_chunks: list[str],
    ground_truth_context: str,
) -> float:
    """Fraction of ground truth context covered by retrieved chunks."""
    if not ground_truth_context:
        return 1.0
    gt_words = {w.lower() for w in ground_truth_context.split() if len(w) > 4}
    if not gt_words:
        return 1.0
    retrieved_text = " ".join(retrieved_chunks).lower()
    return sum(1 for w in gt_words if w in retrieved_text) / len(gt_words)


def faithfulness(
    answer: str,
    retrieved_chunks: list[str],
    llm: BaseLLM,
) -> float:
    """
    Prompt the LLM to score: does the answer contain only claims
    supported by the retrieved chunks? Returns 0–1.
    """
    context = "\n---\n".join(retrieved_chunks)
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    raw = llm.generate(prompt).strip()
    try:
        return max(0.0, min(1.0, float(raw.split()[0])))
    except (ValueError, IndexError):
        return 0.0


def answer_relevancy(
    question: str,
    answer: str,
    llm: BaseLLM,
    embedder: BaseEmbedder,
    n_paraphrases: int = 3,
) -> float:
    """
    Prompt the LLM to generate N paraphrases of the question from the answer,
    measure cosine similarity of their embeddings to the original question embedding.
    """
    prompt = RELEVANCY_PROMPT.format(answer=answer, n=n_paraphrases)
    raw = llm.generate(prompt)
    try:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        paraphrases: list[str] = json.loads(raw[start:end])[:n_paraphrases]
    except (ValueError, json.JSONDecodeError):
        return 0.0

    if not paraphrases:
        return 0.0

    original_emb = embedder.embed_query(question)
    para_embs = embedder.embed(paraphrases)
    sims = [_cosine(original_emb, e) for e in para_embs]
    return sum(sims) / len(sims)


def answer_correctness(
    answer: str,
    ground_truth: str,
    embedder: BaseEmbedder,
) -> float:
    """
    For unanswerable: 1.0 if answer contains refusal phrase, 0.0 otherwise.
    For others: cosine similarity between answer embedding and ground_truth embedding.
    """
    if ground_truth.upper() == "UNANSWERABLE":
        return 1.0 if any(p in answer.lower() for p in REFUSAL_PHRASES) else 0.0
    a_emb = embedder.embed_query(answer)
    gt_emb = embedder.embed_query(ground_truth)
    return _cosine(a_emb, gt_emb)
