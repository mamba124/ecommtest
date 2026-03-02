"""
Generate a structured report (Markdown) that includes:
Aggregate scores per metric
Per-question breakdown showing where the pipeline fails
At least 3 specific failure cases with your analysis of why they failed
At least 2 concrete suggestions for improvements you would make if given more time
"""
import logging
from pathlib import Path

logger = logging.getLogger("report")


def generate_report(results, output_path: str) -> None:
    """
    Generates a Markdown evaluation report from EvalResults.
    """
    agg = results.aggregate()

    lines: list[str] = [
        "# RAG Evaluation Report\n",
        "## Aggregate Scores\n",
        "| Metric              | Score |",
        "|---------------------|-------|",
        f"| Context Precision   | {agg.get('context_precision', 0):.2f}  |",
        f"| Context Recall      | {agg.get('context_recall', 0):.2f}  |",
        f"| Faithfulness        | {agg.get('faithfulness', 0):.2f}  |",
        f"| Answer Relevancy    | {agg.get('answer_relevancy', 0):.2f}  |",
        f"| Answer Correctness  | {agg.get('answer_correctness', 0):.2f}  |",
        "",
        "## Per-Question Breakdown\n",
        "| ID   | Type        | CP   | CR   | F    | AR   | AC   | Notes |",
        "|------|-------------|------|------|------|------|------|-------|",
    ]

    for r in results.per_question:
        lines.append(
            f"| {r.id:<4} | {r.type:<11} "
            f"| {r.context_precision:.2f} "
            f"| {r.context_recall:.2f} "
            f"| {r.faithfulness:.2f} "
            f"| {r.answer_relevancy:.2f} "
            f"| {r.answer_correctness:.2f} "
            f"| {r.notes} |"
        )

    # Failure cases: any metric < 0.5
    _METRICS = [
        "context_precision", "context_recall",
        "faithfulness", "answer_relevancy", "answer_correctness",
    ]
    failures = [
        r for r in results.per_question
        if any(getattr(r, m) < 0.5 for m in _METRICS)
    ]

    lines += ["", "## Failure Case Analysis\n"]
    if not failures:
        lines.append("No failure cases detected (all metrics ≥ 0.5 for every question).\n")
    else:
        for i, r in enumerate(failures[:5], 1):   # cap at 5 for readability
            bad = [m for m in _METRICS if getattr(r, m) < 0.5]
            lines += [
                f"### Case {i}: [{r.id}] {r.question}",
                f"**Metrics:** "
                + ", ".join(f"{m.replace('_', ' ').title()}={getattr(r, m):.2f}" for m in bad),
                f"**Retrieved chunks:** {r.retrieved_chunks}",
                f"**Generated answer:** {r.answer[:300]}{'…' if len(r.answer) > 300 else ''}",
                f"**Ground truth:** (see dataset {r.id})",
                f"**Analysis:** Low {', '.join(bad)} indicates "
                + (
                    "the retriever did not surface the relevant document for this query."
                    if "context_precision" in bad or "context_recall" in bad
                    else "the LLM introduced unsupported claims or failed to address the question."
                ),
                "",
            ]

    lines += [
        "## Improvement Suggestions\n",
        "1. **Tune `chunk_overlap`** — if context recall is consistently low, increasing overlap "
        "from 20% to 30% reduces the chance of answer-bearing sentences falling across chunk boundaries.",
        "2. **Adjust `hybrid_alpha`** — for keyword-heavy technical queries, lowering alpha from 0.7 "
        "to 0.5 gives BM25 more weight and can improve precision on exact-match terms.",
        "3. **Use a larger reranker model** — replacing `MiniLM-L-6-v2` with `MiniLM-L-12-v2` or "
        "`ms-marco-electra-base` adds ≈10% latency but typically improves faithfulness.",
        "4. **Expand the corpus** — unanswerable question failures often indicate genuine knowledge "
        "gaps; ingesting additional documentation sections directly addresses this.",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Evaluation report written to {output_path}")
