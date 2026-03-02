"""import datasets for evaluation from /data/eval_sets, 
feed each pair to the entrypoint in order to obtain the generated result. 
use call to LLM, prompt for evaluating the ground truth and generated answer.
after all samples results collected build a report based on the logic from module report

CLI usage:
    python -m app.evaluation.runner --dataset data/eval_sets/eval_dataset.json --output docs/eval_report.md
"""
import argparse
import logging
from dataclasses import dataclass, field

import httpx

from app.evaluation.datasets import load_dataset, EvalQuestion

logger = logging.getLogger("eval.runner")

BASE_URL = "http://localhost:8080"


@dataclass
class QuestionResult:
    id: str
    type: str
    question: str
    answer: str
    retrieved_chunks: list[str]
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    answer_correctness: float
    notes: str = ""


@dataclass
class EvalResults:
    per_question: list[QuestionResult] = field(default_factory=list)

    def aggregate(self) -> dict[str, float]:
        if not self.per_question:
            return {}
        metrics = [
            "context_precision", "context_recall",
            "faithfulness", "answer_relevancy", "answer_correctness",
        ]
        return {
            m: sum(getattr(r, m) for r in self.per_question) / len(self.per_question)
            for m in metrics
        }


def run_evaluation(dataset_path: str, config=None) -> EvalResults:
    from app.core.config import get_config
    from app.ingestion.embeddings import EmbedderFactory
    from app.generation.llm import LLMFactory
    import app.evaluation.metrics as M

    cfg = config or get_config()
    embedder = EmbedderFactory.create(cfg)
    llm = LLMFactory.create(cfg)

    dataset: list[EvalQuestion] = load_dataset(dataset_path)
    results = EvalResults()

    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        for item in dataset:
            try:
                resp = client.post("/query", json={"question": item.question, "stream": False})
                resp.raise_for_status()
                data = resp.json()
                answer: str = data["answer"]
                chunk_texts: list[str] = [c.get("chunk_id", "") for c in data.get("citations", [])]
                gt_ctx = item.ground_truth_context or ""

                cp = M.context_precision(item.question, chunk_texts, gt_ctx)
                cr = M.context_recall(chunk_texts, gt_ctx)
                f  = M.faithfulness(answer, chunk_texts, llm)
                ar = M.answer_relevancy(item.question, answer, llm, embedder)
                ac = M.answer_correctness(answer, item.ground_truth_answer, embedder)

                results.per_question.append(QuestionResult(
                    id=item.id,
                    type=item.question_type,
                    question=item.question,
                    answer=answer,
                    retrieved_chunks=chunk_texts,
                    context_precision=cp,
                    context_recall=cr,
                    faithfulness=f,
                    answer_relevancy=ar,
                    answer_correctness=ac,
                ))
                logger.info(
                    f"Evaluated {item.id}: "
                    f"CP={cp:.2f} CR={cr:.2f} F={f:.2f} AR={ar:.2f} AC={ac:.2f}"
                )
            except Exception as exc:
                logger.error(f"Failed to evaluate {item.id}: {exc}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval_sets/eval_dataset.json")
    parser.add_argument("--output", default="docs/eval_report.md")
    args = parser.parse_args()

    from app.evaluation.report import generate_report
    results = run_evaluation(args.dataset)
    generate_report(results, args.output)
