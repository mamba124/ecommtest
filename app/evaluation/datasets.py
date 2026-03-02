"""encompasses sample_generator in the flow/script that can be run separately.
the final file must be saved as eval_dataset.json in /data/eval_sets"""
import argparse
import json
import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("datasets")

DATASET_PATH = "data/eval_sets/eval_dataset.json"

QuestionType = Literal["factual", "multi_hop", "unanswerable", "paraphrased"]


class EvalQuestion(BaseModel):
    id: str
    question_type: QuestionType = Field(..., alias="type")
    question: str
    ground_truth_answer: str
    ground_truth_context: Optional[str] = None

    model_config = {"populate_by_name": True}


def load_dataset(path: str) -> list[EvalQuestion]:
    with open(path, "r") as f:
        data = json.load(f)
    return [EvalQuestion.model_validate(item) for item in data]


def save_dataset(samples: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    logger.info(f"Saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Generate evaluation dataset")
    parser.add_argument(
        "--source-dir", default="data/raw_docs",
        help="Directory containing source documents"
    )
    parser.add_argument(
        "--output", default=DATASET_PATH,
        help="Output path for eval_dataset.json"
    )
    parser.add_argument("--total", type=int, default=40, help="Total samples (divisible by 4)")
    args = parser.parse_args()

    from app.core.config import get_config
    from app.generation.llm import LLMFactory
    from app.evaluation.sample_generator import generate_dataset

    config = get_config()
    llm = LLMFactory.create(config)

    samples = generate_dataset(args.source_dir, llm, args.total)
    save_dataset(samples, args.output)
