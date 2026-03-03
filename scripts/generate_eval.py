"""
Generate evaluation dataset from ingested documentation.
Usage: python scripts/generate_eval.py [--source data/anthropic_docs] [--output data/eval_sets/eval_dataset.json] [--total 40]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_config
from app.core.logging import setup_logging
from app.generation.llm import LLMFactory
from app.evaluation.sample_generator import generate_dataset

setup_logging()
logger = logging.getLogger("generate_eval")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG evaluation dataset")
    parser.add_argument("--source", default="data/anthropic_docs", help="Directory with source docs")
    parser.add_argument("--output", default="data/eval_sets/eval_dataset.json", help="Output JSON path")
    parser.add_argument("--total", type=int, default=40, help="Total number of samples to generate")
    args = parser.parse_args()

    config = get_config()
    llm = LLMFactory.create(config)

    logger.info(f"Generating {args.total} eval samples from {args.source}")
    samples = generate_dataset(args.source, llm, total=args.total)

    if not samples:
        logger.error("No samples generated — aborting")
        sys.exit(1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved {len(samples)} samples to {out}")
