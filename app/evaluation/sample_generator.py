"""here are functions responsible for generation samples for the evaluation dataset
all possible options:

- Factual questions with clear, unambiguous answers
- Multi-hop questions that require synthesizing info from multiple chunks
- Questions with no answer in the corpus (to test hallucination resistance)
- Paraphrased questions (same question asked in a different way than the source text)

all options must be distributed equally along the dataset (must contain 40 samples in the end)

the sample example:
{
"question": "What is the maximum number of tools you can pass in a single API request to Claude?",
"ground_truth_answer": "You can pass up to 128 tools in a single API request.",
"ground_truth_context": "The maximum number of tools that can be defined in a single API request is 128."
}

to compose a sample import modules from /app/generation/ llm and prompt for corresponding API calls and prompts
"""
import json
import logging
from pathlib import Path

from app.generation.llm import BaseLLM
from app.generation.prompts import SAMPLE_GENERATION_PROMPT

logger = logging.getLogger("sample_generator")

QUESTION_TYPES = ["factual", "multi_hop", "unanswerable", "paraphrased"]
SAMPLES_PER_TYPE = 10       # 4 types × 10 = 40 total
CHUNK_SIZE_FOR_PROMPT = 3000


def _parse_samples(raw: str, id_prefix: str, q_type: str) -> list[dict]:
    try:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        items: list[dict] = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(f"Failed to parse JSON for type={q_type}: {exc}")
        return []

    for i, item in enumerate(items):
        if "id" not in item:
            item["id"] = f"{id_prefix}{i + 1:03d}"
        item.setdefault("type", q_type)
    return items


def generate_samples_for_type(
    texts: list[str],
    q_type: str,
    n: int,
    llm: BaseLLM,
    id_prefix: str = "q",
) -> list[dict]:
    """Generate n samples of a specific question type from the provided texts."""
    combined = "\n\n---\n\n".join(texts)[:CHUNK_SIZE_FOR_PROMPT]
    prompt = SAMPLE_GENERATION_PROMPT.format(
        n=n,
        text=combined,
    )
    # Append type restriction
    prompt += f"\n\nGenerate ONLY questions of type: {q_type}."

    raw = llm.generate(prompt)
    return _parse_samples(raw, id_prefix, q_type)


def generate_dataset(
    source_dir: str,
    llm: BaseLLM,
    total: int = 40,
) -> list[dict]:
    """
    Read all .md/.txt files from source_dir, generate total samples
    equally distributed across 4 question types.
    Returns list of sample dicts.
    """
    path = Path(source_dir)
    texts: list[str] = []
    for ext in ("*.md", "*.txt"):
        for fp in path.rglob(ext):
            try:
                texts.append(fp.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"Could not read {fp}: {exc}")

    if not texts:
        logger.error(f"No source documents found in {source_dir}")
        return []

    n_per_type = total // len(QUESTION_TYPES)
    all_samples: list[dict] = []
    global_idx = 1

    for q_type in QUESTION_TYPES:
        prefix = f"q{global_idx:03d}_"
        samples = generate_samples_for_type(texts, q_type, n_per_type, llm, prefix)
        # Re-assign clean sequential IDs
        for i, s in enumerate(samples):
            s["id"] = f"q{global_idx:03d}"
            global_idx += 1
        all_samples.extend(samples)
        logger.info(f"Generated {len(samples)} {q_type} samples")

    return all_samples
