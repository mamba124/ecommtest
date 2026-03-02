# RAG Service — Complete Technical Specification

---

## 0. Project Layout

```
rag-service/
├── docker-compose.yml
├── .env.example
├── config.yaml                    # all tuneable parameters live here
├── app/
│   ├── main.py                    # FastAPI entrypoint
│   ├── api/
│   │   ├── routes/
│   │   │   ├── ingest.py
│   │   │   ├── query.py
│   │   │   ├── health.py
│   │   │   └── research.py        # agentic endpoint
│   │   └── models.py              # Pydantic request/response schemas
│   ├── core/
│   │   ├── config.py              # loads config.yaml + env vars
│   │   ├── loader_factory.py      # file type dispatch
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── vector_store.py        # ChromaDB wrapper
│   │   ├── retriever.py           # hybrid search (vector + BM25) + reranker
│   │   ├── llm.py                 # LLM abstraction (Ollama / Gemini / Groq)
│   │   ├── cache.py               # in-memory query cache
│   │   └── agent.py               # auto-research agent
│   └── eval/
│       ├── dataset.py             # load/validate eval_dataset.json
│       ├── runner.py              # run all metrics
│       ├── metrics.py             # faithfulness, relevancy, precision, recall, correctness
│       └── report.py              # generates eval_report.md
├── scripts/
│   └── scrape_docs.py             # one-shot scraper for Anthropic docs
├── data/
│   ├── raw_docs/                  # scraped documents land here
│   └── eval_dataset.json
├── docs/
│   ├── eval_report.md             # generated output
│   └── claude-code-session/       # AI session export
├── ui/
│   └── index.html                 # plain HTML frontend (no framework)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 1. Configuration (`config.yaml`)

All runtime parameters must be read from this file. No magic constants anywhere in the code.

```yaml
llm:
  provider: "ollama"          # "ollama" | "gemini" | "groq"
  model: "llama3.2"           # model name per provider
  base_url: "http://ollama:11434"
  temperature: 0.1
  max_tokens: 1024

embeddings:
  provider: "ollama"          # "ollama" | "sentence_transformers"
  model: "nomic-embed-text"
  base_url: "http://ollama:11434"
  dimension: 768

chunking:
  strategy: "recursive"       # "recursive" | "fixed" | "semantic"
  chunk_size: 800
  chunk_overlap: 160          # 20% of chunk_size
  separators: ["\n\n", "\n", ". ", " ", ""]

retrieval:
  top_k: 10                   # initial retrieval pool
  rerank_top_k: 4             # after reranking, keep this many
  use_hybrid: true            # vector + BM25
  hybrid_alpha: 0.7           # weight for vector score (1-alpha = BM25)
  use_reranker: true
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

vector_store:
  provider: "chromadb"
  host: "chromadb"
  port: 8000
  collection_name: "rag_docs"

cache:
  enabled: true
  max_size: 512               # max cached queries (LRU)

agent:
  max_sub_questions: 5
  synthesis_model: null       # null = use llm.model
```

---

## 2. Environment Variables (`.env.example`)

```env
# Required only if provider = gemini or groq
GEMINI_API_KEY=
GROQ_API_KEY=

# Optional overrides
LOG_LEVEL=INFO
```

---

## 3. Docker Compose (`docker-compose.yml`)

Three services: **api**, **chromadb**, **ollama**.

```yaml
version: "3.9"

services:
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    # pulls required models on startup
    entrypoint: >
      sh -c "ollama serve &
             sleep 5 &&
             ollama pull llama3.2 &&
             ollama pull nomic-embed-text &&
             wait"

  api:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./config.yaml:/app/config.yaml
    env_file: .env
    depends_on:
      chromadb:
        condition: service_healthy
      ollama:
        condition: service_started
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 15s
      timeout: 5s
      retries: 5

volumes:
  chroma_data:
  ollama_data:
```

---

## 4. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# system deps for PDF parsing
RUN apt-get update && apt-get install -y \
    libmagic1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## 5. Requirements (`requirements.txt`)

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.0
pydantic-settings==2.2.0
pyyaml==6.0.1

# Document loading
pymupdf==1.24.3           # PDF → fitz
python-magic==0.4.27
markdown==3.6

# Chunking & embeddings
langchain-text-splitters==0.2.0
sentence-transformers==3.0.0

# Vector store
chromadb==0.5.0

# Hybrid search
rank_bm25==0.2.2

# Reranker
# (sentence-transformers already covers cross-encoder)

# LLM clients
ollama==0.2.1
google-generativeai==0.7.0
groq==0.9.0

# HTTP (scraper)
httpx==0.27.0
beautifulsoup4==4.12.3

# Evaluation
ragas==0.1.15
datasets==2.20.0

# Utilities
python-dotenv==1.0.1
cachetools==5.3.3
sse-starlette==2.1.0      # streaming
```

---

## 6. Pydantic Schemas (`app/api/models.py`)

Every request and response must use these schemas — no raw dicts.

```python
from pydantic import BaseModel, Field
from typing import Optional

# ── Ingest ──────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    path: str = Field(..., description="Absolute or relative path to folder of documents")

class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    collection: str

# ── Query ────────────────────────────────────────────────────────────────
class Citation(BaseModel):
    chunk_id: str
    source: str
    page: Optional[int] = None
    score: float

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None          # overrides config if set
    stream: bool = False

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    from_cache: bool = False

# ── Health ───────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str                           # "ok" | "degraded" | "error"
    vector_db: str
    embedding_model: str
    llm: str
    collection_count: int

# ── Research Agent ───────────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    topic: str
    max_sub_questions: Optional[int] = None

class GapItem(BaseModel):
    sub_question: str
    reason: str

class ResearchResponse(BaseModel):
    topic: str
    sub_questions: list[str]
    report: str
    gaps: list[GapItem]
    all_citations: list[Citation]
```

---

## 7. API Routes

### `POST /ingest`

**Logic:**
1. Validate `path` exists on disk.
2. Walk directory recursively, collect all `.pdf`, `.md`, `.txt` files.
3. For each file: dispatch to `LoaderFactory` → get `Document` objects.
4. Run all documents through `Chunker` → list of chunks.
5. Batch-embed all chunks via `Embedder`.
6. Upsert into ChromaDB collection (idempotent — use `chunk_id` as document ID so re-ingestion is safe).
7. Return `IngestResponse`.

**Error handling:** If any single file fails to load, log the error and continue. Return summary including failed files.

---

### `POST /query`

**Logic:**
1. Check cache — if hit, return cached response with `from_cache: true`.
2. Embed the question.
3. `Retriever.search(query, query_embedding)` → ranked chunks (hybrid + rerank).
4. Build prompt from `SYSTEM_PROMPT` template + injected chunks.
5. Call LLM.
6. Parse response, attach citations from retrieved chunks.
7. Store in cache.
8. If `stream: true`, return `StreamingResponse` via SSE. Otherwise return `QueryResponse`.

**Streaming format (SSE):**
```
data: {"token": "The"}
data: {"token": " maximum"}
...
data: {"done": true, "citations": [...]}
```

---

### `GET /health`

Returns liveness of each component. Must not raise — catch all exceptions and return `status: "degraded"` with details.

---

### `POST /research`

See Section 11 (Agentic Feature).

---

## 8. Core Components

### 8.1 `LoaderFactory` (`app/core/loader_factory.py`)

```python
class LoaderFactory:
    @staticmethod
    def load(filepath: str) -> list[Document]:
        """
        Returns list of Document(text=..., metadata={source, page, section}).
        Dispatch by file extension.
        """
```

| Extension | Library | Notes |
|-----------|---------|-------|
| `.pdf` | `pymupdf` (fitz) | Extract text page-by-page; store `page` in metadata |
| `.md` | stdlib `open()` | Read as plain text; store H1/H2 headings as `section` |
| `.txt` | stdlib `open()` | Read as plain text |

**`Document` dataclass:**
```python
@dataclass
class Document:
    text: str
    metadata: dict   # keys: source (str), page (int|None), section (str|None)
```

---

### 8.2 `Chunker` (`app/core/chunker.py`)

Use `langchain_text_splitters.RecursiveCharacterTextSplitter` as the default implementation.

```python
class Chunker:
    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """
        Returns list of Chunk with stable deterministic chunk_id.
        chunk_id = sha256(source_path + str(chunk_index))[:12]
        """
```

**`Chunk` dataclass:**
```python
@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict   # inherits from Document, adds chunk_index
```

---

### 8.3 `Embedder` (`app/core/embedder.py`)

Abstract interface with two implementations:

```python
class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    
    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

class OllamaEmbedder(BaseEmbedder): ...
class SentenceTransformerEmbedder(BaseEmbedder): ...
```

`OllamaEmbedder` calls `POST http://ollama:11434/api/embeddings` with `{"model": ..., "prompt": ...}`.

Batch embedding: process in batches of 32 to avoid memory issues.

---

### 8.4 `VectorStore` (`app/core/vector_store.py`)

Thin wrapper around ChromaDB client.

```python
class VectorStore:
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...
```

**`RetrievedChunk`:**
```python
@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    score: float       # cosine similarity, 0–1
```

ChromaDB stores: `embeddings`, `documents` (chunk text), `metadatas`, `ids` (chunk_id).

---

### 8.5 `Retriever` (`app/core/retriever.py`)

Implements hybrid search and optional reranking.

```python
class Retriever:
    def search(self, query: str, query_embedding: list[float]) -> list[RetrievedChunk]:
        """
        1. Vector search → top_k results (scores normalized 0–1)
        2. BM25 search → top_k results (scores normalized 0–1)
        3. Fuse with RRF: score = alpha * vector_score + (1-alpha) * bm25_score
        4. Take top_k fused results
        5. If use_reranker: pass (query, chunk_text) pairs through cross-encoder
           → reorder by cross-encoder score, keep rerank_top_k
        6. Return final ranked list
        """
```

**BM25 index:** built in-memory at startup by loading all chunk texts from ChromaDB. Rebuilt after each ingest. Store as `self._bm25_index: BM25Okapi`.

**Reranker:** `sentence_transformers.CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`. Run `model.predict([(query, chunk.text) for chunk in candidates])` → sort descending.

---

### 8.6 `LLM` (`app/core/llm.py`)

```python
class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str: ...
    
    @abstractmethod
    def stream(self, prompt: str, system: str = "") -> Iterator[str]: ...

class OllamaLLM(BaseLLM): ...
class GeminiLLM(BaseLLM): ...
class GroqLLM(BaseLLM): ...
```

**Factory:** `LLMFactory.create(config) -> BaseLLM` based on `config.llm.provider`.

---

### 8.7 Prompt Template

This is the exact prompt to use for query answering. Do not improvise.

```python
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
```

**Context injection format:**
```
[chunk_id: abc123] (source: docs/tools.md, page: 3)
The maximum number of tools...

[chunk_id: def456] (source: docs/models.md, page: 1)
Claude Opus supports...
```

---

### 8.8 `Cache` (`app/core/cache.py`)

LRU cache keyed on `sha256(question.strip().lower())`.

```python
from cachetools import LRUCache

class QueryCache:
    def __init__(self, max_size: int): ...
    def get(self, question: str) -> QueryResponse | None: ...
    def set(self, question: str, response: QueryResponse) -> None: ...
```

---

## 9. Scraper (`scripts/scrape_docs.py`)

Scrapes the Anthropic documentation. Standalone script, not part of the API.

```
Usage: python scripts/scrape_docs.py --output data/raw_docs --max-pages 50
```

**Target:** `https://docs.anthropic.com/en/docs/`

**Logic:**
1. Fetch sitemap or crawl starting from the root page.
2. Find all internal links (`/en/docs/...`).
3. For each page: extract main content (`<article>` or `<main>` tag), convert to Markdown using `html2text`.
4. Save as `data/raw_docs/{slug}.md`.
5. Respect `robots.txt`. Add 500ms delay between requests.
6. Stop at `--max-pages`.

**Dependencies to add:** `html2text==2024.2.26`

---

## 10. Evaluation Pipeline (`app/eval/`)

### 10.1 Dataset Format (`data/eval_dataset.json`)

```json
[
  {
    "id": "q001",
    "type": "factual",
    "question": "What is the maximum number of tools you can pass in a single API request to Claude?",
    "ground_truth_answer": "You can pass up to 128 tools in a single API request.",
    "ground_truth_context": "The maximum number of tools that can be defined in a single API request is 128."
  },
  {
    "id": "q002",
    "type": "unanswerable",
    "question": "What is Claude's monthly active user count?",
    "ground_truth_answer": "UNANSWERABLE",
    "ground_truth_context": null
  }
]
```

**Required question types (minimum 20 total):**
- `factual` — at least 8
- `multi_hop` — at least 4 (require synthesizing ≥2 chunks)
- `unanswerable` — at least 4 (test hallucination resistance)
- `paraphrased` — at least 4 (same fact, different wording from source)

---

### 10.2 Metrics (`app/eval/metrics.py`)

Implement each metric as a standalone function. Use RAGAS where available; implement manually as fallback.

```python
def context_precision(question: str, retrieved_chunks: list[str], ground_truth_context: str) -> float:
    """Fraction of retrieved chunks that are relevant to the ground truth."""

def context_recall(retrieved_chunks: list[str], ground_truth_context: str) -> float:
    """Fraction of ground truth context covered by retrieved chunks."""

def faithfulness(answer: str, retrieved_chunks: list[str], llm: BaseLLM) -> float:
    """
    Prompt the LLM to score: does the answer contain only claims 
    supported by the retrieved chunks? Returns 0–1.
    """

def answer_relevancy(question: str, answer: str, llm: BaseLLM) -> float:
    """
    Prompt the LLM to generate N paraphrases of the question from the answer,
    measure cosine similarity of their embeddings to the original question embedding.
    """

def answer_correctness(answer: str, ground_truth: str, embedder: BaseEmbedder) -> float:
    """
    For unanswerable: 1.0 if answer contains refusal phrase, 0.0 otherwise.
    For others: cosine similarity between answer embedding and ground_truth embedding.
    """
```

---

### 10.3 Evaluation Runner (`app/eval/runner.py`)

```python
def run_evaluation(dataset_path: str, config: Config) -> EvalResults:
    """
    For each question in dataset:
      1. Call the live POST /query endpoint (use httpx)
      2. Collect retrieved chunks from response
      3. Compute all 5 metrics
      4. Store per-question results
    Return aggregate means + per-question breakdown.
    """
```

**Run via CLI:**
```
python -m app.eval.runner --dataset data/eval_dataset.json --output docs/eval_report.md
```

---

### 10.4 Evaluation Report (`docs/eval_report.md`)

Generated Markdown report with this exact structure:

```markdown
# RAG Evaluation Report

## Aggregate Scores

| Metric              | Score |
|---------------------|-------|
| Context Precision   | 0.XX  |
| Context Recall      | 0.XX  |
| Faithfulness        | 0.XX  |
| Answer Relevancy    | 0.XX  |
| Answer Correctness  | 0.XX  |

## Per-Question Breakdown

| ID   | Type        | CP   | CR   | F    | AR   | AC   | Notes |
|------|-------------|------|------|------|------|------|-------|
| q001 | factual     | 1.00 | 0.90 | 1.00 | 0.95 | 0.88 |       |

## Failure Case Analysis

### Case 1: [Question ID + text]
**Metrics:** CP=X, CR=X, F=X
**Retrieved chunks:** [list]
**Generated answer:** ...
**Ground truth:** ...
**Analysis:** [why it failed]

## Improvement Suggestions

1. ...
2. ...
```

---

## 11. Agentic Feature — Auto-Research Agent (`app/core/agent.py`)

**Endpoint:** `POST /research`

**Full pipeline:**

```
Input: { "topic": "Claude tool use" }
    ↓
Step 1 — Sub-question generation
  Prompt LLM: "Given the topic '{topic}', generate {N} specific sub-questions 
               that together would give a comprehensive understanding.
               Return as JSON array of strings."
    ↓
Step 2 — Parallel retrieval
  For each sub-question: call Retriever.search() concurrently (asyncio.gather)
    ↓
Step 3 — Parallel answer generation
  For each sub-question + its chunks: call LLM.generate() concurrently
    ↓
Step 4 — Gap detection
  For each sub-question where answer contains refusal phrase → mark as gap
    ↓
Step 5 — Synthesis
  Prompt LLM with all sub-questions + answers:
  "Synthesize these findings into a structured report on '{topic}'.
   Use markdown with sections. Be factual."
    ↓
Output: { "report": "...", "gaps": [...], "sub_questions": [...], "all_citations": [...] }
```

**Concurrency:** use `asyncio.gather` with a semaphore of 3 to limit concurrent LLM calls.

---

## 12. Simple Web UI (`ui/index.html`)

Single HTML file. No framework, no build step. Served statically by FastAPI via `StaticFiles`.

**Mount in `main.py`:**
```python
app.mount("/ui", StaticFiles(directory="ui"), name="ui")
```

**Features:**
- Text input for question
- "Ask" button → calls `POST /query`
- Renders answer in a styled div
- Shows citations as clickable pills below the answer
- "Research" tab → calls `POST /research`, renders report as formatted HTML
- Uses `EventSource` for streaming when `stream: true`
- Dark/light mode toggle

**No external dependencies** — use plain CSS variables and vanilla JS only.

---

## 13. FastAPI App Entry Point (`app/main.py`)

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.core.config import get_config
from app.core.vector_store import VectorStore
from app.core.embedder import EmbedderFactory
from app.core.retriever import Retriever
from app.core.llm import LLMFactory
from app.core.cache import QueryCache
from app.api.routes import ingest, query, health, research

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize all singletons at startup
    config = get_config()
    app.state.config = config
    app.state.vector_store = VectorStore(config)
    app.state.embedder = EmbedderFactory.create(config)
    app.state.retriever = Retriever(app.state.vector_store, app.state.embedder, config)
    app.state.llm = LLMFactory.create(config)
    app.state.cache = QueryCache(config.cache.max_size)
    yield

app = FastAPI(title="RAG Service", lifespan=lifespan)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(health.router)
app.include_router(research.router)
app.mount("/ui", StaticFiles(directory="ui"), name="ui")
```

All route handlers access shared state via `request.app.state.*`. No global variables.

---

## 14. Error Handling Contract

| Scenario | HTTP Status | Response body |
|----------|------------|---------------|
| Path not found in `/ingest` | 400 | `{"detail": "Path does not exist: ..."}` |
| No documents found in path | 400 | `{"detail": "No supported files found in path"}` |
| LLM unreachable | 503 | `{"detail": "LLM service unavailable"}` |
| ChromaDB unreachable | 503 | `{"detail": "Vector store unavailable"}` |
| Empty question | 422 | FastAPI default validation error |
| All other unhandled | 500 | `{"detail": "Internal error", "error": str(e)}` |

Use a global FastAPI exception handler for the 500 case. All other cases raise `HTTPException` explicitly.

---

## 15. Logging

Use Python `logging` module. Format:
```
2024-01-15 10:23:45 INFO  [ingest] Loaded 47 documents, created 312 chunks
2024-01-15 10:23:52 INFO  [retriever] Query: "what is tool use" → 4 chunks retrieved (hybrid, reranked)
2024-01-15 10:23:53 INFO  [cache] Cache hit for question hash a3f2...
```

Log level from `LOG_LEVEL` env var (default: `INFO`). Log to stdout only.

---

## 16. Implementation Notes & Constraints

- **No authentication, rate limiting, or multi-tenancy.** This is explicitly out of scope per the task.
- **Single ChromaDB collection.** Re-ingestion is idempotent via stable `chunk_id`.
- **BM25 index is in-memory.** Rebuilt from ChromaDB on startup and after each ingest. This is acceptable for 30–50 docs.
- **Cross-encoder runs on CPU.** Slow is fine per hardware expectations.
- **Everything must work with `docker-compose up` on a fresh machine.** Ollama model pulls happen in the container entrypoint.
- **Python 3.11+** for type hint syntax (`list[str]` not `List[str]`).
- **All functions must have type hints.** No untyped function signatures.
- **Config is a singleton.** Load once via `functools.lru_cache` on `get_config()`.
