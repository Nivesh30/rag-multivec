# rag-multivec

RAG system supporting multiple vectorization strategies with a normalization middleware layer.

## Structure
- `src/vectorizers/` — pluggable dense embedding backends: Voyage AI, OpenAI, local sentence-transformers
- `src/middleware/` — data normalization pipeline (raw record -> `Document`) before vectorization
- `src/ingestion/` — sentence-aware chunking, plus file loaders (`.txt`/`.md`/`.html`/`.pdf`/`.csv`, and a `load_directory()` walker) that produce raw records ready for `ingest()`
- `src/retrieval/` — Chroma vector store, BM25 sparse index, and reciprocal-rank-fusion hybrid retriever
- `src/generation/` — pluggable answer-generation backends (Anthropic Claude, OpenAI; add more by implementing `BaseGenerator`), with inline `[n]` citations parsed back to source document ids
- `src/eval/` — retrieval evaluation harness (recall@k, MRR) against a labeled question -> document set
- `src/errors.py` — typed `EmbeddingError`/`GenerationError` raised on unrecoverable provider failures
- `src/logging_config.py` — opt-in `configure_logging()` for ingest/retrieve/query counts and latency
- `src/config/` — environment-driven settings for every stage
- `src/pipeline.py` — wires it all together: `ingest()`, `delete()`, and `query()`
- `src/api.py` — optional FastAPI HTTP service around the pipeline (`/ingest`, `/query`, `/query/stream`, `/documents/{id}`, `/health`)
- `tests/` — unit tests (chunker, hybrid fusion, BM25, retrieval eval, full pipeline with fakes)
- `examples/quickstart.py` — minimal end-to-end script
- `examples/eval_retrieval.py` — minimal retrieval-quality eval script
- `examples/streaming_query.py` — stream an answer token-by-token instead of waiting for the full response
- `examples/ingest_directory.py` — ingest every supported file under a directory
- `ROADMAP.md` — prioritized list of what's implemented vs. still missing

## Design
Middleware normalizes raw input (schema, encoding, cleaning) into a common `Document` format before it hits any vectorizer, so vectorizers stay swappable. Chunking packs whole sentences per chunk (never splitting mid-sentence) with sentence-level overlap between consecutive chunks. Retrieval defaults to hybrid search: dense Chroma similarity + sparse BM25, combined via reciprocal rank fusion. Generation is provider-agnostic behind `BaseGenerator` — swap Anthropic for OpenAI or your own implementation without touching the rest of the pipeline. Re-ingesting a document id replaces its chunks (in both Chroma and the BM25 corpus) instead of duplicating them; `pipeline.delete(document_id)` removes a document entirely. Set `CHROMA_PERSIST_DIR` to persist both the vector store and the BM25 corpus to disk; leave it unset for an ephemeral, in-memory-only pipeline (e.g. in tests).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in the backend(s) you're using
```

You only need the dependency/API key for the backends you select via `VECTORIZER_BACKEND` and `GENERATION_PROVIDER`:

| Setting | Options |
|---|---|
| `VECTORIZER_BACKEND` | `voyage`, `openai`, `sentence_transformers` (local, no key needed) |
| `GENERATION_PROVIDER` | `anthropic`, `openai` |

## Usage

```python
from src.config.settings import load_settings
from src.pipeline import build_pipeline

pipeline = build_pipeline(load_settings())

pipeline.ingest([
    {"id": "doc-1", "text": "..."},
    {"id": "doc-2", "text": "..."},
])

result = pipeline.query("What does the document say about X?")
print(result.answer)
print(result.sources)            # ScoredDocument list, most relevant first
print(result.cited_source_ids)   # subset of source ids the model actually cited

pipeline.delete("doc-1")  # remove a document (and its chunks) from the index

# Stream the answer instead of waiting for the full response:
streaming_answer = pipeline.query_stream("What does the document say about X?")
for chunk in streaming_answer:
    print(chunk, end="", flush=True)
print(streaming_answer.cited_source_ids)  # available after the loop consumes the stream
```

Or load documents from disk instead of hand-writing raw records:

```python
from src.ingestion.loaders import load_directory, load_csv_file

pipeline.ingest(load_directory("./docs"))                                  # .txt/.md/.html/.pdf
pipeline.ingest(load_csv_file("./faq.csv", text_column="answer", id_column="id"))
```

Or run the bundled examples: `python -m examples.quickstart`, `python -m examples.eval_retrieval`, `python -m examples.streaming_query`, and `python -m examples.ingest_directory <dir>`.

### HTTP API

```bash
uvicorn src.api:app --reload
```

| Endpoint | Method | Body | Notes |
|---|---|---|---|
| `/ingest` | POST | `{"records": [{"id": ..., "text": ...}, ...]}` | Same raw-record shape as `pipeline.ingest()` |
| `/query` | POST | `{"question": "...", "top_k": 5}` | `top_k` optional |
| `/query/stream` | POST | same as `/query` | Streams the answer as chunked `text/plain` |
| `/documents/{id}` | DELETE | — | |
| `/health` | GET | — | |

A `RAGError` (embedding/generation failure surviving retries) returns HTTP 502; a malformed ingest record returns 400. `create_app(pipeline=...)` lets you inject a pipeline (e.g. in tests) instead of building the default one from environment settings.

The Anthropic/OpenAI SDKs already retry transient (429/5xx/connection) failures internally — tune how many via `GENERATION_MAX_RETRIES`/`VECTORIZER_MAX_RETRIES`. A failure that survives those retries raises `src.errors.GenerationError`/`EmbeddingError` rather than a raw SDK exception. Call `src.logging_config.configure_logging()` once at startup (as the examples do) to see per-call counts and latency; control verbosity with `LOG_LEVEL`. Large ingests are embedded in batches of `VECTORIZER_BATCH_SIZE` (default 100) rather than one request, to stay under provider request-size limits.

## Tests

```bash
pytest
```
