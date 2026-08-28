# rag-multivec

RAG system supporting multiple vectorization strategies with a normalization middleware layer.

## Structure
- `src/vectorizers/` — pluggable dense embedding backends: Voyage AI, OpenAI, local sentence-transformers
- `src/middleware/` — data normalization pipeline (raw record -> `Document`) before vectorization
- `src/ingestion/` — chunking of normalized documents
- `src/retrieval/` — Chroma vector store, BM25 sparse index, and reciprocal-rank-fusion hybrid retriever
- `src/generation/` — pluggable answer-generation backends (Anthropic Claude, OpenAI; add more by implementing `BaseGenerator`)
- `src/config/` — environment-driven settings for every stage
- `src/pipeline.py` — wires it all together: `ingest()` and `query()`
- `tests/` — unit tests (chunker, hybrid fusion, BM25, full pipeline with fakes)
- `examples/quickstart.py` — minimal end-to-end script

## Design
Middleware normalizes raw input (schema, encoding, cleaning) into a common `Document` format before it hits any vectorizer, so vectorizers stay swappable. Retrieval defaults to hybrid search: dense Chroma similarity + sparse BM25, combined via reciprocal rank fusion. Generation is provider-agnostic behind `BaseGenerator` — swap Anthropic for OpenAI or your own implementation without touching the rest of the pipeline.

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
print(result.sources)  # ScoredDocument list, most relevant first
```

Or run the bundled example: `python -m examples.quickstart`.

## Tests

```bash
pytest
```
