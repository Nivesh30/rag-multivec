# rag-multivec

RAG system supporting multiple vectorization strategies with a normalization middleware layer.

## Structure
- `src/vectorizers/` — pluggable embedding backends (dense, sparse, hybrid)
- `src/middleware/` — data normalization pipeline before vectorization
- `src/ingestion/` — document loading/chunking
- `src/retrieval/` — vector store + query interface
- `src/config/` — settings per vectorizer/backend
- `tests/`
- `data/`

## Design
Middleware normalizes raw input (schema, encoding, cleaning) into a common `Document` format before it hits any vectorizer, so vectorizers stay swappable.
