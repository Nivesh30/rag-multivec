# Roadmap

Status of `rag-multivec` as of the initial pipeline implementation: a working,
tested RAG system (pluggable dense embeddings, hybrid dense+BM25 retrieval,
pluggable generation) suitable for a demo or small dataset, not yet
production-grade. This is the prioritized list of what's missing.

## P0 — correctness / quality of the core loop

- [x] **Better chunking.** `chunk_document` now splits on sentence/paragraph
      boundaries (`src/ingestion/chunker.py`) and packs whole sentences into
      each chunk, only word-splitting a sentence that alone exceeds
      `chunk_size`. Overlap carries trailing sentences forward instead of a
      raw word slice.
- [x] **Retrieval evaluation harness.** `src/eval/evaluate.py` runs an
      `EvalCase` (question -> expected doc/chunk id) list through a
      `Retriever` and reports recall@k and MRR. See
      `examples/eval_retrieval.py` and `tests/test_eval.py`.
- [x] **Citations in generated answers.** `RAG_SYSTEM_PROMPT` now asks the
      model to cite `[n]` markers it relied on;
      `extract_cited_source_ids()` parses them back to source document ids,
      surfaced as `RAGAnswer.cited_source_ids`.
- [ ] **Incremental indexing.** `ingest()`/`delete()` now rebuild BM25 from
      a deduplicated corpus (no more unbounded duplicate growth - see
      "Dedup on ingest" below), but it's still a full `O(n)` rebuild per
      call rather than a true incremental update. Fine for a demo, not for
      frequent small ingests at scale — either persist BM25 like Chroma
      does, or swap to a sparse index that supports incremental updates.

## P1 — production readiness

- [x] **Document updates & deletes.** `RAGPipeline.ingest()` now replaces
      an existing document's chunks (in both Chroma and the BM25 corpus)
      when re-ingesting the same id, and `RAGPipeline.delete(document_id)`
      removes a document's chunks from both entirely
      (`ChromaVectorStore.delete_by_parent_id`).
- [x] **Dedup on ingest.** Covered by the same change above —
      `_chunks_by_parent` tracks chunks per source document id so a
      re-ingest replaces rather than duplicates them.
- [x] **Error handling around LLM/embedding calls.** The Anthropic/OpenAI
      SDKs already retry connection errors, 429, and 5xx with exponential
      backoff internally (now configurable via `GENERATION_MAX_RETRIES` /
      `VECTORIZER_MAX_RETRIES`, passed to the client's own `max_retries`).
      What was missing is handled now: `src/errors.py` defines
      `EmbeddingError`/`GenerationError`, and each provider call in
      `src/vectorizers/` and `src/generation/` is wrapped in a
      most-specific-first typed exception chain (auth / rate limit /
      connection / status) that raises one of these instead of a raw SDK
      exception leaking out of `ingest()`/`query()`. Voyage's SDK doesn't
      expose a documented typed hierarchy, so it uses a duck-typed
      transient check (`is_transient_by_signature`) for logging purposes
      and always raises `EmbeddingError`.
- [x] **Streaming generation.** `BaseGenerator.stream()` (default: yields
      `generate()`'s result as one chunk, so existing/custom generators
      keep working unmodified) is overridden in `AnthropicGenerator`
      (`client.messages.stream`) and `OpenAIGenerator` (`stream=True`) with
      the same typed exception handling as `generate()`.
      `RAGPipeline.query_stream()` runs retrieval eagerly (sources are
      known immediately) and returns a `StreamingRAGAnswer` you iterate
      for text chunks; `.answer`/`.cited_source_ids` reflect what's been
      consumed so far. See `examples/streaming_query.py`.
- [x] **Persistent BM25.** `BM25Index.save()`/`load()` serialize the
      document corpus as JSON (rank_bm25 has no native serialization, so
      the index is cheaply rebuilt from the saved corpus on load) to
      `<CHROMA_PERSIST_DIR>/bm25_corpus.json` — same on/off switch as
      Chroma's own persistence (unset `CHROMA_PERSIST_DIR` => ephemeral,
      in-memory only for both). `RAGPipeline` loads it on startup
      (repopulating `_chunks_by_parent` so dedup-on-reingest still works
      across a restart) and saves after every `ingest()`/`delete()`.
- [x] **Structured logging & basic observability.** `src/logging_config.py`
      (`configure_logging()`, opt-in - library code never installs
      handlers itself) plus `logging.getLogger("rag_multivec.*")` calls in
      `RAGPipeline.ingest()`/`delete()`/`query()` and `Retriever.retrieve()`
      logging record/chunk counts, hybrid vs. dense mode, hit counts,
      citation counts, and latency in ms. Configurable via `LOG_LEVEL`.

## P2 — scale & interface

- [x] **Batch embedding for large ingests.** `RAGPipeline._embed_all()`
      splits the chunk list into batches of `VECTORIZER_BATCH_SIZE`
      (default 100) and calls `vectorizer.embed()` per batch, logging
      progress; a single batch is used as before for a small ingest. A
      failure partway through raises (via the typed `EmbeddingError` from
      the P1 error-handling work) before anything from that `ingest()`
      call is added to the vector store.
- [ ] **A real document loader layer.** `src/ingestion/` currently only
      has the chunker; add loaders for common formats (PDF, HTML, Markdown,
      CSV) feeding into `PlainTextNormalizer`/custom `Normalizer`s.
- [ ] **HTTP API.** Wrap `RAGPipeline` in a small FastAPI (or similar)
      service with `/ingest` and `/query` endpoints, so it's usable outside
      a Python script.
- [ ] **Auth / multi-tenancy.** If this ever serves more than one user or
      dataset, collections/namespacing per tenant and basic API auth.
- [ ] **Swap vector store without code changes.** `VectorStoreConfig.backend`
      exists but only `chroma` is implemented — either implement
      Qdrant/Pinecone/FAISS behind the same interface or drop the unused
      config field.

## P3 — nice to have

- [ ] **Async pipeline** (`AsyncAnthropic`/`AsyncOpenAI`) for concurrent
      ingestion/query throughput.
- [ ] **Reranking step** (cross-encoder or Claude-as-judge) after hybrid
      retrieval, before generation, for higher-precision top-k.
- [ ] **CLI** (`rag-multivec ingest <path>`, `rag-multivec query "..."`) as
      a thin wrapper over `examples/quickstart.py`.
- [ ] **CI** (GitHub Actions) running `pytest` on push/PR.

---

Suggested order to actually tackle this: P0 items first (they affect whether
the system gives good answers at all), then persistence/error-handling from
P1 before anything touches real user traffic, then P2 only once there's a
concrete need to scale past a single-process demo.
