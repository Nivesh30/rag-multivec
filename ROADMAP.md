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
- [ ] **Error handling around LLM/embedding calls.** No retry/backoff or
      typed-exception handling around the Anthropic/OpenAI/Voyage calls in
      `src/vectorizers/` and `src/generation/` — a transient 429/5xx today
      just raises out of `ingest()`/`query()`.
- [ ] **Streaming generation.** `AnthropicGenerator`/`OpenAIGenerator` use
      non-streaming calls; fine for short answers, but there's no path to
      stream tokens back to a caller (e.g. a future API/UI layer).
- [ ] **Persistent BM25.** `BM25Index` is pure in-memory and rebuilt from
      scratch on process restart — pair it with the same persistence model
      Chroma already has (`CHROMA_PERSIST_DIR`).
- [ ] **Structured logging & basic observability.** No logging at all
      today — at minimum log ingest counts, retrieval latency/hit counts,
      and generation token usage per call.

## P2 — scale & interface

- [ ] **Batch embedding for large ingests.** `embed()` is called with the
      full chunk list in one shot; add batching/backpressure for large
      document sets to avoid provider request-size limits.
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
