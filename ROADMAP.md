# Roadmap

Status of `rag-multivec` as of the initial pipeline implementation: a working,
tested RAG system (pluggable dense embeddings, hybrid dense+BM25 retrieval,
pluggable generation) suitable for a demo or small dataset, not yet
production-grade. This is the prioritized list of what's missing.

## P0 — correctness / quality of the core loop

- [ ] **Better chunking.** Current chunker splits on raw word count and can
      cut sentences mid-thought. Move to sentence- or paragraph-aware
      splitting (e.g. respect `.`/`\n\n` boundaries, or token-based chunking
      with a tokenizer instead of `str.split(" ")`).
- [ ] **Retrieval evaluation harness.** No way to know if hybrid search is
      actually better than dense-only on real data. Add a small labeled
      eval set (question -> expected source doc id) and a script that
      reports recall@k / MRR, so retrieval changes can be measured instead
      of guessed at.
- [ ] **Citations in generated answers.** `build_context_block` numbers
      sources (`[1]`, `[2]`, ...) but the generator prompt doesn't ask
      Claude to cite them, and answers don't surface which source(s) were
      actually used. Wire citation markers through to `RAGAnswer`.
- [ ] **Incremental indexing.** `ingest()` rebuilds the entire BM25 index
      from all chunks seen so far on every call (`O(n)` per ingest). Fine
      for a demo, not for repeated small ingests at scale — either persist
      BM25 like Chroma does, or document the current in-memory limit
      clearly and cap it.

## P1 — production readiness

- [ ] **Document updates & deletes.** No way to re-ingest a changed
      document (old chunks would linger) or remove a document's chunks
      from either the vector store or the BM25 index.
- [ ] **Dedup on ingest.** Re-ingesting the same raw record twice currently
      just upserts by id in Chroma but appends duplicate chunks to the
      in-memory BM25 corpus.
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
