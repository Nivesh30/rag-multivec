import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from src.config.settings import Settings, load_settings
from src.generation.base import BaseGenerator, extract_cited_source_ids
from src.generation.factory import build_generator
from src.ingestion.chunker import chunk_documents
from src.middleware.normalizer import Document, Normalizer, PlainTextNormalizer
from src.retrieval.bm25_index import BM25Index
from src.retrieval.retriever import Retriever, ScoredDocument
from src.retrieval.vector_store import ChromaVectorStore
from src.vectorizers.base import BaseVectorizer
from src.vectorizers.factory import build_vectorizer

logger = logging.getLogger("rag_multivec.pipeline")


def _batched(items: List, batch_size: int) -> Iterator[List]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: List[ScoredDocument]
    cited_source_ids: List[str] = field(default_factory=list)


@dataclass
class StreamingRAGAnswer:
    """Result of RAGPipeline.query_stream().

    `sources` are available immediately (retrieval runs synchronously
    before generation starts). Iterate the answer itself - `for chunk in
    streaming_answer:` - to consume the generated text as it arrives;
    `.answer` and `.cited_source_ids` only reflect what's been streamed so
    far, so read them after fully consuming the iterator for the final
    values (a generator raising mid-stream, e.g. GenerationError, still
    leaves whatever was yielded so far in `.answer`).
    """

    question: str
    sources: List[ScoredDocument]
    _token_iter: Iterator[str]
    _parts: List[str] = field(default_factory=list, repr=False)

    def __iter__(self) -> Iterator[str]:
        for chunk in self._token_iter:
            self._parts.append(chunk)
            yield chunk

    @property
    def answer(self) -> str:
        return "".join(self._parts)

    @property
    def cited_source_ids(self) -> List[str]:
        return extract_cited_source_ids(self.answer, self.sources)


class RAGPipeline:
    """End-to-end pipeline: normalize -> chunk -> vectorize -> store -> retrieve -> generate."""

    def __init__(
        self,
        settings: Settings,
        vectorizer: BaseVectorizer = None,
        generator: BaseGenerator = None,
        normalizer: Normalizer = None,
    ):
        self.settings = settings
        self.vectorizer = vectorizer or build_vectorizer(settings.vectorizer)
        self.generator = generator or build_generator(settings.generation)
        self.normalizer = normalizer or PlainTextNormalizer()

        self.vector_store = ChromaVectorStore(
            collection_name=settings.vector_store.collection_name,
            persist_directory=settings.vector_store.persist_directory,
        )
        self.bm25_index = BM25Index() if settings.retrieval.use_hybrid else None
        self.retriever = Retriever(
            vectorizer=self.vectorizer,
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            use_hybrid=settings.retrieval.use_hybrid,
            rrf_k=settings.retrieval.rrf_k,
        )
        # Chunks currently indexed, keyed by source document id - lets us
        # replace a document's chunks in place on re-ingest instead of
        # accumulating stale duplicates in the BM25 corpus.
        self._chunks_by_parent: Dict[str, List[Document]] = {}

        # BM25 has no native persistence, so we save/load its corpus as JSON
        # next to Chroma's own on-disk data - same on/off switch
        # (CHROMA_PERSIST_DIR unset => ephemeral, both in-memory only).
        self._bm25_persist_path: Optional[str] = None
        if self.bm25_index is not None and settings.vector_store.persist_directory:
            self._bm25_persist_path = os.path.join(settings.vector_store.persist_directory, "bm25_corpus.json")
            loaded_chunks = self.bm25_index.load(self._bm25_persist_path)
            for chunk in loaded_chunks:
                parent_id = chunk.metadata.get("parent_id", chunk.id)
                self._chunks_by_parent.setdefault(parent_id, []).append(chunk)

    def ingest(self, raw_records: List[dict]) -> List[Document]:
        """Normalize raw records, chunk them, embed the chunks, and index them.

        Re-ingesting a record with an id already seen replaces its chunks
        (in both the vector store and the BM25 corpus) rather than
        duplicating them.
        """
        started = time.monotonic()
        documents = [self.normalizer.normalize(raw) for raw in raw_records]
        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.chunking.chunk_size,
            chunk_overlap=self.settings.chunking.chunk_overlap,
        )

        replaced = [doc.id for doc in documents if doc.id in self._chunks_by_parent]
        for doc_id in replaced:
            self.vector_store.delete_by_parent_id(doc_id)
            del self._chunks_by_parent[doc_id]

        if chunks:
            embeddings = self._embed_all([chunk.text for chunk in chunks])
            self.vector_store.add_documents(chunks, embeddings)
            for doc in documents:
                self._chunks_by_parent[doc.id] = [c for c in chunks if c.metadata.get("parent_id") == doc.id]

        self._rebuild_bm25()

        elapsed_ms = (time.monotonic() - started) * 1000
        logger.info(
            "ingest(%d record(s)) -> %d chunk(s), %d replaced in %.1fms",
            len(raw_records),
            len(chunks),
            len(replaced),
            elapsed_ms,
        )
        return chunks

    def delete(self, document_id: str) -> None:
        """Remove a previously ingested document's chunks from the index entirely."""
        if document_id not in self._chunks_by_parent:
            logger.info("delete(%s): no-op, not currently indexed", document_id)
            return
        self.vector_store.delete_by_parent_id(document_id)
        del self._chunks_by_parent[document_id]
        self._rebuild_bm25()
        logger.info("delete(%s): removed from index", document_id)

    def _embed_all(self, texts: List[str]) -> List[List[float]]:
        """Embed `texts` in batches of settings.vectorizer.batch_size rather than
        one call, so a large ingest doesn't hit a provider's per-request size/
        rate limits. If any batch fails, the exception propagates and nothing
        from this ingest() call is added to the vector store."""
        batch_size = max(1, self.settings.vectorizer.batch_size)
        if len(texts) <= batch_size:
            return self.vectorizer.embed(texts)

        embeddings: List[List[float]] = []
        batches = list(_batched(texts, batch_size))
        for i, batch in enumerate(batches, start=1):
            logger.info("embedding batch %d/%d (%d chunk(s))", i, len(batches), len(batch))
            embeddings.extend(self.vectorizer.embed(batch))
        return embeddings

    def _all_chunks(self) -> List[Document]:
        return [chunk for chunks in self._chunks_by_parent.values() for chunk in chunks]

    def _rebuild_bm25(self) -> None:
        if self.bm25_index is None:
            return
        self.bm25_index.build(self._all_chunks())
        if self._bm25_persist_path:
            self.bm25_index.save(self._bm25_persist_path)

    def query(self, question: str, top_k: Optional[int] = None) -> RAGAnswer:
        top_k = top_k or self.settings.retrieval.top_k
        started = time.monotonic()
        sources = self.retriever.retrieve(question, top_k=top_k)
        answer = self.generator.generate(question, sources)
        cited_source_ids = extract_cited_source_ids(answer, sources)
        elapsed_ms = (time.monotonic() - started) * 1000
        logger.info(
            "query() -> %d source(s), %d cited, %.1fms total",
            len(sources),
            len(cited_source_ids),
            elapsed_ms,
        )
        return RAGAnswer(question=question, answer=answer, sources=sources, cited_source_ids=cited_source_ids)

    def query_stream(self, question: str, top_k: Optional[int] = None) -> StreamingRAGAnswer:
        """Like query(), but the answer streams token-by-token instead of
        being generated in one blocking call. Retrieval still runs eagerly
        (sources are known before any generation starts); see
        StreamingRAGAnswer for how to consume the streamed text."""
        top_k = top_k or self.settings.retrieval.top_k
        sources = self.retriever.retrieve(question, top_k=top_k)
        logger.info("query_stream() -> %d source(s), generation starting", len(sources))
        return StreamingRAGAnswer(question=question, sources=sources, _token_iter=self.generator.stream(question, sources))


def build_pipeline(settings: Settings = None) -> RAGPipeline:
    return RAGPipeline(settings or load_settings())
