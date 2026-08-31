import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: List[ScoredDocument]
    cited_source_ids: List[str] = field(default_factory=list)


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
            embeddings = self.vectorizer.embed([chunk.text for chunk in chunks])
            self.vector_store.add_documents(chunks, embeddings)
            for doc in documents:
                self._chunks_by_parent[doc.id] = [c for c in chunks if c.metadata.get("parent_id") == doc.id]

        if self.bm25_index is not None:
            self.bm25_index.build(self._all_chunks())

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
        if self.bm25_index is not None:
            self.bm25_index.build(self._all_chunks())
        logger.info("delete(%s): removed from index", document_id)

    def _all_chunks(self) -> List[Document]:
        return [chunk for chunks in self._chunks_by_parent.values() for chunk in chunks]

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


def build_pipeline(settings: Settings = None) -> RAGPipeline:
    return RAGPipeline(settings or load_settings())
