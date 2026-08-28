from dataclasses import dataclass
from typing import List

from src.config.settings import Settings, load_settings
from src.generation.base import BaseGenerator
from src.generation.factory import build_generator
from src.ingestion.chunker import chunk_documents
from src.middleware.normalizer import Document, Normalizer, PlainTextNormalizer
from src.retrieval.bm25_index import BM25Index
from src.retrieval.retriever import Retriever, ScoredDocument
from src.retrieval.vector_store import ChromaVectorStore
from src.vectorizers.base import BaseVectorizer
from src.vectorizers.factory import build_vectorizer


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: List[ScoredDocument]


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
        self._all_chunks: List[Document] = []

    def ingest(self, raw_records: List[dict]) -> List[Document]:
        """Normalize raw records, chunk them, embed the chunks, and index them."""
        documents = [self.normalizer.normalize(raw) for raw in raw_records]
        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.chunking.chunk_size,
            chunk_overlap=self.settings.chunking.chunk_overlap,
        )
        if not chunks:
            return []

        embeddings = self.vectorizer.embed([chunk.text for chunk in chunks])
        self.vector_store.add_documents(chunks, embeddings)

        self._all_chunks.extend(chunks)
        if self.bm25_index is not None:
            self.bm25_index.build(self._all_chunks)

        return chunks

    def query(self, question: str, top_k: int = None) -> RAGAnswer:
        top_k = top_k or self.settings.retrieval.top_k
        sources = self.retriever.retrieve(question, top_k=top_k)
        answer = self.generator.generate(question, sources)
        return RAGAnswer(question=question, answer=answer, sources=sources)


def build_pipeline(settings: Settings = None) -> RAGPipeline:
    return RAGPipeline(settings or load_settings())
