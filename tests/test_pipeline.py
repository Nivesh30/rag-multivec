import hashlib
from typing import List

from src.config.settings import (
    ChunkingConfig,
    GenerationConfig,
    RetrievalConfig,
    Settings,
    VectorStoreConfig,
    VectorizerConfig,
)
from src.generation.base import BaseGenerator
from src.pipeline import RAGPipeline
from src.retrieval.retriever import ScoredDocument
from src.vectorizers.base import BaseVectorizer


class FakeVectorizer(BaseVectorizer):
    """Deterministic bag-of-words style embedding so similar text -> similar vectors."""

    _VOCAB = ["fox", "vector", "database", "dog", "python", "search"]

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([float(lowered.count(word)) for word in self._VOCAB])
        return vectors


class FakeGenerator(BaseGenerator):
    def __init__(self):
        self.last_context = None

    def generate(self, question: str, context: List[ScoredDocument]) -> str:
        self.last_context = context
        joined = " | ".join(s.document.text for s in context)
        return f"Answer to '{question}' using: {joined}"


def _settings(tmp_path, use_hybrid: bool) -> Settings:
    return Settings(
        vectorizer=VectorizerConfig(backend="sentence_transformers"),
        vector_store=VectorStoreConfig(
            backend="chroma",
            collection_name=f"test-{hashlib.md5(str(tmp_path).encode()).hexdigest()[:8]}",
            persist_directory=None,
        ),
        generation=GenerationConfig(provider="anthropic"),
        chunking=ChunkingConfig(chunk_size=50, chunk_overlap=5),
        retrieval=RetrievalConfig(top_k=2, use_hybrid=use_hybrid, rrf_k=60),
    )


def test_pipeline_ingest_and_query(tmp_path):
    settings = _settings(tmp_path, use_hybrid=True)
    pipeline = RAGPipeline(settings, vectorizer=FakeVectorizer(), generator=FakeGenerator())

    chunks = pipeline.ingest(
        [
            {"id": "doc-fox", "text": "The quick fox uses vector search over a database."},
            {"id": "doc-dog", "text": "A lazy dog sleeps in the python garden all day."},
        ]
    )
    assert len(chunks) == 2

    result = pipeline.query("Tell me about the fox and vector database", top_k=1)

    assert result.question.startswith("Tell me about the fox")
    assert len(result.sources) == 1
    assert "fox" in result.sources[0].document.text.lower()
    assert result.answer.startswith("Answer to")


def test_pipeline_dense_only(tmp_path):
    settings = _settings(tmp_path, use_hybrid=False)
    pipeline = RAGPipeline(settings, vectorizer=FakeVectorizer(), generator=FakeGenerator())

    pipeline.ingest([{"id": "doc-fox", "text": "The quick fox uses vector search."}])
    result = pipeline.query("fox", top_k=1)

    assert len(result.sources) == 1


def test_pipeline_reingest_replaces_stale_chunks(tmp_path):
    settings = _settings(tmp_path, use_hybrid=True)
    pipeline = RAGPipeline(settings, vectorizer=FakeVectorizer(), generator=FakeGenerator())

    pipeline.ingest([{"id": "doc-fox", "text": "The quick fox uses vector search."}])
    assert pipeline.vector_store.count() == 1
    assert len(pipeline.bm25_index) == 1

    # Re-ingesting the same id with different text should replace, not duplicate.
    pipeline.ingest([{"id": "doc-fox", "text": "A completely different sentence about python."}])
    assert pipeline.vector_store.count() == 1
    assert len(pipeline.bm25_index) == 1

    result = pipeline.query("python", top_k=1)
    assert "python" in result.sources[0].document.text.lower()


def test_pipeline_delete_removes_document(tmp_path):
    settings = _settings(tmp_path, use_hybrid=True)
    pipeline = RAGPipeline(settings, vectorizer=FakeVectorizer(), generator=FakeGenerator())

    pipeline.ingest(
        [
            {"id": "doc-fox", "text": "The quick fox uses vector search."},
            {"id": "doc-dog", "text": "A lazy dog sleeps all day."},
        ]
    )
    assert pipeline.vector_store.count() == 2

    pipeline.delete("doc-fox")

    assert pipeline.vector_store.count() == 1
    assert len(pipeline.bm25_index) == 1
    assert "doc-fox" not in pipeline._chunks_by_parent


def test_pipeline_query_extracts_citations(tmp_path):
    class CitingGenerator(BaseGenerator):
        def generate(self, question, context):
            return "The fox searches with vectors [1]."

    settings = _settings(tmp_path, use_hybrid=True)
    pipeline = RAGPipeline(settings, vectorizer=FakeVectorizer(), generator=CitingGenerator())

    pipeline.ingest([{"id": "doc-fox", "text": "The quick fox uses vector search."}])
    result = pipeline.query("fox", top_k=1)

    assert result.cited_source_ids == [result.sources[0].document.id]
