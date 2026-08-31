import os

from src.config.settings import (
    ChunkingConfig,
    GenerationConfig,
    RetrievalConfig,
    Settings,
    VectorStoreConfig,
    VectorizerConfig,
)
from src.pipeline import RAGPipeline
from tests.test_pipeline import FakeGenerator, FakeVectorizer


def _persistent_settings(tmp_path) -> Settings:
    return Settings(
        vectorizer=VectorizerConfig(backend="sentence_transformers"),
        vector_store=VectorStoreConfig(
            backend="chroma",
            collection_name="persistence-test",
            persist_directory=str(tmp_path),
        ),
        generation=GenerationConfig(provider="anthropic"),
        chunking=ChunkingConfig(chunk_size=50, chunk_overlap=5),
        retrieval=RetrievalConfig(top_k=2, use_hybrid=True, rrf_k=60),
    )


def test_bm25_corpus_survives_pipeline_restart(tmp_path):
    settings = _persistent_settings(tmp_path)

    pipeline_a = RAGPipeline(settings, vectorizer=FakeVectorizer(), generator=FakeGenerator())
    pipeline_a.ingest(
        [
            {"id": "doc-fox", "text": "The quick fox uses vector search over a database."},
            {"id": "doc-dog", "text": "A lazy dog sleeps in the python garden all day."},
        ]
    )
    assert len(pipeline_a.bm25_index) == 2
    assert os.path.exists(os.path.join(str(tmp_path), "bm25_corpus.json"))

    # Simulate a fresh process: brand-new pipeline instance, same persist dir.
    pipeline_b = RAGPipeline(settings, vectorizer=FakeVectorizer(), generator=FakeGenerator())

    assert len(pipeline_b.bm25_index) == 2
    assert set(pipeline_b._chunks_by_parent.keys()) == {"doc-fox", "doc-dog"}

    result = pipeline_b.query("fox vector database", top_k=1)
    assert "fox" in result.sources[0].document.text.lower()


def test_reingest_after_restart_still_dedups(tmp_path):
    settings = _persistent_settings(tmp_path)

    pipeline_a = RAGPipeline(settings, vectorizer=FakeVectorizer(), generator=FakeGenerator())
    pipeline_a.ingest([{"id": "doc-fox", "text": "The quick fox uses vector search."}])

    pipeline_b = RAGPipeline(settings, vectorizer=FakeVectorizer(), generator=FakeGenerator())
    pipeline_b.ingest([{"id": "doc-fox", "text": "A totally different sentence about python."}])

    assert pipeline_b.vector_store.count() == 1
    assert len(pipeline_b.bm25_index) == 1
    result = pipeline_b.query("python", top_k=1)
    assert "python" in result.sources[0].document.text.lower()
