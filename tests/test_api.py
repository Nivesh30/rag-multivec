import hashlib

import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.config.settings import (
    ChunkingConfig,
    GenerationConfig,
    RetrievalConfig,
    Settings,
    VectorStoreConfig,
    VectorizerConfig,
)
from src.errors import GenerationError
from src.generation.base import BaseGenerator
from src.pipeline import RAGPipeline
from tests.test_pipeline import FakeGenerator, FakeVectorizer


def _settings(tmp_path) -> Settings:
    return Settings(
        vectorizer=VectorizerConfig(backend="sentence_transformers"),
        vector_store=VectorStoreConfig(
            backend="chroma",
            collection_name=f"api-test-{hashlib.md5(str(tmp_path).encode()).hexdigest()[:8]}",
            persist_directory=None,
        ),
        generation=GenerationConfig(provider="anthropic"),
        chunking=ChunkingConfig(chunk_size=50, chunk_overlap=5),
        retrieval=RetrievalConfig(top_k=2, use_hybrid=True, rrf_k=60),
    )


@pytest.fixture
def client(tmp_path):
    pipeline = RAGPipeline(_settings(tmp_path), vectorizer=FakeVectorizer(), generator=FakeGenerator())
    app = create_app(pipeline=pipeline)
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_and_query(client):
    response = client.post(
        "/ingest",
        json={"records": [{"id": "doc-fox", "text": "The quick fox uses vector search over a database."}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chunks_indexed"] == 1
    assert body["chunk_ids"] == ["doc-fox::chunk0"]

    response = client.post("/query", json={"question": "fox vector database", "top_k": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "fox vector database"
    assert body["answer"].startswith("Answer to")
    assert len(body["sources"]) == 1
    assert "fox" in body["sources"][0]["text"].lower()


def test_query_stream(client):
    client.post("/ingest", json={"records": [{"id": "doc-fox", "text": "The quick fox uses vector search."}]})

    with client.stream("POST", "/query/stream", json={"question": "fox", "top_k": 1}) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())

    assert text.startswith("Answer to")


def test_delete_document(client):
    client.post("/ingest", json={"records": [{"id": "doc-fox", "text": "The quick fox uses vector search."}]})

    response = client.delete("/documents/doc-fox")
    assert response.status_code == 200
    assert response.json() == {"deleted": "doc-fox"}


def test_ingest_missing_id_returns_400(client):
    response = client.post("/ingest", json={"records": [{"text": "no id here"}]})
    assert response.status_code == 400


def test_query_generation_error_returns_502(tmp_path):
    class FailingGenerator(BaseGenerator):
        def generate(self, question, context):
            raise GenerationError("fake", "boom")

    pipeline = RAGPipeline(_settings(tmp_path), vectorizer=FakeVectorizer(), generator=FailingGenerator())
    pipeline.ingest([{"id": "doc-fox", "text": "The quick fox uses vector search."}])
    app = create_app(pipeline=pipeline)

    with TestClient(app) as client:
        response = client.post("/query", json={"question": "fox"})

    assert response.status_code == 502
