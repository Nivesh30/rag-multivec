from src.middleware.normalizer import Document
from src.retrieval.bm25_index import BM25Index


def test_bm25_index_ranks_matching_document_first():
    docs = [
        Document(id="1", text="the quick brown fox jumps over the lazy dog"),
        Document(id="2", text="vector databases store embeddings for similarity search"),
        Document(id="3", text="foxes are wild canids found across the globe"),
    ]
    index = BM25Index()
    index.build(docs)

    results = index.query("fox", top_k=3)
    result_ids = [doc_id for doc_id, _ in results]

    assert result_ids[0] in ("1", "3")
    assert "2" not in result_ids


def test_bm25_index_empty_corpus_returns_nothing():
    index = BM25Index()
    index.build([])
    assert index.query("anything") == []
