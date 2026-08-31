import os

from src.middleware.normalizer import Document
from src.retrieval.bm25_index import BM25Index


def test_bm25_save_and_load_round_trip(tmp_path):
    docs = [
        Document(id="doc-1::chunk0", text="The quick fox jumps over the lazy dog.", metadata={"parent_id": "doc-1"}),
        Document(id="doc-2::chunk0", text="Vector search over a database.", metadata={"parent_id": "doc-2"}),
        Document(id="doc-3::chunk0", text="Foxes are wild canids found worldwide.", metadata={"parent_id": "doc-3"}),
    ]
    index = BM25Index()
    index.build(docs)
    original_results = index.query("fox", top_k=3)
    assert original_results, "sanity check: query should match before persisting at all"

    path = str(tmp_path / "bm25_corpus.json")
    index.save(path)
    assert os.path.exists(path)

    reloaded = BM25Index()
    loaded_docs = reloaded.load(path)

    assert [d.id for d in loaded_docs] == [d.id for d in docs]
    assert len(reloaded) == 3
    # Save/load should be lossless: identical query results before and after.
    assert reloaded.query("fox", top_k=3) == original_results


def test_bm25_load_missing_file_returns_empty(tmp_path):
    index = BM25Index()
    loaded = index.load(str(tmp_path / "does-not-exist.json"))
    assert loaded == []
    assert len(index) == 0


def test_bm25_save_creates_parent_directories(tmp_path):
    index = BM25Index()
    index.build([Document(id="d1", text="hello world")])

    nested_path = str(tmp_path / "nested" / "dir" / "bm25_corpus.json")
    index.save(nested_path)

    assert os.path.exists(nested_path)
