from src.eval.evaluate import EvalCase, evaluate_retrieval
from src.middleware.normalizer import Document
from src.retrieval.bm25_index import BM25Index
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import ChromaVectorStore
from tests.test_pipeline import FakeVectorizer


def _build_retriever(collection_name: str) -> Retriever:
    docs = [
        Document(id="doc-fox", text="The quick fox uses vector search over a database."),
        Document(id="doc-dog", text="A lazy dog sleeps in the python garden all day."),
    ]
    vectorizer = FakeVectorizer()
    store = ChromaVectorStore(collection_name=collection_name)
    store.add_documents(docs, vectorizer.embed([d.text for d in docs]))
    bm25 = BM25Index()
    bm25.build(docs)
    return Retriever(vectorizer=vectorizer, vector_store=store, bm25_index=bm25, use_hybrid=True)


def test_evaluate_retrieval_reports_recall_and_mrr():
    retriever = _build_retriever("eval-test-1")
    eval_set = [
        EvalCase(question="fox vector database", expected_document_id="doc-fox"),
        EvalCase(question="python dog", expected_document_id="doc-dog"),
    ]

    report = evaluate_retrieval(retriever, eval_set, top_k=2)

    assert report.recall_at_k == 1.0
    assert report.mrr > 0
    assert all(r.hit for r in report.results)


def test_evaluate_retrieval_handles_miss():
    retriever = _build_retriever("eval-test-2")
    eval_set = [EvalCase(question="fox vector database", expected_document_id="doc-nonexistent")]

    report = evaluate_retrieval(retriever, eval_set, top_k=2)

    assert report.recall_at_k == 0.0
    assert report.mrr == 0.0
    assert report.results[0].rank == 0


def test_evaluate_retrieval_empty_set():
    retriever = _build_retriever("eval-test-3")
    report = evaluate_retrieval(retriever, [], top_k=2)
    assert report.recall_at_k == 0.0
    assert report.results == []
