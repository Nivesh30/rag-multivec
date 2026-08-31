"""Example: measure retrieval quality (recall@k, MRR) against a small labeled set.

Run with the repo root on PYTHONPATH, e.g.:
    python -m examples.eval_retrieval
"""
from dotenv import load_dotenv

from src.config.settings import load_settings
from src.eval.evaluate import EvalCase, evaluate_retrieval
from src.pipeline import build_pipeline

load_dotenv()


def main():
    pipeline = build_pipeline(load_settings())

    pipeline.ingest(
        [
            {"id": "doc-rag", "text": "RAG combines a retriever with a generator model to answer questions grounded in retrieved passages."},
            {"id": "doc-hybrid", "text": "Hybrid retrieval fuses dense vector search with sparse BM25 keyword search using reciprocal rank fusion."},
            {"id": "doc-chunking", "text": "Chunking splits long documents into smaller overlapping pieces before embedding."},
        ]
    )

    eval_set = [
        EvalCase(question="What is RAG?", expected_document_id="doc-rag"),
        EvalCase(question="How does hybrid search combine dense and sparse retrieval?", expected_document_id="doc-hybrid"),
        EvalCase(question="Why split documents into overlapping pieces?", expected_document_id="doc-chunking"),
    ]

    report = evaluate_retrieval(pipeline.retriever, eval_set, top_k=3)
    print(f"recall@3 = {report.recall_at_k:.2f}")
    print(f"MRR      = {report.mrr:.2f}")
    for r in report.results:
        status = "HIT" if r.hit else "MISS"
        print(f"  [{status}] rank={r.rank} '{r.case.question}' -> {r.retrieved_ids}")


if __name__ == "__main__":
    main()
