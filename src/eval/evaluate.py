from dataclasses import dataclass
from typing import List

from src.retrieval.retriever import Retriever


@dataclass
class EvalCase:
    question: str
    expected_document_id: str
    """Either a chunk id (e.g. 'doc-1::chunk0') or a source document id
    (e.g. 'doc-1') - matched against both the retrieved chunk's id and its
    parent_id metadata, so eval sets can be written at whichever granularity
    is convenient."""


@dataclass
class CaseResult:
    case: EvalCase
    hit: bool
    rank: int  # 1-based rank of the first matching hit, or 0 if not found
    retrieved_ids: List[str]


@dataclass
class EvalReport:
    recall_at_k: float
    mrr: float
    results: List[CaseResult]


def _matches(document_id: str, metadata_parent_id, expected_id: str) -> bool:
    return document_id == expected_id or metadata_parent_id == expected_id


def evaluate_retrieval(retriever: Retriever, eval_set: List[EvalCase], top_k: int = 5) -> EvalReport:
    """Run each eval case's question through the retriever and score it.

    - recall@k: fraction of cases where the expected document appears
      anywhere in the top-k retrieved results.
    - MRR (mean reciprocal rank): average of 1/rank of the first matching
      hit across cases (0 for a case with no hit at all).
    """
    if not eval_set:
        return EvalReport(recall_at_k=0.0, mrr=0.0, results=[])

    results: List[CaseResult] = []
    for case in eval_set:
        scored = retriever.retrieve(case.question, top_k=top_k)
        retrieved_ids = [s.document.id for s in scored]

        rank = 0
        for i, s in enumerate(scored, start=1):
            if _matches(s.document.id, s.document.metadata.get("parent_id"), case.expected_document_id):
                rank = i
                break

        results.append(CaseResult(case=case, hit=rank > 0, rank=rank, retrieved_ids=retrieved_ids))

    recall_at_k = sum(1 for r in results if r.hit) / len(results)
    mrr = sum(1.0 / r.rank for r in results if r.hit) / len(results)
    return EvalReport(recall_at_k=recall_at_k, mrr=mrr, results=results)
