import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from src.middleware.normalizer import Document
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.vector_store import ChromaVectorStore
from src.vectorizers.base import BaseVectorizer

logger = logging.getLogger("rag_multivec.retrieval")


@dataclass
class ScoredDocument:
    document: Document
    score: float


class Retriever:
    """Dense (vector store) retrieval, optionally fused with sparse BM25 (hybrid search)."""

    def __init__(
        self,
        vectorizer: BaseVectorizer,
        vector_store: ChromaVectorStore,
        bm25_index: Optional[BM25Index] = None,
        use_hybrid: bool = True,
        rrf_k: int = 60,
    ):
        self._vectorizer = vectorizer
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._use_hybrid = use_hybrid and bm25_index is not None
        self._rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> List[ScoredDocument]:
        started = time.monotonic()
        results = self._retrieve(query, top_k)
        elapsed_ms = (time.monotonic() - started) * 1000
        logger.info(
            "retrieve(%s, top_k=%d) -> %d hit(s) in %.1fms",
            "hybrid" if self._use_hybrid else "dense",
            top_k,
            len(results),
            elapsed_ms,
        )
        return results

    def _retrieve(self, query: str, top_k: int) -> List[ScoredDocument]:
        fetch_k = max(top_k * 4, top_k)
        query_embedding = self._vectorizer.embed([query])[0]
        dense_hits = self._vector_store.query(query_embedding, top_k=fetch_k)

        if not self._use_hybrid:
            top_hits = dense_hits[:top_k]
            docs_by_id = {doc.id: doc for doc in self._vector_store.get_by_ids([h[0] for h in top_hits])}
            return [
                ScoredDocument(document=docs_by_id[doc_id], score=score)
                for doc_id, score in top_hits
                if doc_id in docs_by_id
            ]

        sparse_hits = self._bm25_index.query(query, top_k=fetch_k)
        fused = reciprocal_rank_fusion(
            [[doc_id for doc_id, _ in dense_hits], [doc_id for doc_id, _ in sparse_hits]],
            k=self._rrf_k,
        )[:top_k]

        docs_by_id = {doc.id: doc for doc in self._vector_store.get_by_ids([doc_id for doc_id, _ in fused])}
        return [
            ScoredDocument(document=docs_by_id[doc_id], score=score)
            for doc_id, score in fused
            if doc_id in docs_by_id
        ]
