import re
from typing import List, Tuple

from src.middleware.normalizer import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory sparse keyword index (Okapi BM25) for hybrid retrieval."""

    def __init__(self):
        self._doc_ids: List[str] = []
        self._documents: List[Document] = []
        self._bm25 = None

    def build(self, documents: List[Document]) -> None:
        from rank_bm25 import BM25Okapi

        self._documents = list(documents)
        self._doc_ids = [doc.id for doc in self._documents]
        tokenized_corpus = [_tokenize(doc.text) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def query(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked = sorted(zip(self._doc_ids, scores), key=lambda pair: pair[1], reverse=True)
        return [(doc_id, score) for doc_id, score in ranked[:top_k] if score > 0]

    def __len__(self) -> int:
        return len(self._doc_ids)
