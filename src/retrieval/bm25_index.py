import json
import logging
import os
import re
from typing import List, Tuple

from src.middleware.normalizer import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")

logger = logging.getLogger("rag_multivec.retrieval.bm25")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Sparse keyword index (Okapi BM25) for hybrid retrieval.

    rank_bm25 itself has no incremental-update or serialization support, so
    "persistence" here means saving/loading the underlying document corpus
    as JSON and rebuilding the BM25Okapi index from it - cheap relative to
    embedding, and mirrors the ChromaVectorStore.persist_directory model so
    the sparse side survives a process restart too.
    """

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

    def save(self, path: str) -> None:
        """Persist the current corpus to `path` as JSON. Overwrites any existing file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = [{"id": doc.id, "text": doc.text, "metadata": doc.metadata} for doc in self._documents]
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)  # atomic on POSIX - avoids a half-written file on crash
        logger.info("BM25 corpus saved to %s (%d document(s))", path, len(self._documents))

    def load(self, path: str) -> List[Document]:
        """Load a previously saved corpus from `path`, rebuild the index, and
        return the loaded documents (so a caller can rebuild any parallel
        bookkeeping, e.g. RAGPipeline's per-source-document chunk map).

        Returns an empty list if the file doesn't exist yet (e.g. first
        run) - a missing file is not an error.
        """
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        documents = [Document(id=item["id"], text=item["text"], metadata=item.get("metadata") or {}) for item in payload]
        self.build(documents)
        logger.info("BM25 corpus loaded from %s (%d document(s))", path, len(documents))
        return documents

    def __len__(self) -> int:
        return len(self._doc_ids)
