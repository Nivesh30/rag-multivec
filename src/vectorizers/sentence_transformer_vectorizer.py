from typing import List, Optional

from src.vectorizers.base import BaseVectorizer

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerVectorizer(BaseVectorizer):
    """Local dense embeddings via sentence-transformers. No API key required."""

    def __init__(self, model: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model or DEFAULT_MODEL)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
