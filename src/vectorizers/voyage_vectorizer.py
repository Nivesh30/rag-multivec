import logging
from typing import List, Optional

from src.errors import EmbeddingError, is_transient_by_signature
from src.vectorizers.base import BaseVectorizer

DEFAULT_MODEL = "voyage-3.5"

logger = logging.getLogger("rag_multivec.vectorizers.voyage")


class VoyageVectorizer(BaseVectorizer):
    """Dense embeddings via the Voyage AI API (Anthropic's recommended embedding partner).

    The `voyageai` client doesn't expose a documented typed-exception
    hierarchy here, so failures are classified with a duck-typed transient
    check (rate-limit/connection/5xx-shaped) purely for logging, and always
    surfaced as a clear EmbeddingError.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        import voyageai

        if not api_key:
            raise ValueError("Voyage API key is required (set VOYAGE_API_KEY)")
        self.model = model
        self._client = voyageai.Client(api_key=api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            result = self._client.embed(texts, model=self.model, input_type="document")
        except Exception as exc:
            if is_transient_by_signature(exc):
                logger.error("Voyage embedding call failed (transient-looking): %s", exc)
            else:
                logger.error("Voyage embedding call failed: %s", exc)
            raise EmbeddingError("voyage", str(exc), cause=exc) from exc

        return result.embeddings
