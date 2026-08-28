from typing import List, Optional

from src.vectorizers.base import BaseVectorizer

DEFAULT_MODEL = "voyage-3.5"


class VoyageVectorizer(BaseVectorizer):
    """Dense embeddings via the Voyage AI API (Anthropic's recommended embedding partner)."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        import voyageai

        if not api_key:
            raise ValueError("Voyage API key is required (set VOYAGE_API_KEY)")
        self.model = model
        self._client = voyageai.Client(api_key=api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result = self._client.embed(texts, model=self.model, input_type="document")
        return result.embeddings
