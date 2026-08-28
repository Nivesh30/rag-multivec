from typing import List, Optional

from src.vectorizers.base import BaseVectorizer

DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIVectorizer(BaseVectorizer):
    """Dense embeddings via the OpenAI embeddings API."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        from openai import OpenAI

        if not api_key:
            raise ValueError("OpenAI API key is required (set OPENAI_API_KEY)")
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
