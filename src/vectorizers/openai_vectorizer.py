import logging
from typing import List, Optional

from src.errors import EmbeddingError
from src.vectorizers.base import BaseVectorizer

DEFAULT_MODEL = "text-embedding-3-small"

logger = logging.getLogger("rag_multivec.vectorizers.openai")


class OpenAIVectorizer(BaseVectorizer):
    """Dense embeddings via the OpenAI embeddings API.

    The OpenAI SDK already retries connection errors, 429, and 5xx with
    exponential backoff (`max_retries`). A failure that survives those
    retries surfaces as a clear EmbeddingError rather than a raw SDK
    exception.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 2,
    ):
        from openai import OpenAI

        if not api_key:
            raise ValueError("OpenAI API key is required (set OPENAI_API_KEY)")
        self.model = model
        self._client = OpenAI(api_key=api_key, max_retries=max_retries)

    def embed(self, texts: List[str]) -> List[List[float]]:
        import openai

        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=self.model, input=texts)
        except openai.AuthenticationError as exc:
            raise EmbeddingError("openai", "invalid API key", cause=exc) from exc
        except openai.RateLimitError as exc:
            logger.error("OpenAI rate limit exhausted retries: %s", exc)
            raise EmbeddingError("openai", "rate limited after retries", cause=exc) from exc
        except openai.APIConnectionError as exc:
            logger.error("OpenAI connection error exhausted retries: %s", exc)
            raise EmbeddingError("openai", "connection failed after retries", cause=exc) from exc
        except openai.APIStatusError as exc:
            logger.error("OpenAI API error (status %s): %s", exc.status_code, exc)
            raise EmbeddingError("openai", f"API error (status {exc.status_code})", cause=exc) from exc

        return [item.embedding for item in response.data]
