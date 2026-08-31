import logging
from typing import Iterator, List, Optional

from src.errors import GenerationError
from src.generation.base import RAG_SYSTEM_PROMPT, BaseGenerator, build_context_block
from src.retrieval.retriever import ScoredDocument

DEFAULT_MODEL = "gpt-4o-mini"

logger = logging.getLogger("rag_multivec.generation.openai")


class OpenAIGenerator(BaseGenerator):
    """Answer generation via the OpenAI Chat Completions API.

    The OpenAI SDK already retries connection errors, 429, and 5xx with
    exponential backoff (`max_retries`, configurable here via
    GENERATION_MAX_RETRIES). Calls wrap the SDK exceptions so a failure that
    survives those retries surfaces as a clear GenerationError instead of a
    raw SDK exception leaking out of the pipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        max_retries: int = 2,
    ):
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key, max_retries=max_retries)

    def _messages(self, question: str, context: List[ScoredDocument]):
        context_block = build_context_block(context)
        user_message = f"Context:\n{context_block}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def generate(self, question: str, context: List[ScoredDocument]) -> str:
        import openai

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=self._messages(question, context),
            )
        except openai.AuthenticationError as exc:
            raise GenerationError("openai", "invalid API key", cause=exc) from exc
        except openai.RateLimitError as exc:
            logger.error("OpenAI rate limit exhausted retries: %s", exc)
            raise GenerationError("openai", "rate limited after retries", cause=exc) from exc
        except openai.APIConnectionError as exc:
            logger.error("OpenAI connection error exhausted retries: %s", exc)
            raise GenerationError("openai", "connection failed after retries", cause=exc) from exc
        except openai.APIStatusError as exc:
            logger.error("OpenAI API error (status %s): %s", exc.status_code, exc)
            raise GenerationError("openai", f"API error (status {exc.status_code})", cause=exc) from exc

        return response.choices[0].message.content or ""

    def stream(self, question: str, context: List[ScoredDocument]) -> Iterator[str]:
        import openai

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=self._messages(question, context),
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except openai.AuthenticationError as exc:
            raise GenerationError("openai", "invalid API key", cause=exc) from exc
        except openai.RateLimitError as exc:
            logger.error("OpenAI rate limit exhausted retries mid-stream: %s", exc)
            raise GenerationError("openai", "rate limited after retries", cause=exc) from exc
        except openai.APIConnectionError as exc:
            logger.error("OpenAI connection lost mid-stream: %s", exc)
            raise GenerationError("openai", "connection failed after retries", cause=exc) from exc
        except openai.APIStatusError as exc:
            logger.error("OpenAI API error mid-stream (status %s): %s", exc.status_code, exc)
            raise GenerationError("openai", f"API error (status {exc.status_code})", cause=exc) from exc
