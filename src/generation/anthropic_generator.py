import logging
from typing import Iterator, List, Optional

from src.errors import GenerationError
from src.generation.base import RAG_SYSTEM_PROMPT, BaseGenerator, build_context_block
from src.retrieval.retriever import ScoredDocument

DEFAULT_MODEL = "claude-opus-5"

logger = logging.getLogger("rag_multivec.generation.anthropic")


class AnthropicGenerator(BaseGenerator):
    """Answer generation via the Anthropic Claude API.

    The Anthropic SDK already retries connection errors, 429, and 5xx with
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
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        kwargs = {"max_retries": max_retries}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**kwargs)

    def _user_message(self, question: str, context: List[ScoredDocument]) -> str:
        context_block = build_context_block(context)
        return f"Context:\n{context_block}\n\nQuestion: {question}"

    def generate(self, question: str, context: List[ScoredDocument]) -> str:
        import anthropic

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=RAG_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": self._user_message(question, context)}],
            )
        except anthropic.AuthenticationError as exc:
            raise GenerationError("anthropic", "invalid API key", cause=exc) from exc
        except anthropic.RateLimitError as exc:
            logger.error("Anthropic rate limit exhausted retries: %s", exc)
            raise GenerationError("anthropic", "rate limited after retries", cause=exc) from exc
        except anthropic.APIConnectionError as exc:
            logger.error("Anthropic connection error exhausted retries: %s", exc)
            raise GenerationError("anthropic", "connection failed after retries", cause=exc) from exc
        except anthropic.APIStatusError as exc:
            logger.error("Anthropic API error (status %s): %s", exc.status_code, exc)
            raise GenerationError("anthropic", f"API error (status {exc.status_code})", cause=exc) from exc

        if response.stop_reason == "refusal":
            logger.warning("Anthropic refused to answer: %s", getattr(response, "stop_details", None))
            raise GenerationError("anthropic", "model declined to answer (refusal)")

        return "".join(block.text for block in response.content if block.type == "text")

    def stream(self, question: str, context: List[ScoredDocument]) -> Iterator[str]:
        import anthropic

        try:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=RAG_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": self._user_message(question, context)}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
                final_message = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise GenerationError("anthropic", "invalid API key", cause=exc) from exc
        except anthropic.RateLimitError as exc:
            logger.error("Anthropic rate limit exhausted retries mid-stream: %s", exc)
            raise GenerationError("anthropic", "rate limited after retries", cause=exc) from exc
        except anthropic.APIConnectionError as exc:
            logger.error("Anthropic connection lost mid-stream: %s", exc)
            raise GenerationError("anthropic", "connection failed after retries", cause=exc) from exc
        except anthropic.APIStatusError as exc:
            logger.error("Anthropic API error mid-stream (status %s): %s", exc.status_code, exc)
            raise GenerationError("anthropic", f"API error (status {exc.status_code})", cause=exc) from exc

        if final_message.stop_reason == "refusal":
            logger.warning("Anthropic refused to answer: %s", getattr(final_message, "stop_details", None))
            raise GenerationError("anthropic", "model declined to answer (refusal)")
