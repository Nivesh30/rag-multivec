from typing import List, Optional

from src.generation.base import RAG_SYSTEM_PROMPT, BaseGenerator, build_context_block
from src.retrieval.retriever import ScoredDocument

DEFAULT_MODEL = "claude-opus-5"


class AnthropicGenerator(BaseGenerator):
    """Answer generation via the Anthropic Claude API."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL, max_tokens: int = 2048):
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def generate(self, question: str, context: List[ScoredDocument]) -> str:
        context_block = build_context_block(context)
        user_message = f"Context:\n{context_block}\n\nQuestion: {question}"

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=RAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
