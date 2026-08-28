from typing import List, Optional

from src.generation.base import RAG_SYSTEM_PROMPT, BaseGenerator, build_context_block
from src.retrieval.retriever import ScoredDocument

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIGenerator(BaseGenerator):
    """Answer generation via the OpenAI Chat Completions API."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL, max_tokens: int = 2048):
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key)

    def generate(self, question: str, context: List[ScoredDocument]) -> str:
        context_block = build_context_block(context)
        user_message = f"Context:\n{context_block}\n\nQuestion: {question}"

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""
