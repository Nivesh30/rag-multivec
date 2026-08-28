from abc import ABC, abstractmethod
from typing import List

from src.retrieval.retriever import ScoredDocument

RAG_SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the user's question using only "
    "the provided context. If the context does not contain the answer, say so plainly "
    "instead of guessing."
)


def build_context_block(scored_documents: List[ScoredDocument]) -> str:
    parts = []
    for i, scored in enumerate(scored_documents, start=1):
        parts.append(f"[{i}] {scored.document.text}")
    return "\n\n".join(parts)


class BaseGenerator(ABC):
    """Pluggable answer-generation backend: any LLM provider can implement this."""

    @abstractmethod
    def generate(self, question: str, context: List[ScoredDocument]) -> str:
        ...
