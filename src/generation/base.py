import re
from abc import ABC, abstractmethod
from typing import Iterator, List

from src.retrieval.retriever import ScoredDocument

RAG_SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the user's question using only "
    "the provided context. If the context does not contain the answer, say so plainly "
    "instead of guessing. The context is numbered [1], [2], etc. - cite the sources you "
    "actually relied on inline, e.g. 'Paris is the capital of France [1].' Cite every "
    "source you use at least once; do not cite sources you didn't use."
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


def build_context_block(scored_documents: List[ScoredDocument]) -> str:
    parts = []
    for i, scored in enumerate(scored_documents, start=1):
        parts.append(f"[{i}] {scored.document.text}")
    return "\n\n".join(parts)


def extract_cited_source_ids(answer_text: str, context: List[ScoredDocument]) -> List[str]:
    """Map [n] citation markers in generated text back to source document ids.

    Returns document ids in first-cited order, deduplicated. Out-of-range
    markers (a hallucinated [7] when there were only 3 sources) are ignored.
    """
    cited_ids: List[str] = []
    seen = set()
    for match in _CITATION_RE.finditer(answer_text):
        index = int(match.group(1)) - 1
        if 0 <= index < len(context):
            doc_id = context[index].document.id
            if doc_id not in seen:
                seen.add(doc_id)
                cited_ids.append(doc_id)
    return cited_ids


class BaseGenerator(ABC):
    """Pluggable answer-generation backend: any LLM provider can implement this."""

    @abstractmethod
    def generate(self, question: str, context: List[ScoredDocument]) -> str:
        ...

    def stream(self, question: str, context: List[ScoredDocument]) -> Iterator[str]:
        """Stream the answer as it's generated, one text chunk at a time.

        Default fallback: call generate() and yield its result as a single
        chunk. Providers that support real token streaming (Anthropic,
        OpenAI) override this; a custom BaseGenerator only needs to
        implement generate() to remain a valid, if non-streaming, backend.
        """
        yield self.generate(question, context)
