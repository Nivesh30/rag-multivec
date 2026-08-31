import re
from typing import List

from src.middleware.normalizer import Document

# Splits on sentence-ending punctuation followed by whitespace, and on blank
# lines (paragraph breaks) - good enough for prose without pulling in a full
# sentence-segmentation dependency.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def chunk_document(document: Document, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    """Split a Document's text into overlapping chunks along sentence boundaries.

    Chunks are packed to at most `chunk_size` words without splitting a
    sentence, except when a single sentence itself exceeds `chunk_size` (then
    that sentence is word-split on its own). Consecutive chunks overlap by
    roughly `chunk_overlap` words' worth of trailing sentences, so retrieval
    doesn't lose context at a chunk boundary.

    Each chunk is its own Document, id'd as "<parent_id>::chunk<n>", carrying
    the parent id and chunk index in metadata so results can be traced back.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = _split_sentences(document.text)
    if not sentences:
        return []

    # Each entry: (sentence_text, word_count)
    sized = [(s, len(s.split())) for s in sentences]

    groups: List[List[str]] = []
    current: List[str] = []
    current_words = 0

    def flush():
        if current:
            groups.append(list(current))

    i = 0
    while i < len(sized):
        sentence, word_count = sized[i]

        if word_count > chunk_size:
            # A single sentence is too long on its own - flush what we have,
            # then hard-split this sentence by words as a last resort.
            flush()
            current, current_words = [], 0
            words = sentence.split()
            for start in range(0, len(words), chunk_size):
                groups.append([" ".join(words[start : start + chunk_size])])
            i += 1
            continue

        if current_words + word_count > chunk_size and current:
            flush()
            # Carry trailing sentences into the next chunk for overlap.
            overlap_sentences: List[str] = []
            overlap_words = 0
            for s in reversed(current):
                s_words = len(s.split())
                if overlap_words + s_words > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_words += s_words
            current = overlap_sentences
            current_words = overlap_words

        current.append(sentence)
        current_words += word_count
        i += 1

    flush()

    chunks: List[Document] = []
    for index, group in enumerate(groups):
        chunks.append(
            Document(
                id=f"{document.id}::chunk{index}",
                text=" ".join(group),
                metadata={**document.metadata, "parent_id": document.id, "chunk_index": index},
            )
        )
    return chunks


def chunk_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    result: List[Document] = []
    for doc in documents:
        result.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return result
