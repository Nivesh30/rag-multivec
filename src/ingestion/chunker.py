from typing import List

from src.middleware.normalizer import Document


def chunk_document(document: Document, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    """Split a Document's text into overlapping, word-bounded chunks.

    Each chunk is its own Document, id'd as "<parent_id>::chunk<n>", carrying
    the parent id and chunk index in metadata so results can be traced back.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = document.text.split(" ")
    if not words or not document.text:
        return []

    chunks: List[Document] = []
    start = 0
    step = chunk_size - chunk_overlap
    index = 0
    while start < len(words):
        piece = " ".join(words[start : start + chunk_size])
        chunks.append(
            Document(
                id=f"{document.id}::chunk{index}",
                text=piece,
                metadata={**document.metadata, "parent_id": document.id, "chunk_index": index},
            )
        )
        index += 1
        start += step

    return chunks


def chunk_documents(documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    result: List[Document] = []
    for doc in documents:
        result.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return result
