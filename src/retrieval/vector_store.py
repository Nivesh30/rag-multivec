from typing import List, Optional, Tuple

from src.middleware.normalizer import Document


class ChromaVectorStore:
    """Thin wrapper around a Chroma collection for dense similarity search."""

    def __init__(self, collection_name: str = "documents", persist_directory: Optional[str] = None):
        import chromadb

        self._client = (
            chromadb.PersistentClient(path=persist_directory)
            if persist_directory
            else chromadb.EphemeralClient()
        )
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        if not documents:
            return
        self._collection.upsert(
            ids=[doc.id for doc in documents],
            embeddings=embeddings,
            documents=[doc.text for doc in documents],
            # Chroma rejects an empty metadata dict outright, so document a
            # placeholder for callers that don't supply any metadata.
            metadatas=[doc.metadata or {"_no_metadata": True} for doc in documents],
        )

    def query(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
        ids = result["ids"][0]
        distances = result["distances"][0]
        # Chroma returns a distance (lower = more similar); convert to a similarity score.
        return [(doc_id, 1.0 / (1.0 + dist)) for doc_id, dist in zip(ids, distances)]

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        if not ids:
            return []
        result = self._collection.get(ids=ids)
        documents = []
        for doc_id, text, metadata in zip(result["ids"], result["documents"], result["metadatas"]):
            documents.append(Document(id=doc_id, text=text, metadata=metadata or {}))
        return documents

    def delete_by_parent_id(self, parent_id: str) -> None:
        """Remove every chunk previously indexed for a given source document id."""
        self._collection.delete(where={"parent_id": parent_id})

    def count(self) -> int:
        return self._collection.count()
