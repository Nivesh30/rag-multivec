from src.ingestion.chunker import chunk_document
from src.middleware.normalizer import Document


def test_chunk_document_respects_size_and_overlap():
    words = [f"word{i}" for i in range(25)]
    doc = Document(id="doc1", text=" ".join(words))

    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)

    assert [c.id for c in chunks] == ["doc1::chunk0", "doc1::chunk1", "doc1::chunk2", "doc1::chunk3"]
    assert chunks[0].text.split(" ") == words[0:10]
    assert chunks[1].text.split(" ") == words[8:18]
    assert chunks[2].text.split(" ") == words[16:25]
    assert chunks[3].text.split(" ") == words[24:25]
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["parent_id"] == "doc1"
        assert chunk.metadata["chunk_index"] == i


def test_chunk_document_empty_text_returns_no_chunks():
    doc = Document(id="empty", text="")
    assert chunk_document(doc) == []


def test_chunk_document_rejects_overlap_gte_size():
    doc = Document(id="doc1", text="a b c")
    try:
        chunk_document(doc, chunk_size=5, chunk_overlap=5)
        assert False, "expected ValueError"
    except ValueError:
        pass
