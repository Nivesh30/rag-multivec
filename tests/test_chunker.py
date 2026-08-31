from src.ingestion.chunker import chunk_document, _split_sentences
from src.middleware.normalizer import Document


def test_split_sentences_handles_punctuation_and_paragraphs():
    text = "First sentence. Second sentence! Third one?\n\nNew paragraph here."
    assert _split_sentences(text) == [
        "First sentence.",
        "Second sentence!",
        "Third one?",
        "New paragraph here.",
    ]


def test_chunk_document_never_splits_a_sentence_across_chunks():
    sentences = [f"Sentence number {i} has five words." for i in range(10)]
    doc = Document(id="doc1", text=" ".join(sentences))

    chunks = chunk_document(doc, chunk_size=20, chunk_overlap=5)

    assert len(chunks) > 1
    for chunk in chunks:
        for piece in _split_sentences(chunk.text):
            assert piece in sentences


def test_chunk_document_consecutive_chunks_overlap():
    sentences = [f"Sentence number {i} has five words." for i in range(10)]
    doc = Document(id="doc1", text=" ".join(sentences))

    chunks = chunk_document(doc, chunk_size=20, chunk_overlap=8)

    assert len(chunks) >= 2
    first_sentences = set(_split_sentences(chunks[0].text))
    second_sentences = set(_split_sentences(chunks[1].text))
    assert first_sentences & second_sentences, "expected trailing sentence(s) to carry into next chunk"


def test_chunk_document_ids_and_metadata():
    doc = Document(id="doc1", text="Only one short sentence here.", metadata={"source": "x"})
    chunks = chunk_document(doc, chunk_size=50, chunk_overlap=5)

    assert len(chunks) == 1
    assert chunks[0].id == "doc1::chunk0"
    assert chunks[0].metadata["parent_id"] == "doc1"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["source"] == "x"


def test_chunk_document_oversized_single_sentence_is_word_split():
    huge_sentence = " ".join(f"w{i}" for i in range(30)) + "."
    doc = Document(id="doc1", text=huge_sentence)

    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)

    assert len(chunks) == 3
    assert sum(len(c.text.split()) for c in chunks) == 30


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
