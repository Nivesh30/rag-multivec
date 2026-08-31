from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.loaders import (
    html_to_text,
    load_csv_file,
    load_directory,
    load_html_file,
    load_markdown_file,
    load_pdf_file,
    load_text_file,
)


def test_html_to_text_strips_tags_and_collapses_whitespace():
    html = "<html><body><h1>Title</h1>\n<p>Some   text with <b>bold</b> words.</p></body></html>"
    assert html_to_text(html) == "Title Some text with bold words."


def test_html_to_text_drops_script_and_style_content():
    html = "<p>Visible</p><script>var x = 1;</script><style>.a{color:red}</style><p>Also visible</p>"
    assert html_to_text(html) == "Visible Also visible"


def test_load_text_file_uses_filename_as_default_id(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Hello world.")

    record = load_text_file(str(path))

    assert record == {"id": "notes", "text": "Hello world.", "source": str(path)}


def test_load_text_file_explicit_id(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Hello world.")

    record = load_text_file(str(path), id="custom-id")

    assert record["id"] == "custom-id"


def test_load_markdown_file_is_raw_text(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# Heading\n\nSome **bold** text.")

    record = load_markdown_file(str(path))

    assert record["text"] == "# Heading\n\nSome **bold** text."
    assert record["id"] == "doc"


def test_load_html_file_extracts_text(tmp_path):
    path = tmp_path / "page.html"
    path.write_text("<html><body><p>Hello <em>world</em>.</p></body></html>")

    record = load_html_file(str(path))

    assert record["text"] == "Hello world ."
    assert record["id"] == "page"


def test_load_csv_file_with_id_column(tmp_path):
    path = tmp_path / "faq.csv"
    path.write_text("id,question,answer\nq1,What is RAG?,Retrieval-augmented generation.\n")

    records = load_csv_file(str(path), text_column="answer", id_column="id")

    assert records == [
        {
            "id": "q1",
            "text": "Retrieval-augmented generation.",
            "source": str(path),
            "question": "What is RAG?",
        }
    ]


def test_load_csv_file_without_id_column_auto_ids(tmp_path):
    path = tmp_path / "faq.csv"
    path.write_text("question,answer\nWhat is RAG?,Retrieval-augmented generation.\n")

    records = load_csv_file(str(path), text_column="answer")

    assert records[0]["id"] == "faq-row0"


def test_load_csv_file_missing_column_raises(tmp_path):
    path = tmp_path / "faq.csv"
    path.write_text("question,answer\nWhat is RAG?,Retrieval-augmented generation.\n")

    with pytest.raises(ValueError, match="no column"):
        load_csv_file(str(path), text_column="does_not_exist")


def test_load_pdf_file_joins_page_text():
    fake_page_1 = MagicMock()
    fake_page_1.extract_text.return_value = "Page one text."
    fake_page_2 = MagicMock()
    fake_page_2.extract_text.return_value = "Page two text."

    fake_reader = MagicMock()
    fake_reader.pages = [fake_page_1, fake_page_2]

    with patch("pypdf.PdfReader", return_value=fake_reader):
        record = load_pdf_file("/fake/path/doc.pdf")

    assert record["text"] == "Page one text.\n\nPage two text."
    assert record["id"] == "doc"
    assert record["page_count"] == 2


def test_load_pdf_file_handles_page_with_no_extractable_text():
    fake_page = MagicMock()
    fake_page.extract_text.return_value = None
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    with patch("pypdf.PdfReader", return_value=fake_reader):
        record = load_pdf_file("/fake/path/blank.pdf")

    assert record["text"] == ""


def test_load_directory_loads_supported_extensions_only(tmp_path):
    (tmp_path / "a.txt").write_text("Text file.")
    (tmp_path / "b.md").write_text("# Markdown file.")
    (tmp_path / "c.html").write_text("<p>HTML file.</p>")
    (tmp_path / "d.unsupported").write_text("should be skipped")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "e.txt").write_text("Nested text file.")

    records = load_directory(str(tmp_path))

    ids = sorted(r["id"] for r in records)
    assert ids == ["a", "b", "c", "e"]


def test_load_directory_non_recursive(tmp_path):
    (tmp_path / "a.txt").write_text("Top level.")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("Nested.")

    records = load_directory(str(tmp_path), recursive=False)

    assert [r["id"] for r in records] == ["a"]
