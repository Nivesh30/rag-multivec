import csv
import logging
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("rag_multivec.ingestion.loaders")


class _HTMLTextExtractor(HTMLParser):
    """Strips tags and collapses whitespace; drops <script>/<style> content entirely."""

    _SKIP_TAGS = {"script", "style"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self._parts).split())


def html_to_text(html: str) -> str:
    """Extract visible text from an HTML document (stdlib only, no BeautifulSoup dependency)."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.text()


def _default_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_text_file(path: str, id: Optional[str] = None) -> dict:
    """Load a plain-text file into a raw record shaped for RAGPipeline.ingest()."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return {"id": id or _default_id(path), "text": text, "source": path}


def load_markdown_file(path: str, id: Optional[str] = None) -> dict:
    """Markdown is ingested as raw text - the LLM handles markdown syntax fine as-is,
    and stripping it would lose structure (headings, lists) that's often useful context."""
    return load_text_file(path, id=id)


def load_html_file(path: str, id: Optional[str] = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    return {"id": id or _default_id(path), "text": html_to_text(html), "source": path}


def load_pdf_file(path: str, id: Optional[str] = None) -> dict:
    from pypdf import PdfReader

    reader = PdfReader(path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return {"id": id or _default_id(path), "text": text, "source": path, "page_count": len(reader.pages)}


def load_csv_file(path: str, text_column: str, id_column: Optional[str] = None) -> List[dict]:
    """Load a CSV file where one column holds the document text. Every other
    column (including id_column, if given) is carried through as metadata.
    Without id_column, rows are id'd as "<filename>-row<n>"."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or text_column not in reader.fieldnames:
            raise ValueError(f"CSV file '{path}' has no column '{text_column}'")

        for i, row in enumerate(reader):
            record_id = row[id_column] if id_column else f"{_default_id(path)}-row{i}"
            metadata = {k: v for k, v in row.items() if k != text_column}
            records.append({"id": record_id, "text": row[text_column], "source": path, **metadata})
    return records


_LOADER_BY_EXTENSION = {
    ".txt": load_text_file,
    ".md": load_markdown_file,
    ".markdown": load_markdown_file,
    ".html": load_html_file,
    ".htm": load_html_file,
    ".pdf": load_pdf_file,
}


def load_directory(path: str, recursive: bool = True) -> List[dict]:
    """Load every file under `path` with a registered extension
    (.txt, .md/.markdown, .html/.htm, .pdf) into raw records.

    CSV files aren't included here - load_csv_file() needs a `text_column`
    per file and produces multiple records per file, so call it directly
    for each CSV rather than folding it into a generic directory walk.
    """
    records: List[dict] = []
    pattern = "**/*" if recursive else "*"
    skipped = 0
    for file_path in sorted(Path(path).glob(pattern)):
        if not file_path.is_file():
            continue
        loader = _LOADER_BY_EXTENSION.get(file_path.suffix.lower())
        if loader is None:
            skipped += 1
            continue
        records.append(loader(str(file_path)))

    logger.info("load_directory(%s) -> %d record(s), %d file(s) skipped (unsupported extension)", path, len(records), skipped)
    return records
