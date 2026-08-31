"""Example: ingest every supported file (.txt, .md, .html, .pdf) under a directory.

Run with the repo root on PYTHONPATH, e.g.:
    python -m examples.ingest_directory ./some/docs/dir
Requires a .env (see .env.example) with a generation provider API key set,
and whichever embedding backend's dependency/key you configured.
"""
import sys

from dotenv import load_dotenv

from src.config.settings import load_settings
from src.ingestion.loaders import load_directory
from src.logging_config import configure_logging
from src.pipeline import build_pipeline

load_dotenv()


def main():
    configure_logging()
    if len(sys.argv) != 2:
        print("Usage: python -m examples.ingest_directory <directory>")
        raise SystemExit(1)

    directory = sys.argv[1]
    records = load_directory(directory)
    if not records:
        print(f"No supported files (.txt/.md/.html/.pdf) found under {directory}")
        return

    pipeline = build_pipeline(load_settings())
    pipeline.ingest(records)
    print(f"Ingested {len(records)} file(s) from {directory}")

    question = "Summarize what these documents are about."
    result = pipeline.query(question)
    print(f"\nQ: {question}\nA: {result.answer}")


if __name__ == "__main__":
    main()
