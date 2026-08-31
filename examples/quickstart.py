"""Minimal end-to-end example: ingest a few documents, then ask a question.

Run with the repo root on PYTHONPATH, e.g.:
    python -m examples.quickstart
Requires a .env (see .env.example) with a generation provider API key set,
and whichever embedding backend's dependency/key you configured.
"""
from dotenv import load_dotenv

from src.config.settings import load_settings
from src.logging_config import configure_logging
from src.pipeline import build_pipeline

load_dotenv()


def main():
    configure_logging()
    pipeline = build_pipeline(load_settings())

    pipeline.ingest(
        [
            {
                "id": "doc-1",
                "text": (
                    "RAG (Retrieval-Augmented Generation) combines a retriever, which "
                    "fetches relevant passages from a knowledge base, with a generator "
                    "model that composes an answer grounded in those passages."
                ),
                "source": "intro.md",
            },
            {
                "id": "doc-2",
                "text": (
                    "Hybrid retrieval fuses dense vector similarity search with sparse "
                    "keyword search (e.g. BM25) using reciprocal rank fusion, which "
                    "often outperforms either method alone."
                ),
                "source": "retrieval.md",
            },
        ]
    )

    result = pipeline.query("How does hybrid retrieval work?")
    print("Answer:", result.answer)
    print("\nSources:")
    for scored in result.sources:
        print(f"  ({scored.score:.3f}) {scored.document.id}: {scored.document.text[:80]}...")


if __name__ == "__main__":
    main()
