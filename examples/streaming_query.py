"""Example: stream an answer token-by-token instead of waiting for the full response.

Run with the repo root on PYTHONPATH, e.g.:
    python -m examples.streaming_query
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
        ]
    )

    streaming_answer = pipeline.query_stream("What is RAG?")

    # Sources are already known - retrieval runs before generation starts.
    print("Sources:")
    for scored in streaming_answer.sources:
        print(f"  ({scored.score:.3f}) {scored.document.id}")

    print("\nAnswer: ", end="", flush=True)
    for chunk in streaming_answer:
        print(chunk, end="", flush=True)
    print()

    # .answer and .cited_source_ids reflect the fully streamed text once consumed.
    print("\nCited sources:", streaming_answer.cited_source_ids)


if __name__ == "__main__":
    main()
