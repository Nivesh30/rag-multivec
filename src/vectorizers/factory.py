from typing import TYPE_CHECKING

from src.vectorizers.base import BaseVectorizer

if TYPE_CHECKING:
    from src.config.settings import VectorizerConfig

BACKENDS = ("voyage", "openai", "sentence_transformers")


def build_vectorizer(config: "VectorizerConfig") -> BaseVectorizer:
    backend = config.backend.lower()

    if backend == "voyage":
        from src.vectorizers.voyage_vectorizer import DEFAULT_MODEL, VoyageVectorizer

        return VoyageVectorizer(api_key=config.voyage_api_key, model=config.model or DEFAULT_MODEL)

    if backend == "openai":
        from src.vectorizers.openai_vectorizer import DEFAULT_MODEL, OpenAIVectorizer

        return OpenAIVectorizer(api_key=config.openai_api_key, model=config.model or DEFAULT_MODEL)

    if backend == "sentence_transformers":
        from src.vectorizers.sentence_transformer_vectorizer import SentenceTransformerVectorizer

        return SentenceTransformerVectorizer(model=config.model)

    raise ValueError(f"Unknown vectorizer backend '{config.backend}'. Choose from {BACKENDS}.")
