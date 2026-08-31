from unittest.mock import MagicMock

import pytest

from src.errors import EmbeddingError, GenerationError, is_transient_by_signature


def _make_anthropic_status_error(status_code: int):
    import anthropic

    response = MagicMock()
    response.status_code = status_code
    response.headers = {}
    return anthropic.APIStatusError("boom", response=response, body=None)


def _make_openai_status_error(status_code: int):
    import openai

    response = MagicMock()
    response.status_code = status_code
    response.headers = {}
    return openai.APIStatusError("boom", response=response, body=None)


def test_anthropic_generator_wraps_status_error_as_generation_error(monkeypatch):
    from src.generation.anthropic_generator import AnthropicGenerator

    generator = AnthropicGenerator.__new__(AnthropicGenerator)
    generator.model = "claude-opus-5"
    generator.max_tokens = 100
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = _make_anthropic_status_error(500)
    generator._client = fake_client

    with pytest.raises(GenerationError) as exc_info:
        generator.generate("question?", [])

    assert exc_info.value.provider == "anthropic"
    assert "500" in str(exc_info.value)


def test_openai_generator_wraps_status_error_as_generation_error():
    from src.generation.openai_generator import OpenAIGenerator

    generator = OpenAIGenerator.__new__(OpenAIGenerator)
    generator.model = "gpt-4o-mini"
    generator.max_tokens = 100
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = _make_openai_status_error(503)
    generator._client = fake_client

    with pytest.raises(GenerationError) as exc_info:
        generator.generate("question?", [])

    assert exc_info.value.provider == "openai"
    assert "503" in str(exc_info.value)


def test_openai_vectorizer_wraps_status_error_as_embedding_error():
    from src.vectorizers.openai_vectorizer import OpenAIVectorizer

    vectorizer = OpenAIVectorizer.__new__(OpenAIVectorizer)
    vectorizer.model = "text-embedding-3-small"
    fake_client = MagicMock()
    fake_client.embeddings.create.side_effect = _make_openai_status_error(429)
    vectorizer._client = fake_client

    with pytest.raises(EmbeddingError) as exc_info:
        vectorizer.embed(["hello"])

    assert exc_info.value.provider == "openai"


def test_voyage_vectorizer_wraps_any_failure_as_embedding_error():
    from src.vectorizers.voyage_vectorizer import VoyageVectorizer

    vectorizer = VoyageVectorizer.__new__(VoyageVectorizer)
    vectorizer.model = "voyage-3.5"
    fake_client = MagicMock()
    fake_client.embed.side_effect = RuntimeError("rate limited")
    vectorizer._client = fake_client

    with pytest.raises(EmbeddingError) as exc_info:
        vectorizer.embed(["hello"])

    assert exc_info.value.provider == "voyage"


def test_is_transient_by_signature():
    class RateLimitError(Exception):
        pass

    class SomeValidationError(Exception):
        pass

    class StatusCoded(Exception):
        status_code = 503

    assert is_transient_by_signature(RateLimitError("x"))
    assert not is_transient_by_signature(SomeValidationError("x"))
    assert is_transient_by_signature(StatusCoded("x"))
