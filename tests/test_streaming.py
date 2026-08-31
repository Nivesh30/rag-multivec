from unittest.mock import MagicMock

import pytest

from src.errors import GenerationError


def test_base_generator_default_stream_falls_back_to_generate():
    from src.generation.base import BaseGenerator

    class SimpleGenerator(BaseGenerator):
        def generate(self, question, context):
            return "a single-shot answer"

    chunks = list(SimpleGenerator().stream("question?", []))
    assert chunks == ["a single-shot answer"]


def test_anthropic_generator_stream_yields_text_chunks():
    from src.generation.anthropic_generator import AnthropicGenerator

    generator = AnthropicGenerator.__new__(AnthropicGenerator)
    generator.model = "claude-opus-5"
    generator.max_tokens = 100

    final_message = MagicMock()
    final_message.stop_reason = "end_turn"

    stream_ctx = MagicMock()
    stream_ctx.__enter__.return_value.text_stream = iter(["Hel", "lo", " world"])
    stream_ctx.__enter__.return_value.get_final_message.return_value = final_message
    stream_ctx.__exit__.return_value = False

    fake_client = MagicMock()
    fake_client.messages.stream.return_value = stream_ctx
    generator._client = fake_client

    chunks = list(generator.stream("question?", []))
    assert chunks == ["Hel", "lo", " world"]


def test_anthropic_generator_stream_raises_on_refusal_after_yielding():
    from src.generation.anthropic_generator import AnthropicGenerator

    generator = AnthropicGenerator.__new__(AnthropicGenerator)
    generator.model = "claude-opus-5"
    generator.max_tokens = 100

    final_message = MagicMock()
    final_message.stop_reason = "refusal"
    final_message.stop_details = None

    stream_ctx = MagicMock()
    stream_ctx.__enter__.return_value.text_stream = iter(["partial"])
    stream_ctx.__enter__.return_value.get_final_message.return_value = final_message
    stream_ctx.__exit__.return_value = False

    fake_client = MagicMock()
    fake_client.messages.stream.return_value = stream_ctx
    generator._client = fake_client

    gen = generator.stream("question?", [])
    assert next(gen) == "partial"
    with pytest.raises(GenerationError):
        next(gen)


def test_openai_generator_stream_yields_text_chunks():
    from src.generation.openai_generator import OpenAIGenerator

    generator = OpenAIGenerator.__new__(OpenAIGenerator)
    generator.model = "gpt-4o-mini"
    generator.max_tokens = 100

    def make_chunk(text):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        return chunk

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = iter(
        [make_chunk("Hel"), make_chunk("lo"), make_chunk(None), make_chunk(" world")]
    )
    generator._client = fake_client

    chunks = list(generator.stream("question?", []))
    assert chunks == ["Hel", "lo", " world"]
