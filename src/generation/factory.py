from typing import TYPE_CHECKING

from src.generation.base import BaseGenerator

if TYPE_CHECKING:
    from src.config.settings import GenerationConfig

PROVIDERS = ("anthropic", "openai")


def build_generator(config: "GenerationConfig") -> BaseGenerator:
    """Build a generator from config. Any provider can be added by implementing
    BaseGenerator and registering it here - the rest of the pipeline only depends
    on that interface, not on a specific vendor."""
    provider = config.provider.lower()

    if provider == "anthropic":
        from src.generation.anthropic_generator import DEFAULT_MODEL, AnthropicGenerator

        return AnthropicGenerator(
            api_key=config.anthropic_api_key,
            model=config.model or DEFAULT_MODEL,
            max_tokens=config.max_tokens,
            max_retries=config.max_retries,
        )

    if provider == "openai":
        from src.generation.openai_generator import DEFAULT_MODEL, OpenAIGenerator

        return OpenAIGenerator(
            api_key=config.openai_api_key,
            model=config.model or DEFAULT_MODEL,
            max_tokens=config.max_tokens,
            max_retries=config.max_retries,
        )

    raise ValueError(f"Unknown generation provider '{config.provider}'. Choose from {PROVIDERS}.")
