import os
from dataclasses import dataclass, field
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


@dataclass
class VectorizerConfig:
    backend: str = field(default_factory=lambda: _env("VECTORIZER_BACKEND", "sentence_transformers"))
    model: Optional[str] = field(default_factory=lambda: _env("VECTORIZER_MODEL"))
    voyage_api_key: Optional[str] = field(default_factory=lambda: _env("VOYAGE_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    max_retries: int = field(default_factory=lambda: int(_env("VECTORIZER_MAX_RETRIES", "2")))
    batch_size: int = field(default_factory=lambda: int(_env("VECTORIZER_BATCH_SIZE", "100")))


@dataclass
class VectorStoreConfig:
    backend: str = field(default_factory=lambda: _env("VECTOR_STORE_BACKEND", "chroma"))
    collection_name: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "documents"))
    persist_directory: Optional[str] = field(default_factory=lambda: _env("CHROMA_PERSIST_DIR"))


@dataclass
class GenerationConfig:
    provider: str = field(default_factory=lambda: _env("GENERATION_PROVIDER", "anthropic"))
    model: Optional[str] = field(default_factory=lambda: _env("GENERATION_MODEL"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    max_tokens: int = field(default_factory=lambda: int(_env("GENERATION_MAX_TOKENS", "2048")))
    max_retries: int = field(default_factory=lambda: int(_env("GENERATION_MAX_RETRIES", "2")))


@dataclass
class ChunkingConfig:
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "100")))


@dataclass
class RetrievalConfig:
    top_k: int = field(default_factory=lambda: int(_env("RETRIEVAL_TOP_K", "5")))
    use_hybrid: bool = field(default_factory=lambda: _env("RETRIEVAL_HYBRID", "true").lower() == "true")
    rrf_k: int = field(default_factory=lambda: int(_env("RETRIEVAL_RRF_K", "60")))


@dataclass
class Settings:
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


def load_settings() -> Settings:
    """Build Settings from environment variables (see .env.example)."""
    return Settings()
