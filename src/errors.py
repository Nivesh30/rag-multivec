class RAGError(Exception):
    """Base class for domain-level errors raised by this pipeline."""


class EmbeddingError(RAGError):
    """Raised when an embedding backend fails after its retries are exhausted."""

    def __init__(self, provider: str, message: str, cause: BaseException = None):
        super().__init__(f"[{provider} embedding] {message}")
        self.provider = provider
        self.__cause__ = cause


class GenerationError(RAGError):
    """Raised when a generation backend fails after its retries are exhausted."""

    def __init__(self, provider: str, message: str, cause: BaseException = None):
        super().__init__(f"[{provider} generation] {message}")
        self.provider = provider
        self.__cause__ = cause


def is_transient_by_signature(exc: BaseException) -> bool:
    """Duck-typed transient-error check for a provider whose SDK doesn't
    expose a stable typed exception hierarchy here (e.g. Voyage AI). Looks
    for a rate-limit/connection/server-error-shaped class name, or a
    429/5xx-looking `status_code` attribute."""
    name = type(exc).__name__
    if any(key in name for key in ("RateLimit", "Timeout", "Connection", "ServiceUnavailable", "InternalServer")):
        return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429 or status_code >= 500
    return False
