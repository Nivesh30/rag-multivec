import logging
import os


def configure_logging(level: str = None) -> None:
    """Configure logging for this package.

    Call this once at process/script startup (see examples/). Library code
    itself only ever does `logging.getLogger(__name__)` and never configures
    handlers, so importing src.pipeline doesn't silently install a root
    handler in someone else's application.
    """
    resolved = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
