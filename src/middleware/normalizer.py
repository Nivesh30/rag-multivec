from pydantic import BaseModel


class Document(BaseModel):
    id: str
    text: str
    metadata: dict = {}


class Normalizer:
    """Base class: implement per-source cleaning/schema mapping."""
    def normalize(self, raw: dict) -> Document:
        raise NotImplementedError


class PlainTextNormalizer(Normalizer):
    """Default normalizer for {"id": ..., "text": ...} style raw records."""

    def normalize(self, raw: dict) -> Document:
        text = " ".join(str(raw.get("text", "")).split())
        metadata = {k: v for k, v in raw.items() if k not in ("id", "text")}
        return Document(id=str(raw["id"]), text=text, metadata=metadata)
