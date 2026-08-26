from pydantic import BaseModel

class Document(BaseModel):
    id: str
    text: str
    metadata: dict = {}

class Normalizer:
    """Base class: implement per-source cleaning/schema mapping."""
    def normalize(self, raw: dict) -> Document:
        raise NotImplementedError
