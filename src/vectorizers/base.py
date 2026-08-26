from abc import ABC, abstractmethod
from typing import List

class BaseVectorizer(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        ...
