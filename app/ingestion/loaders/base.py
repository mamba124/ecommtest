from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    text: str
    metadata: dict   # keys: source (str), page (int|None), section (str|None)


class BaseLoader(ABC):
    @abstractmethod
    def load(self, filepath: str) -> list[Document]:
        """Load a file and return a list of Documents."""
        ...
