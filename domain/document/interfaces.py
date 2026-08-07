from __future__ import annotations

from abc import ABC, abstractmethod


class Chunkable(ABC):
    @abstractmethod
    def chunk(self) -> None:
        """Chunk the document."""