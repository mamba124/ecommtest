import hashlib
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import ChunkingConfig
from app.ingestion.loaders.base import Document


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict   # inherits from Document, adds chunk_index


class Chunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """
        Returns list of Chunk with stable deterministic chunk_id.
        chunk_id = sha256(source_path + str(chunk_index))[:12]
        """
        chunks: list[Chunk] = []
        chunk_index = 0
        for doc in documents:
            splits = self._splitter.split_text(doc.text)
            for split_text in splits:
                chunk_id = hashlib.sha256(
                    (doc.metadata["source"] + str(chunk_index)).encode()
                ).hexdigest()[:12]
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=split_text,
                    metadata={**doc.metadata, "chunk_index": chunk_index},
                ))
                chunk_index += 1
        return chunks
