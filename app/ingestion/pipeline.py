import logging
from pathlib import Path
from typing import Any

from app.core.config import Config
from app.ingestion.chunking import Chunk, Chunker
from app.ingestion.factory import LoaderFactory
from app.ingestion.loaders.base import Document

logger = logging.getLogger("pipeline")


class IngestionPipeline:
    def __init__(self, config: Config, chunking_config: Any = None) -> None:
        # chunking_config can be a ChunkingConfigBody (from the /config/chunking API)
        # or the ChunkingConfig from the base Config.  Both expose the same fields.
        effective_chunking = chunking_config if chunking_config is not None else config.chunking
        self._chunker = Chunker(effective_chunking)
        self._supported = set(LoaderFactory.supported_extensions())

    def run(self, folder_path: str) -> tuple[list[Document], list[Chunk], list[str]]:
        """
        Walk folder_path recursively, load all supported files, chunk them.
        Returns (documents, chunks, failed_files).
        """
        path = Path(folder_path)
        files = [
            f for f in path.rglob("*")
            if f.is_file() and f.suffix.lower() in self._supported
        ]

        all_docs: list[Document] = []
        failed: list[str] = []

        for file in files:
            try:
                docs = LoaderFactory.load(str(file))
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} document(s) from {file}")
            except Exception as exc:
                logger.error(f"Failed to load {file}: {exc}")
                failed.append(str(file))

        chunks = self._chunker.chunk(all_docs)
        logger.info(f"Loaded {len(all_docs)} documents, created {len(chunks)} chunks")
        return all_docs, chunks, failed
