import logging

from app.ingestion.loaders.base import Document
from app.ingestion.loaders.pdf import PDFLoader
from app.ingestion.loaders.markdown import MarkdownLoader
from app.ingestion.loaders.text import TextLoader

logger = logging.getLogger("loader_factory")

_LOADERS = {
    ".pdf": PDFLoader(),
    ".md": MarkdownLoader(),
    ".txt": TextLoader(),
}


class LoaderFactory:
    @staticmethod
    def load(filepath: str) -> list[Document]:
        """
        Returns list of Document(text=..., metadata={source, page, section}).
        Dispatch by file extension.
        """
        suffix = ""
        if "." in filepath:
            suffix = "." + filepath.rsplit(".", 1)[-1].lower()
        loader = _LOADERS.get(suffix)
        if loader is None:
            logger.warning(f"No loader for extension {suffix!r}, skipping {filepath}")
            return []
        return loader.load(filepath)

    @staticmethod
    def supported_extensions() -> list[str]:
        return list(_LOADERS.keys())
