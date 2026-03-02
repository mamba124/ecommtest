import re

from app.ingestion.loaders.base import BaseLoader, Document


class MarkdownLoader(BaseLoader):
    def load(self, filepath: str) -> list[Document]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        section: str | None = None
        match = re.search(r"^#{1,2}\s+(.+)$", text, re.MULTILINE)
        if match:
            section = match.group(1).strip()

        return [Document(
            text=text,
            metadata={
                "source": filepath,
                "page": None,
                "section": section,
            },
        )]
