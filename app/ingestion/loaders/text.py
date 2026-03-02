from app.ingestion.loaders.base import BaseLoader, Document


class TextLoader(BaseLoader):
    def load(self, filepath: str) -> list[Document]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return [Document(
            text=text,
            metadata={
                "source": filepath,
                "page": None,
                "section": None,
            },
        )]
