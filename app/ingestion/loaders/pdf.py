import fitz  # pymupdf

from app.ingestion.loaders.base import BaseLoader, Document


class PDFLoader(BaseLoader):
    def load(self, filepath: str) -> list[Document]:
        documents: list[Document] = []
        doc = fitz.open(filepath)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                documents.append(Document(
                    text=text,
                    metadata={
                        "source": filepath,
                        "page": page_num + 1,
                        "section": None,
                    },
                ))
        doc.close()
        return documents
