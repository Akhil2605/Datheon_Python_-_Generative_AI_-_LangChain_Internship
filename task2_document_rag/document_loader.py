"""Document loader and text chunking utilities for RAG.

Supports PDF and plain-text documents with page-level metadata tracking
using pypdf and LangChain's RecursiveCharacterTextSplitter.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    """Represents a text segment with provenance metadata for citations."""
    content: str
    source: str
    page: Optional[int] = None
    chunk_index: int = 0


def load_document(file_path: Union[str, Path]) -> List[DocumentChunk]:
    """Loads a single document (PDF or text) and returns raw page/section chunks.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is unsupported or completely empty.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    filename = path.name

    if suffix not in [".pdf", ".txt", ".md"]:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported types are: .pdf, .txt, .md")

    raw_chunks: List[DocumentChunk] = []

    if suffix == ".pdf":
        try:
            reader = pypdf.PdfReader(str(path))
            if len(reader.pages) == 0:
                raise ValueError(f"PDF document is empty (0 pages): {filename}")

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                cleaned = page_text.strip()
                if cleaned:
                    raw_chunks.append(DocumentChunk(content=cleaned, source=filename, page=page_num))

        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"Failed to read PDF '{filename}': {exc}")

    elif suffix in [".txt", ".md"]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                raw_chunks.append(DocumentChunk(content=text, source=filename, page=1))
        except Exception as exc:
            raise ValueError(f"Failed to read text file '{filename}': {exc}")

    if not raw_chunks:
        raise ValueError(f"Document '{filename}' contains no extractable text.")

    return raw_chunks


def split_documents(
    documents: List[DocumentChunk],
    chunk_size: int = 400,
    chunk_overlap: int = 80
) -> List[DocumentChunk]:
    """Splits documents into smaller overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Preserves exact document provenance (source filename and page number).
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    result_chunks: List[DocumentChunk] = []
    idx = 0

    for doc in documents:
        splits = splitter.split_text(doc.content)
        for piece in splits:
            cleaned = piece.strip()
            if cleaned:
                result_chunks.append(
                    DocumentChunk(
                        content=cleaned,
                        source=doc.source,
                        page=doc.page,
                        chunk_index=idx
                    )
                )
                idx += 1

    return result_chunks
