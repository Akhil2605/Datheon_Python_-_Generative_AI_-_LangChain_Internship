"""Unit tests for document loader (PDF, text, error handling)."""

from pathlib import Path
import pytest
from task2_document_rag.document_loader import load_document, DocumentChunk

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_documents"


def test_load_text_document():
    txt_path = SAMPLE_DIR / "company_policy.txt"
    assert txt_path.exists(), "Sample text file should exist"

    chunks = load_document(txt_path)
    assert len(chunks) == 1
    assert isinstance(chunks[0], DocumentChunk)
    assert chunks[0].source == "company_policy.txt"
    assert chunks[0].page == 1
    assert "ANNUAL LEAVE & TIME OFF POLICY" in chunks[0].content


def test_load_pdf_document():
    pdf_path = SAMPLE_DIR / "product_specifications.pdf"
    assert pdf_path.exists(), "Sample PDF file should exist"

    pages = load_document(pdf_path)
    assert len(pages) == 2, "PDF should contain exactly 2 pages"
    assert pages[0].source == "product_specifications.pdf"
    assert pages[0].page == 1
    assert "Hardware & Performance" in pages[0].content or "ApexCore" in pages[0].content

    assert pages[1].source == "product_specifications.pdf"
    assert pages[1].page == 2
    assert "Warranty" in pages[1].content


def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_document("non_existent_document_12345.pdf")


def test_load_unsupported_file(tmp_path):
    fake_csv = tmp_path / "data.csv"
    fake_csv.write_text("a,b,c\n1,2,3")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(fake_csv)


def test_load_empty_file(tmp_path):
    empty_txt = tmp_path / "empty.txt"
    empty_txt.write_text("   \n\t  ")
    with pytest.raises(ValueError, match="contains no extractable text"):
        load_document(empty_txt)
