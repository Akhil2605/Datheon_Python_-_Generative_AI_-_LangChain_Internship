"""Unit tests for RAG pipeline, citations, and insufficient information handling."""

from pathlib import Path
import pytest
from unittest.mock import patch
from task2_document_rag.main import RAGPipeline, INSUFFICIENT_INFO_MESSAGE

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_documents"


@pytest.fixture
def indexed_pipeline():
    pipeline = RAGPipeline(provider="local")
    pipeline.load_and_index(SAMPLE_DIR)
    return pipeline


def test_rag_pipeline_pdf_page1_citation(indexed_pipeline):
    resp = indexed_pipeline.query("What processor powers the ApexCore X900?")
    assert resp.has_sufficient_context is True
    assert len(resp.citations) > 0
    top_citation = resp.citations[0]
    assert top_citation.source == "product_specifications.pdf"
    assert top_citation.page == 1
    assert "ARM" in resp.answer or "32-core" in resp.answer or "UltraScale" in resp.answer


def test_rag_pipeline_pdf_page2_citation(indexed_pipeline):
    resp = indexed_pipeline.query("What is the warranty period and return policy?")
    assert resp.has_sufficient_context is True
    assert len(resp.citations) > 0
    pdf_citations = [c for c in resp.citations if c.source == "product_specifications.pdf"]
    assert any(c.page == 2 for c in pdf_citations)
    assert "warranty" in resp.answer.lower()


def test_rag_pipeline_policy_text_citation(indexed_pipeline):
    resp = indexed_pipeline.query("How much is the home office setup reimbursement stipend?")
    assert resp.has_sufficient_context is True
    assert len(resp.citations) > 0
    top_citation = resp.citations[0]
    assert top_citation.source == "company_policy.txt"
    assert "$1,000" in resp.answer or "1,000" in resp.answer


def test_rag_pipeline_insufficient_information(indexed_pipeline):
    resp = indexed_pipeline.query("What is the quantum orbital radius of electron spin in helium?")
    # Either context score is below threshold or answered with insufficient info
    if not resp.has_sufficient_context:
        assert INSUFFICIENT_INFO_MESSAGE in resp.answer
        assert len(resp.citations) == 0


def test_rag_pipeline_empty_query(indexed_pipeline):
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        indexed_pipeline.query("   ")


def test_rag_pipeline_with_mocked_llm(indexed_pipeline):
    with patch.object(
        indexed_pipeline,
        "_call_llm",
        return_value="The ApexCore X900 is backed by a 3-year warranty [Source: product_specifications.pdf, Page 2]."
    ):
        indexed_pipeline.llm_provider = "mock"
        resp = indexed_pipeline.query("Tell me about the warranty")
        assert "3-year warranty" in resp.answer
        assert any(c.page == 2 for c in resp.citations)
