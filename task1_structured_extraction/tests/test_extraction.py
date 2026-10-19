"""Unit tests for extraction pipeline, response parsing, and error handling."""

import pytest
from unittest.mock import MagicMock
from task1_structured_extraction.main import (
    clean_and_parse_json,
    extract_structured_data,
    LLMClient
)
from task1_structured_extraction.schemas import InvoiceData, ResumeData


def test_clean_and_parse_json_direct():
    payload = '{"document_type": "invoice", "invoice_number": "INV-100", "total_amount": 100.0}'
    result = clean_and_parse_json(payload)
    assert result["invoice_number"] == "INV-100"
    assert result["total_amount"] == 100.0


def test_clean_and_parse_json_markdown_fence():
    payload = """```json
{
  "document_type": "invoice",
  "invoice_number": "INV-200",
  "total_amount": 250.50
}
```"""
    result = clean_and_parse_json(payload)
    assert result["invoice_number"] == "INV-200"
    assert result["total_amount"] == 250.50


def test_clean_and_parse_json_conversational_wrapping():
    payload = """Here is the extracted invoice data you requested:
{
  "document_type": "invoice",
  "invoice_number": "INV-300",
  "total_amount": 99.00
}
I hope this meets your requirements."""
    result = clean_and_parse_json(payload)
    assert result["invoice_number"] == "INV-300"
    assert result["total_amount"] == 99.00


def test_clean_and_parse_json_malformed():
    with pytest.raises(ValueError, match="Could not extract a valid JSON object"):
        clean_and_parse_json("Not a json string at all")


def test_clean_and_parse_json_empty():
    with pytest.raises(ValueError, match="empty response"):
        clean_and_parse_json("   ")


def test_extract_structured_data_empty_input():
    with pytest.raises(ValueError, match="Source text is empty"):
        extract_structured_data("   ")


def test_extract_structured_data_invoice_mocked():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.generate.return_value = """{
        "document_type": "invoice",
        "invoice_number": "INV-999",
        "vendor_name": "Apex Cloud",
        "line_items": [
            {"description": "Server hosting", "quantity": 1, "unit_price": 500.0, "total_price": 500.0}
        ],
        "total_amount": 500.0,
        "missing_or_uncertain_fields": ["customer_address", "due_date"]
    }"""

    result = extract_structured_data("Some sample invoice text", doc_type="invoice", client=mock_client)
    assert isinstance(result, InvoiceData)
    assert result.invoice_number == "INV-999"
    assert result.vendor_name == "Apex Cloud"
    assert len(result.line_items) == 1
    assert result.line_items[0].total_price == 500.0
    assert "due_date" in result.missing_or_uncertain_fields


def test_extract_structured_data_resume_mocked():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.generate.return_value = """{
        "document_type": "resume",
        "full_name": "Jane Smith",
        "email": "jane@tech.org",
        "skills": ["Python", "FastAPI", "Docker"],
        "work_experience": [
            {
                "company": "DataCorp",
                "position": "Backend Lead",
                "responsibilities_and_achievements": ["Built data pipelines"]
            }
        ],
        "education": [],
        "missing_or_uncertain_fields": ["phone", "location"]
    }"""

    result = extract_structured_data("Some resume text", doc_type="resume", client=mock_client)
    assert isinstance(result, ResumeData)
    assert result.full_name == "Jane Smith"
    assert "Python" in result.skills
    assert result.work_experience[0].company == "DataCorp"
    assert "phone" in result.missing_or_uncertain_fields


def test_llm_client_missing_key_behavior(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="No API provider configured"):
        LLMClient()
