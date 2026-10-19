"""Unit tests for Task 1 prompts and document type detection."""

from task1_structured_extraction.prompts import (
    build_invoice_prompt,
    build_resume_prompt,
    SYSTEM_INSTRUCTION
)
from task1_structured_extraction.main import detect_document_type


def test_system_instruction_anti_hallucination():
    assert "NEVER invent" in SYSTEM_INSTRUCTION
    assert "missing_or_uncertain_fields" in SYSTEM_INSTRUCTION


def test_build_invoice_prompt():
    raw_sample = "Acme Inc. Invoice 101. Total $50.00"
    prompt = build_invoice_prompt(raw_sample)
    assert "Target Schema" in prompt
    assert "Example Input" in prompt
    assert "Example Output" in prompt
    assert "Acme Inc. Invoice 101" in prompt


def test_build_resume_prompt():
    raw_sample = "John Doe. Experience: Engineer at Acme. Education: MIT."
    prompt = build_resume_prompt(raw_sample)
    assert "Target Schema" in prompt
    assert "Example Input" in prompt
    assert "John Doe" in prompt


def test_detect_document_type():
    invoice_text = "INVOICE #994. Billed To: John Doe. Subtotal: $100. Total Due: $110."
    resume_text = "Jane Smith. Professional Summary: Senior Developer. Experience: Cyberdyne. Education: Stanford."

    assert detect_document_type(invoice_text) == "invoice"
    assert detect_document_type(resume_text) == "resume"
