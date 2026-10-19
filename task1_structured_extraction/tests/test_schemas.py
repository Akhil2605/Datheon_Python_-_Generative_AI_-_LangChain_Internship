"""Unit tests for Task 1 Pydantic Schemas."""

import pytest
from pydantic import ValidationError
from task1_structured_extraction.schemas import (
    InvoiceData,
    InvoiceItem,
    ResumeData,
    WorkExperience,
    Education,
)


def test_invoice_item_validation():
    item = InvoiceItem(description="Cloud Hosting", quantity=2, unit_price=100.0, total_price=200.0)
    assert item.description == "Cloud Hosting"
    assert item.quantity == 2.0
    assert item.total_price == 200.0

    # Test missing mandatory description
    with pytest.raises(ValidationError):
        InvoiceItem(quantity=1)


def test_invoice_data_defaults_and_missing_fields():
    data = InvoiceData(
        invoice_number="INV-001",
        total_amount=500.0,
        missing_or_uncertain_fields=["due_date", "vendor_contact"]
    )
    assert data.document_type == "invoice"
    assert data.invoice_number == "INV-001"
    assert data.total_amount == 500.0
    assert data.line_items == []
    assert "due_date" in data.missing_or_uncertain_fields
    assert "vendor_contact" in data.missing_or_uncertain_fields


def test_resume_data_validation():
    work = WorkExperience(
        company="TechCorp",
        position="Senior Developer",
        start_date="2021",
        end_date="Present",
        is_current=True,
        responsibilities_and_achievements=["Built API", "Optimized queries"]
    )
    edu = Education(
        institution="State University",
        degree="B.S.",
        field_of_study="Computer Science",
        graduation_year="2020"
    )
    resume = ResumeData(
        full_name="Jane Doe",
        email="jane@example.com",
        skills=["Python", "SQL"],
        work_experience=[work],
        education=[edu],
        missing_or_uncertain_fields=["certifications"]
    )

    assert resume.document_type == "resume"
    assert resume.full_name == "Jane Doe"
    assert len(resume.work_experience) == 1
    assert resume.work_experience[0].company == "TechCorp"
    assert resume.education[0].graduation_year == "2020"
    assert "certifications" in resume.missing_or_uncertain_fields


def test_schema_json_roundtrip():
    data = InvoiceData(
        invoice_number="INV-1234",
        vendor_name="Acme",
        line_items=[InvoiceItem(description="Widget", quantity=5, unit_price=10.0, total_price=50.0)],
        total_amount=50.0
    )
    json_str = data.model_dump_json()
    reloaded = InvoiceData.model_validate_json(json_str)
    assert reloaded.invoice_number == "INV-1234"
    assert len(reloaded.line_items) == 1
    assert reloaded.line_items[0].total_price == 50.0
