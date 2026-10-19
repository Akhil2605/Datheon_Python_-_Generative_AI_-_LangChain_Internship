"""Reusable parameterized prompt templates and few-shot examples for structured extraction.

Enforces zero hallucination, strict JSON output according to schemas,
and explicit recording of missing or ambiguous data.
"""

import json
from task1_structured_extraction.schemas import InvoiceData, ResumeData

SYSTEM_INSTRUCTION = """You are an expert precision data extraction assistant.
Your sole purpose is to convert raw unstructured text (invoice or resume) into valid, schema-compliant JSON.

Strict Rules:
1. ONLY extract information directly and explicitly stated in the source text.
2. NEVER invent, fabricate, guess, or extrapolate names, dates, amounts, phone numbers, or experience.
3. If a field is not present or ambiguous, set its value to null (or empty list if appropriate), and list the field name in 'missing_or_uncertain_fields'.
4. Ensure all numerical amounts are pure numbers (floats/integers) without currency symbols.
5. Return ONLY a single raw JSON object matching the requested schema. Do not include markdown fences, comments, or extra conversational text."""

# Few-shot example for Invoice
FEW_SHOT_INVOICE_INPUT = """ACME Logistics Ltd.
Invoice # INV-2024-998
Date: 12 Oct 2024
Billed To: TechCorp Solutions

Items:
- Server Rack Maintenance: $1,200.00 (Qty: 1)
- Ethernet Cabling Service: $350.00
Subtotal: $1,550.00
Tax (10%): $155.00
Total Due: $1,705.00 USD
Please pay within 30 days."""

FEW_SHOT_INVOICE_OUTPUT = {
    "document_type": "invoice",
    "invoice_number": "INV-2024-998",
    "invoice_date": "2024-10-12",
    "due_date": None,
    "vendor_name": "ACME Logistics Ltd.",
    "vendor_address": None,
    "vendor_contact": None,
    "customer_name": "TechCorp Solutions",
    "customer_address": None,
    "customer_contact": None,
    "line_items": [
        {"description": "Server Rack Maintenance", "quantity": 1.0, "unit_price": 1200.0, "total_price": 1200.0},
        {"description": "Ethernet Cabling Service", "quantity": 1.0, "unit_price": 350.0, "total_price": 350.0}
    ],
    "subtotal": 1550.0,
    "tax_amount": 155.0,
    "discount_amount": 0.0,
    "total_amount": 1705.0,
    "currency": "USD",
    "payment_terms": "within 30 days",
    "missing_or_uncertain_fields": [
        "due_date",
        "vendor_address",
        "vendor_contact",
        "customer_address",
        "customer_contact"
    ],
    "notes": "Please pay within 30 days."
}

# Few-shot example for Resume
FEW_SHOT_RESUME_INPUT = """Sarah Connor
Email: sconnor@example.com
Phone: (555) 019-2834
Location: Los Angeles, CA
GitHub: https://github.com/sconnor

Professional Summary:
Versatile Software Engineer with 4 years building distributed backend APIs in Python and Go.

Experience:
Senior Backend Developer | Cyberdyne Systems (2021 - Present)
- Architected high-throughput message pipeline processing 50k events/sec.
- Reduced database query latency by 42% using Redis caching.

Education:
University of California, Los Angeles
B.S. in Computer Science, Graduated 2020

Skills:
Python, Go, PostgreSQL, Redis, Docker, Kubernetes"""

FEW_SHOT_RESUME_OUTPUT = {
    "document_type": "resume",
    "full_name": "Sarah Connor",
    "email": "sconnor@example.com",
    "phone": "(555) 019-2834",
    "location": "Los Angeles, CA",
    "links": ["https://github.com/sconnor"],
    "summary": "Versatile Software Engineer with 4 years building distributed backend APIs in Python and Go.",
    "skills": ["Python", "Go", "PostgreSQL", "Redis", "Docker", "Kubernetes"],
    "work_experience": [
        {
            "company": "Cyberdyne Systems",
            "position": "Senior Backend Developer",
            "start_date": "2021",
            "end_date": "Present",
            "is_current": True,
            "responsibilities_and_achievements": [
                "Architected high-throughput message pipeline processing 50k events/sec.",
                "Reduced database query latency by 42% using Redis caching."
            ],
            "location": None
        }
    ],
    "education": [
        {
            "institution": "University of California, Los Angeles",
            "degree": "B.S.",
            "field_of_study": "Computer Science",
            "graduation_year": "2020",
            "gpa_or_grade": None
        }
    ],
    "certifications": [],
    "projects": [],
    "languages": [],
    "missing_or_uncertain_fields": [
        "certifications",
        "projects",
        "languages",
        "gpa_or_grade",
        "work_experience[0].location"
    ]
}


def build_invoice_prompt(source_text: str) -> str:
    """Builds a formatted prompt for invoice data extraction."""
    schema_json = json.dumps(InvoiceData.model_json_schema(), indent=2)
    example_input = FEW_SHOT_INVOICE_INPUT
    example_output = json.dumps(FEW_SHOT_INVOICE_OUTPUT, indent=2)

    return f"""Target Schema:
{schema_json}

---
Example Input:
{example_input}

Example Output:
{example_output}

---
Now extract data from the following source text. Remember: NEVER invent facts. If missing, set to null and add to 'missing_or_uncertain_fields'.
Output strictly valid JSON matching the schema.

Source Text:
\"\"\"
{source_text.strip()}
\"\"\"
"""


def build_resume_prompt(source_text: str) -> str:
    """Builds a formatted prompt for resume data extraction."""
    schema_json = json.dumps(ResumeData.model_json_schema(), indent=2)
    example_input = FEW_SHOT_RESUME_INPUT
    example_output = json.dumps(FEW_SHOT_RESUME_OUTPUT, indent=2)

    return f"""Target Schema:
{schema_json}

---
Example Input:
{example_input}

Example Output:
{example_output}

---
Now extract data from the following source text. Remember: NEVER invent facts. If missing, set to null and add to 'missing_or_uncertain_fields'.
Output strictly valid JSON matching the schema.

Source Text:
\"\"\"
{source_text.strip()}
\"\"\"
"""
