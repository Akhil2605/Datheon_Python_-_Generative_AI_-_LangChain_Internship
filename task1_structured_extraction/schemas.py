"""Pydantic schemas for structured extraction of invoices and resumes.

Enforces strict validation, field types, and explicit tracking of missing
or uncertain information to prevent hallucination.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class InvoiceItem(BaseModel):
    """Represents a single line item in an invoice."""
    description: str = Field(..., description="Description of the item or service")
    quantity: Optional[float] = Field(None, description="Quantity billed, if stated")
    unit_price: Optional[float] = Field(None, description="Price per unit, if stated")
    total_price: Optional[float] = Field(None, description="Total cost for this line item, if stated")


class InvoiceData(BaseModel):
    """Structured data extracted from an invoice."""
    document_type: Literal["invoice"] = Field("invoice", description="Type identifier")
    invoice_number: Optional[str] = Field(None, description="Invoice identification number")
    invoice_date: Optional[str] = Field(None, description="Date invoice was issued")
    due_date: Optional[str] = Field(None, description="Payment due date")
    
    # Vendor / Seller
    vendor_name: Optional[str] = Field(None, description="Name of company or person issuing invoice")
    vendor_address: Optional[str] = Field(None, description="Physical/mailing address of vendor")
    vendor_contact: Optional[str] = Field(None, description="Email or phone number of vendor")
    
    # Customer / Buyer
    customer_name: Optional[str] = Field(None, description="Name of recipient/customer")
    customer_address: Optional[str] = Field(None, description="Address of customer")
    customer_contact: Optional[str] = Field(None, description="Contact information of customer")
    
    # Line items & Financials
    line_items: List[InvoiceItem] = Field(default_factory=list, description="List of purchased items/services")
    subtotal: Optional[float] = Field(None, description="Subtotal before taxes/discounts")
    tax_amount: Optional[float] = Field(None, description="Tax charged")
    discount_amount: Optional[float] = Field(None, description="Discounts applied")
    total_amount: Optional[float] = Field(None, description="Final total amount due")
    currency: Optional[str] = Field(None, description="Currency code or symbol (e.g., USD, INR, EUR)")
    payment_terms: Optional[str] = Field(None, description="Payment terms (e.g., Net 30, Due on Receipt)")
    
    # Data Quality & Grounding
    missing_or_uncertain_fields: List[str] = Field(
        default_factory=list,
        description="Fields not explicitly found in source text or considered uncertain"
    )
    notes: Optional[str] = Field(None, description="Any additional remarks or disclaimers on invoice")

    @field_validator("line_items", mode="before")
    @classmethod
    def ensure_items_list(cls, v):
        if v is None:
            return []
        return v


class WorkExperience(BaseModel):
    """Represents a single job role or employment history entry."""
    company: str = Field(..., description="Company or organization name")
    position: Optional[str] = Field(None, description="Job title / designation")
    start_date: Optional[str] = Field(None, description="Start date (month/year or year)")
    end_date: Optional[str] = Field(None, description="End date or 'Present'")
    is_current: Optional[bool] = Field(None, description="Whether candidate currently works here")
    responsibilities_and_achievements: List[str] = Field(
        default_factory=list,
        description="Key duties, bullet points, or quantifiable achievements"
    )
    location: Optional[str] = Field(None, description="City, country, or Remote")


class Education(BaseModel):
    """Represents an academic qualification."""
    institution: str = Field(..., description="University, college, or school name")
    degree: Optional[str] = Field(None, description="Degree or diploma name (e.g., B.S., M.Sc., Ph.D.)")
    field_of_study: Optional[str] = Field(None, description="Major, discipline, or specialization")
    graduation_year: Optional[str] = Field(None, description="Year of completion or expected graduation")
    gpa_or_grade: Optional[str] = Field(None, description="GPA or percentage, if mentioned")


class ResumeData(BaseModel):
    """Structured data extracted from a resume or CV."""
    document_type: Literal["resume"] = Field("resume", description="Type identifier")
    full_name: Optional[str] = Field(None, description="Candidate's full name")
    email: Optional[str] = Field(None, description="Contact email address")
    phone: Optional[str] = Field(None, description="Contact telephone number")
    location: Optional[str] = Field(None, description="Current residence (City, State/Country)")
    links: List[str] = Field(default_factory=list, description="URLs (LinkedIn, GitHub, portfolio)")
    summary: Optional[str] = Field(None, description="Professional summary or objective statement")
    
    skills: List[str] = Field(default_factory=list, description="Technical and soft skills")
    work_experience: List[WorkExperience] = Field(default_factory=list, description="Professional experience")
    education: List[Education] = Field(default_factory=list, description="Academic background")
    certifications: List[str] = Field(default_factory=list, description="Certificates and licenses")
    projects: List[str] = Field(default_factory=list, description="Notable projects mentioned")
    languages: List[str] = Field(default_factory=list, description="Spoken/written languages")
    
    missing_or_uncertain_fields: List[str] = Field(
        default_factory=list,
        description="Fields not explicitly found in source text or considered uncertain"
    )
