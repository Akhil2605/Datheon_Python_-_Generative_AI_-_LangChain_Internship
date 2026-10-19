"""Task 1: Structured Invoice and Resume Extraction.

Converts unstructured invoice or resume text into validated JSON models
using Google Gemini or OpenAI with strict Pydantic validation.
"""

import os
import sys
import json
import re
import argparse
from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Ensure parent directory is in sys.path when running as a script
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from task1_structured_extraction.schemas import InvoiceData, ResumeData
from task1_structured_extraction.prompts import (
    SYSTEM_INSTRUCTION,
    build_invoice_prompt,
    build_resume_prompt
)

# Load environment variables from root .env
load_dotenv(dotenv_path=ROOT_DIR / ".env")


def clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """Cleans markdown code fences, comments, or extra text and parses JSON."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Model returned an empty response.")

    text = raw_text.strip()

    # Remove markdown code fences if present (e.g. ```json ... ```)
    code_fence_pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    fence_match = re.search(code_fence_pattern, text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost balanced curly braces
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = text[start_idx : end_idx + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as err:
            raise ValueError(f"Extracted JSON substring was malformed: {err}")

    raise ValueError("Could not extract a valid JSON object from model response.")


class LLMClient:
    """Unified client supporting both Google Gemini and OpenAI with error handling."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or self._detect_provider()
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        if self.provider == "gemini":
            if not self.gemini_key or self.gemini_key.strip() == "your_gemini_api_key_here":
                raise ValueError(
                    "GEMINI_API_KEY is not configured. Please set a valid key in .env or environment."
                )
            from google import genai
            self.client = genai.Client(api_key=self.gemini_key)
            self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        elif self.provider == "openai":
            if not self.openai_key or self.openai_key.strip() == "your_openai_api_key_here":
                raise ValueError(
                    "OPENAI_API_KEY is not configured. Please set a valid key in .env or environment."
                )
            import openai
            self.client = openai.OpenAI(api_key=self.openai_key)
            self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        elif self.provider == "mock":
            self.client = None
            self.model_name = "mock-offline-model"

        else:
            raise ValueError(
                "No API provider configured. Please set GEMINI_API_KEY or OPENAI_API_KEY in your .env file."
            )

    @staticmethod
    def _detect_provider() -> str:
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if gemini_key and gemini_key.strip() and gemini_key != "your_gemini_api_key_here":
            return "gemini"
        if openai_key and openai_key.strip() and openai_key != "your_openai_api_key_here":
            return "openai"
        return "none"

    def generate(self, prompt: str, system_instruction: str = SYSTEM_INSTRUCTION) -> str:
        """Sends the prompt to the selected LLM and returns the text response."""
        if self.provider == "mock":
            raise RuntimeError("Mock provider requires a mocked response handler in tests.")

        try:
            if self.provider == "gemini":
                from google.genai import types
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0,
                    ),
                )
                if not response.text:
                    raise RuntimeError("Gemini returned an empty response.")
                return response.text

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("OpenAI returned an empty response.")
                return content

        except Exception as exc:
            # Mask API keys if they accidentally appear in exception messages
            err_msg = str(exc)
            if self.gemini_key and self.gemini_key in err_msg:
                err_msg = err_msg.replace(self.gemini_key, "[REDACTED]")
            if self.openai_key and self.openai_key in err_msg:
                err_msg = err_msg.replace(self.openai_key, "[REDACTED]")
            raise RuntimeError(f"LLM API Error ({self.provider}): {err_msg}") from None


def detect_document_type(text: str) -> str:
    """Heuristic helper to detect document type if not specified by user."""
    lower = text.lower()
    invoice_keywords = ["invoice", "billed to", "subtotal", "total due", "unit price", "qty", "tax", "due date"]
    resume_keywords = ["experience", "education", "skills", "projects", "certifications", "curriculum vitae", "resume", "work history"]

    invoice_score = sum(1 for kw in invoice_keywords if kw in lower)
    resume_score = sum(1 for kw in resume_keywords if kw in lower)

    return "invoice" if invoice_score >= resume_score else "resume"


def extract_structured_data(
    source_text: str,
    doc_type: Optional[str] = None,
    client: Optional[LLMClient] = None
) -> Union[InvoiceData, ResumeData]:
    """Main extraction pipeline: formats prompt, calls LLM, and validates schema."""
    if not source_text or not source_text.strip():
        raise ValueError("Source text is empty. Please provide valid document content.")

    inferred_type = doc_type.lower() if doc_type else detect_document_type(source_text)
    if inferred_type not in ["invoice", "resume"]:
        raise ValueError(f"Unsupported document type '{inferred_type}'. Must be 'invoice' or 'resume'.")

    prompt = build_invoice_prompt(source_text) if inferred_type == "invoice" else build_resume_prompt(source_text)

    llm = client or LLMClient()
    raw_response = llm.generate(prompt)

    parsed_json = clean_and_parse_json(raw_response)

    if inferred_type == "invoice":
        return InvoiceData.model_validate(parsed_json)
    else:
        return ResumeData.model_validate(parsed_json)


def main():
    parser = argparse.ArgumentParser(
        description="Structured Invoice & Resume Extraction CLI (Datheon Internship)"
    )
    parser.add_argument("--file", "-f", help="Path to input text or document file")
    parser.add_argument("--text", "-t", help="Raw input text to process")
    parser.add_argument("--type", choices=["invoice", "resume"], help="Explicit document type (auto-detected if omitted)")
    parser.add_argument("--output", "-o", help="Optional path to write formatted JSON output")
    parser.add_argument("--provider", choices=["gemini", "openai"], help="Override default LLM provider")

    args = parser.parse_args()

    # Determine input text
    source_text = None
    if args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        source_text = input_path.read_text(encoding="utf-8", errors="replace")
    elif args.text:
        source_text = args.text
    else:
        print("Enter/paste document text (press Ctrl+Z on Windows then Enter to submit):")
        source_text = sys.stdin.read()

    if not source_text or not source_text.strip():
        print("Error: No input text provided.", file=sys.stderr)
        sys.exit(1)

    try:
        client = LLMClient(provider=args.provider)
        result = extract_structured_data(source_text, doc_type=args.type, client=client)
        output_json = result.model_dump_json(indent=2)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output_json, encoding="utf-8")
            print(f"Extraction successful. Output saved to {out_path}")
        else:
            print(output_json)

    except Exception as exc:
        print(f"Extraction Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
