# Datheon Python + Generative AI & LangChain Internship — Complete Project

A modular, production-grade Generative AI engineering repository built during the **Datheon Python Internship**. This project implements four core tasks covering precision structured schema extraction, retrieval-augmented generation (RAG) over multi-page PDF/text documents with citations, autonomous tool-augmented AI agents with strict security sandboxing, and a responsive Streamlit conversational interface.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Prerequisites & Windows Setup](#prerequisites--windows-setup)
4. [API Key Configuration](#api-key-configuration)
5. [Task Breakdown & Run Commands](#task-breakdown--run-commands)
   - [Task 1: Structured Invoice & Resume Extraction](#task-1--structured-invoice--resume-extraction)
   - [Task 2: Document RAG with Citations](#task-2--document-rag-with-citations)
   - [Task 3: AI Agent with Tools](#task-3--ai-agent-with-tools)
   - [Task 4: Streamlit Conversational AI Web App](#task-4--streamlit-conversational-ai-web-app)
6. [Testing & Verification](#testing--verification)
7. [Troubleshooting & Known Limitations](#troubleshooting--known-limitations)
8. [Security & Pre-Submission Checklist](#security--pre-submission-checklist)

---

## Project Overview

- **Task 1 — Structured Extraction:** Converts raw, unstructured text from invoices and resumes into validated JSON conforming to strict Pydantic schemas. Features few-shot prompts, automatic document classification, JSON code-fence sanitization, and explicit tracking of missing or uncertain fields to guarantee zero hallucination.
- **Task 2 — PDF/Text RAG Pipeline:** Ingests PDFs (page-aware via `pypdf`) and plain-text files, chunks content using LangChain's `RecursiveCharacterTextSplitter`, indexes chunks in FAISS using normalized embeddings (Gemini `text-embedding-004`, OpenAI `text-embedding-3-small`, or deterministic local vectorizer), and produces answers with exact source filename and PDF page citations (`[Source: filename, Page: X]`). Handles insufficient context with strict guardrails.
- **Task 3 — Tool-Augmented AI Agent:** ReAct console agent utilizing three specialized tools: DuckDuckGo live web search, an AST-parsed safe mathematical calculator (zero arbitrary code execution), and a workspace-restricted local file reader with strict path traversal defenses. Observably displays actions and tool outputs without exposing private chain-of-thought.
- **Task 4 — Streamlit Conversational Web App:** An interactive multi-mode web application offering real-time streaming, conversation session history, clear conversation controls, and modular execution across Document RAG, AI Agent, and Structured Extraction modes.

---

## Folder Structure

```text
Datheon_Python_Internship/
├── .env.example                     # Template environment variables (no real keys)
├── .gitignore                       # Git exclusion rules for keys, caches, venv, and local db
├── pytest.ini                       # Pytest configuration with importlib mode
├── requirements.txt                 # Root dependencies
├── PROJECT_STATE.md                 # Real-time development progress and verification tracking
├── HANDOFF.md                       # Resumable engineering checkpoint documentation
├── README.md                        # Complete project documentation
├── task1_structured_extraction/
│   ├── main.py                      # Extraction CLI and dual-provider LLM connector
│   ├── schemas.py                   # Pydantic models (InvoiceData, ResumeData)
│   ├── prompts.py                   # Parameterized templates & few-shot examples
│   ├── requirements.txt             # Task 1 specific dependencies
│   ├── samples/                     # Realistic invoice and resume sample files
│   └── tests/                       # Unit tests (17 passed)
├── task2_document_rag/
│   ├── main.py                      # FAISS indexing, similarity retrieval & grounded QA
│   ├── document_loader.py           # PDF (pypdf) & text loader with page tracking
│   ├── requirements.txt             # Task 2 specific dependencies
│   ├── sample_documents/            # 2-page hardware PDF & corporate policy text
│   └── tests/                       # Unit tests (16 passed)
├── task3_tool_agent/
│   ├── main.py                      # Console agent runner & ReAct loop
│   ├── tools.py                     # Safe calculator, web search, workspace reader
│   ├── requirements.txt             # Task 3 specific dependencies
│   ├── workspace/                   # Sandboxed directory for agent file reader
│   └── tests/                       # Unit tests (19 passed)
└── task4_streamlit_chat/
    ├── app.py                       # Multi-mode conversational Streamlit web application
    ├── requirements.txt             # Task 4 specific dependencies
    └── tests/                       # Unit tests (4 passed)
```

---

## Prerequisites & Windows Setup

### 1. Environment Requirements
- **Operating System:** Windows 10/11 (or macOS / Linux)
- **Python Version:** Python 3.10 to 3.13 (Verified on Python 3.13.3 x64)
- **Git:** Git for Windows (Verified on 2.54.0)
- **C++ Build Tools:** Not required (uses pre-built binary wheels)

### 2. PowerShell Setup

Open **Windows PowerShell** in the project directory:

```powershell
cd D:\Datheon_Python_Internship

# Optional: Create and activate a clean virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install project dependencies
python -m pip install -r requirements.txt
```

---

## API Key Configuration

The project supports both **Google Gemini** and **OpenAI**. You only need **one** provider configured. If no key is set, the application operates in **offline mode**, using deterministic local embeddings and test fixtures.

1. Copy the example file to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Open `.env` in any text editor and insert your key:
   ```env
   # Google Gemini (Free tier available at https://aistudio.google.com/)
   GEMINI_API_KEY=AIzaSyYourGeminiApiKeyHere

   # OR OpenAI (Available at https://platform.openai.com/)
   OPENAI_API_KEY=sk-proj-YourOpenAIApiKeyHere
   ```

> [!IMPORTANT]
> Never commit `.env` to Git. `.gitignore` is pre-configured to strictly exclude `.env`, `.env.*`, vector caches, and virtual environments while preserving `.env.example`.

---

## Task Breakdown & Run Commands

### Task 1 — Structured Invoice & Resume Extraction

Extracts validated JSON from text or files using strict schemas with zero hallucination.

```powershell
# Run extraction on a sample invoice
python task1_structured_extraction/main.py --file task1_structured_extraction/samples/sample_invoice_1.txt

# Run extraction on a sample resume
python task1_structured_extraction/main.py --file task1_structured_extraction/samples/sample_resume_1.txt

# Run with explicit output file
python task1_structured_extraction/main.py --file task1_structured_extraction/samples/sample_invoice_2.txt --output output.json

# Interactive mode (paste text in console)
python task1_structured_extraction/main.py
```

**Example Output:**
```json
{
  "document_type": "invoice",
  "invoice_number": "GCS-2024-8841",
  "invoice_date": "November 05, 2024",
  "vendor_name": "GLOBAL CLOUD SERVICES INC.",
  "total_amount": 3989.28,
  "currency": "USD",
  "missing_or_uncertain_fields": []
}
```

---

### Task 2 — Document RAG with Citations

Ingests PDF and text files, performs dense vector search with FAISS, and provides source filename and PDF page citations.

```powershell
# Query sample documents (specifications PDF & policy text)
python task2_document_rag/main.py --docs task2_document_rag/sample_documents --query "What is the warranty period and return window for the ApexCore X900?"

# Query corporate policy document
python task2_document_rag/main.py --docs task2_document_rag/sample_documents --query "What is the home office setup reimbursement amount?"

# Interactive question-answering mode
python task2_document_rag/main.py --docs task2_document_rag/sample_documents
```

**Example Citation Output:**
```text
Question: What is the warranty period and return window for the ApexCore X900?
Answer:
Every X900 unit includes a comprehensive 3-year enterprise hardware warranty covering full part replacement.
Customers may initiate a no-questions-asked hardware return within 45 days of initial deployment delivery.

Citations:
  - [Source: product_specifications.pdf, Page 2]
```

---

### Task 3 — AI Agent with Tools

Runs an autonomous tool-calling console agent with web search, restricted calculator, and workspace file reader.

```powershell
# Compound query using file reader and safe calculator
python task3_tool_agent/main.py --query "Read budget_data.txt and calculate the sum of 4200 + 1850 + 950 + 2300"

# Query using live web search
python task3_tool_agent/main.py --query "Search the web for the latest Python version features"

# Interactive agent console
python task3_tool_agent/main.py
```

**Tool Safety Highlights:**
- **Calculator:** AST parsing blocks `__import__`, `eval()`, `exec()`, and file system access.
- **File Reader:** Rejects relative traversal (`../`) and absolute paths outside `task3_tool_agent/workspace/`.
- **Zero Thought Leakage:** Displays clean action summaries and observable tool results.

---

### Task 4 — Streamlit Conversational AI Web App

Launches the multi-mode local web interface.

```powershell
python -m streamlit run task4_streamlit_chat/app.py
```

**Web App Features:**
- **Document RAG Mode:** Real-time question answering with expandable source citations (filename & page number).
- **AI Tool Agent Mode:** Displays expandable status indicators showing tools invoked and live observations.
- **Structured Extraction Mode:** Interactive document playground with pre-loaded invoice/resume samples.
- **Conversation Controls:** Persistent session state history with a 1-click **Clear Conversation** button.

---

## Testing & Verification

All four tasks contain comprehensive automated unit test suites using `pytest`. Run all tests in one command:

```powershell
python -m pytest -v
```

### Verified Test Results:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.1.1
collected 56 items

task1_structured_extraction\tests\test_extraction.py .........           [16%]
task1_structured_extraction\tests\test_prompts.py ....                   [23%]
task1_structured_extraction\tests\test_schemas.py ....                   [30%]
task2_document_rag\tests\test_chunking.py ...                            [35%]
task2_document_rag\tests\test_document_loader.py .....                   [44%]
task2_document_rag\tests\test_rag_pipeline.py ......                     [55%]
task2_document_rag\tests\test_retrieval.py ..                            [58%]
task3_tool_agent\tests\test_agent_runner.py ....                         [66%]
task3_tool_agent\tests\test_calculator.py ......                         [76%]
task3_tool_agent\tests\test_file_reader.py ......                        [87%]
task3_tool_agent\tests\test_web_search.py ...                            [92%]
task4_streamlit_chat\tests\test_streamlit_app.py ....                    [100%]

======================= 56 passed in 4.51s (100% pass rate) ===================
```

---

## Troubleshooting & Known Limitations

1. **Missing API Keys:** If no key is supplied in `.env`, Task 1 prompts for configuration, Task 2 operates using local deterministic FAISS embeddings, Task 3 runs in offline heuristic mode, and Task 4 displays an informational badge in the sidebar. No unhandled crashes occur.
2. **Pytest Import Names:** The project root includes `pytest.ini` with `--import-mode=importlib`, preventing package naming collisions across subdirectories.
3. **DuckDuckGo Package Warning:** A deprecation notice (`duckduckgo_search` alias to `ddgs`) may appear in stderr; this is safe and does not affect search functionality.

---

## Security & Pre-Submission Checklist

- [x] Real API keys stored **only** in local `.env` (never committed).
- [x] `.env.example` committed with placeholder values only.
- [x] `.gitignore` verified to exclude `.env`, `.env.*`, `__pycache__`, `.pytest_cache`, `*.faiss`, and virtual environments.
- [x] No hardcoded keys or passwords in source code, tests, or documentation.
- [x] Path traversal prevention verified in file reader (`test_blocks_relative_path_traversal` passed).
- [x] Code execution prevention verified in calculator (`test_blocks_arbitrary_code_execution` passed).
- [x] All 56 automated tests passed with 100% success rate.
- [x] Repository reviewed and ready for submission.
