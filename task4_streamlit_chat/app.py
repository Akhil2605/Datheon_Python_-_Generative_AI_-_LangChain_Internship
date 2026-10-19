"""Task 4: Streamlit Conversational AI Web App.

A responsive, modular local web interface integrating:
- Task 2 Document RAG with source & page citations
- Task 3 AI Tool Agent with web search, calculator, and workspace file reader
- Task 1 Structured Extraction interface
- Session state conversation history, streaming output, and new-conversation action.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Generator
import streamlit as st
from dotenv import load_dotenv

# Ensure parent directory is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from task2_document_rag.main import RAGPipeline
from task3_tool_agent.main import AgentRunner
from task1_structured_extraction.main import extract_structured_data

load_dotenv(dotenv_path=ROOT_DIR / ".env")


def check_api_status() -> Dict[str, Any]:
    """Inspects API keys without exposing secrets."""
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    has_gemini = bool(gemini_key and gemini_key.strip() != "your_gemini_api_key_here")
    has_openai = bool(openai_key and openai_key.strip() != "your_openai_api_key_here")

    if has_gemini:
        return {"provider": "gemini", "status": "Online (Google Gemini)", "configured": True}
    elif has_openai:
        return {"provider": "openai", "status": "Online (OpenAI)", "configured": True}
    else:
        return {"provider": "offline", "status": "Offline (Local Deterministic)", "configured": False}


@st.cache_resource(show_spinner="Indexing knowledge base...")
def get_rag_pipeline(provider_name: str) -> RAGPipeline:
    """Caches RAG pipeline so documents are only indexed once."""
    pipeline = RAGPipeline(provider=provider_name)
    sample_dir = ROOT_DIR / "task2_document_rag" / "sample_documents"
    if sample_dir.exists():
        pipeline.load_and_index(sample_dir)
    return pipeline


def stream_text_chunks(text: str) -> Generator[str, None, None]:
    """Simulates streaming output for responsive UI rendering."""
    words = text.split(" ")
    for idx, word in enumerate(words):
        yield word + (" " if idx < len(words) - 1 else "")
        time.sleep(0.015)


def init_session_state():
    """Initializes Streamlit session state keys."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "Document RAG"


def main():
    st.set_page_config(
        page_title="Datheon Generative AI Assistant",
        page_icon="🤖",
        layout="wide"
    )

    init_session_state()
    api_info = check_api_status()

    # --- Sidebar ---
    with st.sidebar:
        st.title("⚙️ AI Configuration")

        if api_info["configured"]:
            st.success(f"Provider: {api_info['status']}")
        else:
            st.warning("⚠️ No API Key Configured")
            st.info("Running in **Offline Local Mode**. Set `GEMINI_API_KEY` or `OPENAI_API_KEY` in `.env` for live cloud models.")

        st.divider()

        st.subheader("Select Operating Mode")
        mode = st.radio(
            "Choose Capability:",
            ["Document RAG", "AI Tool Agent", "Structured Extraction"],
            index=0
        )

        if mode != st.session_state.current_mode:
            st.session_state.current_mode = mode
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # Action: Clear Conversation
        if st.button("🧹 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.markdown(
            "**Datheon Internship Project**  \n"
            "• Python 3.13  \n"
            "• LangChain & FAISS  \n"
            "• DuckDuckGo & AST Tools  \n"
            "• Gemini & OpenAI Support"
        )

    # --- Header ---
    st.title("🤖 Datheon Generative AI & Tool Platform")

    # --- Mode 1: Document RAG ---
    if mode == "Document RAG":
        st.caption("Ask questions about company policies or hardware technical specifications. Citations include filename and PDF page numbers.")

        rag_pipeline = get_rag_pipeline(api_info["provider"])

        # Display history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "citations" in msg and msg["citations"]:
                    with st.expander("📄 Source Citations"):
                        for c in msg["citations"]:
                            st.markdown(f"- **{c['source']}** (Page {c['page']}): *{c['snippet']}*")

        # Chat Input
        if prompt := st.chat_input("Ask a question about the documents (e.g. 'What is the return policy for EdgeServer X900?'):"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching document index..."):
                    response = rag_pipeline.query(prompt)

                citations_data = [
                    {"source": c.source, "page": c.page, "snippet": c.snippet}
                    for c in response.citations
                ]

                # Stream response
                placeholder = st.empty()
                full_answer = ""
                for chunk in stream_text_chunks(response.answer):
                    full_answer += chunk
                    placeholder.markdown(full_answer + "▌")
                placeholder.markdown(full_answer)

                if citations_data:
                    with st.expander("📄 Source Citations", expanded=True):
                        for c in citations_data:
                            page_lbl = f"Page {c['page']}" if c["page"] else "Text"
                            st.markdown(f"- **{c['source']}** ({page_lbl}): *{c['snippet']}*")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "citations": citations_data
            })

    # --- Mode 2: AI Tool Agent ---
    elif mode == "AI Tool Agent":
        st.caption("AI Agent with live DuckDuckGo web search, AST safe calculator, and workspace file reader.")

        agent = AgentRunner(provider=api_info["provider"])

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Enter compound research or calculation task (e.g. 'Read budget_data.txt and calculate total expenses'):"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.status("Agent orchestrating tools...", expanded=True) as status:
                    st.write(f"Executing task with available tools: `calculator`, `web_search`, `read_workspace_file`")
                    answer = agent.run(prompt, verbose=False)
                    status.update(label="Task Complete!", state="complete", expanded=False)

                placeholder = st.empty()
                full_text = ""
                for chunk in stream_text_chunks(answer):
                    full_text += chunk
                    placeholder.markdown(full_text + "▌")
                placeholder.markdown(full_text)

            st.session_state.messages.append({"role": "assistant", "content": answer})

    # --- Mode 3: Structured Extraction ---
    elif mode == "Structured Extraction":
        st.caption("Convert unstructured invoice or resume text into validated, schema-compliant JSON.")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Raw Text Input")
            doc_type_choice = st.selectbox("Document Type:", ["Auto-Detect", "Invoice", "Resume"])
            sample_choice = st.selectbox(
                "Or load a sample document:",
                ["None", "Sample Invoice 1 (Cloud)", "Sample Invoice 2 (Freelance)", "Sample Resume 1 (Senior SWE)", "Sample Resume 2 (Junior Frontend)"]
            )

            preset_text = ""
            if sample_choice == "Sample Invoice 1 (Cloud)":
                preset_text = (ROOT_DIR / "task1_structured_extraction" / "samples" / "sample_invoice_1.txt").read_text(encoding="utf-8")
            elif sample_choice == "Sample Invoice 2 (Freelance)":
                preset_text = (ROOT_DIR / "task1_structured_extraction" / "samples" / "sample_invoice_2.txt").read_text(encoding="utf-8")
            elif sample_choice == "Sample Resume 1 (Senior SWE)":
                preset_text = (ROOT_DIR / "task1_structured_extraction" / "samples" / "sample_resume_1.txt").read_text(encoding="utf-8")
            elif sample_choice == "Sample Resume 2 (Junior Frontend)":
                preset_text = (ROOT_DIR / "task1_structured_extraction" / "samples" / "sample_resume_2.txt").read_text(encoding="utf-8")

            raw_input = st.text_area("Input Document Text:", value=preset_text, height=350)
            extract_clicked = st.button("🚀 Extract Structured JSON", use_container_width=True)

        with col2:
            st.subheader("Validated JSON Output")
            if extract_clicked:
                if not raw_input.strip():
                    st.error("Please provide document text to extract.")
                else:
                    target_type = None
                    if doc_type_choice == "Invoice":
                        target_type = "invoice"
                    elif doc_type_choice == "Resume":
                        target_type = "resume"

                    with st.spinner("Extracting & validating schemas..."):
                        try:
                            # Note: In offline mode, if neither key is configured, prompt user
                            if not api_info["configured"]:
                                st.info("Note: Real-time LLM structured extraction requires GEMINI_API_KEY or OPENAI_API_KEY in .env.")
                            result = extract_structured_data(raw_input, doc_type=target_type)
                            st.json(result.model_dump())
                        except Exception as exc:
                            st.error(f"Extraction Error: {exc}")


if __name__ == "__main__":
    main()
