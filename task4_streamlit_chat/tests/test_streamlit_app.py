"""Unit tests for Task 4 Streamlit application helpers, API status checks, and streaming."""

import os
from task4_streamlit_chat.app import check_api_status, stream_text_chunks


def test_check_api_status_offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    info = check_api_status()
    assert info["provider"] == "offline"
    assert info["configured"] is False
    assert "Offline" in info["status"]


def test_check_api_status_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "real-gemini-key-placeholder")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    info = check_api_status()
    assert info["provider"] == "gemini"
    assert info["configured"] is True


def test_check_api_status_openai(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai-key-placeholder")

    info = check_api_status()
    assert info["provider"] == "openai"
    assert info["configured"] is True


def test_stream_text_chunks():
    sample = "Hello world this is a test"
    stream = list(stream_text_chunks(sample))
    reconstructed = "".join(stream)
    assert reconstructed == sample
    assert len(stream) == 6
