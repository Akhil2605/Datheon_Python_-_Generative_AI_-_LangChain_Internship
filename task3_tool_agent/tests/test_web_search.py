"""Unit tests for DuckDuckGo web search tool."""

from unittest.mock import patch
from task3_tool_agent.tools import search_duckduckgo


def test_empty_search_query():
    res = search_duckduckgo("   ")
    assert "Error: Search query cannot be empty." in res


def test_mocked_web_search():
    mock_results = [
        {"title": "Python 3.13 Released", "body": "Python 3.13 includes a free-threaded build mode.", "href": "https://python.org"},
        {"title": "Python Documentation", "body": "Official documentation for Python 3.13.", "href": "https://docs.python.org"}
    ]

    with patch("duckduckgo_search.DDGS.text", return_value=mock_results):
        res = search_duckduckgo("Python 3.13")
        assert "Python 3.13 Released" in res
        assert "https://python.org" in res
        assert "free-threaded" in res


def test_search_network_exception_handling():
    with patch("duckduckgo_search.DDGS.text", side_effect=Exception("Connection timed out")):
        res = search_duckduckgo("test query")
        assert "Search Error" in res
        assert "Connection timed out" in res
