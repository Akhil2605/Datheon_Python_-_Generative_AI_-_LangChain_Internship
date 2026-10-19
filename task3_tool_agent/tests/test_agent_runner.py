"""Unit tests for AgentRunner orchestration, tool selection, and ReAct loop."""

from unittest.mock import patch, MagicMock
from task3_tool_agent.main import AgentRunner


def test_agent_runner_empty_query():
    agent = AgentRunner(provider="offline")
    assert "Error: User query cannot be empty." in agent.run("   ")


def test_agent_runner_offline_math():
    agent = AgentRunner(provider="offline")
    output = agent.run("Calculate 125 * 8", verbose=False)
    assert "1000" in output


def test_agent_runner_offline_file_read():
    agent = AgentRunner(provider="offline")
    output = agent.run("Read the contents of project_brief.txt from workspace", verbose=False)
    assert "Project Antigravity AI" in output


def test_agent_runner_with_mocked_llm_tool_loop():
    agent = AgentRunner(provider="mock")
    agent.provider = "mock"

    step1_response = "Action: calculator\nAction Input: 25 * 4"
    step2_response = "Final Answer: The calculated total is 100."

    with patch.object(agent, "_call_llm", side_effect=[step1_response, step2_response]):
        output = agent.run("What is 25 times 4?", verbose=False)
        assert "The calculated total is 100." in output
