"""Unit tests for workspace-restricted file reader and path traversal prevention."""

from task3_tool_agent.tools import read_workspace_file


def test_read_valid_workspace_file():
    content = read_workspace_file("project_brief.txt")
    assert "Project Antigravity AI" in content
    assert "Sarah Jenkins" in content


def test_read_budget_data():
    content = read_workspace_file("budget_data.txt")
    assert "Cloud Compute" in content
    assert "4200" in content


def test_blocks_relative_path_traversal():
    res = read_workspace_file("../task1_structured_extraction/schemas.py")
    assert "Access Denied: Path traversal detected" in res

    res2 = read_workspace_file("../../requirements.txt")
    assert "Access Denied: Path traversal detected" in res2


def test_blocks_absolute_path_traversal():
    res = read_workspace_file("C:/Windows/System32/drivers/etc/hosts")
    assert "Access Denied: Path traversal detected" in res


def test_missing_workspace_file():
    res = read_workspace_file("non_existent_file_xyz.txt")
    assert "File Error: File 'non_existent_file_xyz.txt' not found" in res


def test_empty_filename():
    res = read_workspace_file("   ")
    assert "Error: Filename cannot be empty." in res
