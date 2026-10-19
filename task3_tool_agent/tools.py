"""Task 3 Tools: DuckDuckGo Web Search, Safe Calculator, and Restricted Workspace File Reader.

Enforces strict security:
- Calculator uses AST parsing to prevent arbitrary code execution.
- File reader restricts access strictly to the task3_tool_agent/workspace directory.
- DuckDuckGo search provides resilient error handling for network/API limits.
"""

import ast
import math
import operator
from pathlib import Path
from typing import Dict, Any, Union
from langchain_core.tools import tool

# Designated safe workspace directory
WORKSPACE_DIR = (Path(__file__).resolve().parent / "workspace").resolve()


# ==============================================================================
# 1. Safe Calculator (AST Whitelist Based — No eval())
# ==============================================================================

# Whitelisted operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Whitelisted math functions and constants
_MATH_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
}

_MATH_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluates an AST node strictly against whitelisted mathematical types."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    elif isinstance(node, ast.Name):
        if node.id in _MATH_CONSTS:
            return _MATH_CONSTS[node.id]
        raise ValueError(f"Variable or identifier '{node.id}' is not an allowed mathematical constant.")

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)

        # Protect against excessive exponents
        if op_type is ast.Pow and (right > 10000 or right < -10000):
            raise OverflowError("Exponent magnitude is too large for safe calculation.")

        try:
            return _OPERATORS[op_type](left, right)
        except ZeroDivisionError:
            raise ZeroDivisionError("Division by zero in mathematical expression.")

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_ast_node(node.operand)
        return _OPERATORS[op_type](operand)

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _MATH_FUNCS:
            func_name = getattr(node.func, "id", "unnamed")
            raise ValueError(f"Function '{func_name}' is not in the permitted math whitelist.")
        args = [_eval_ast_node(arg) for arg in node.args]
        return _MATH_FUNCS[node.func.id](*args)

    else:
        raise ValueError(f"Disallowed expression syntax: {type(node).__name__}. Arbitrary code execution is blocked.")


def calculate_expression(expression: str) -> str:
    """Safely evaluates an arithmetic expression using AST parsing."""
    if not expression or not expression.strip():
        return "Error: Expression cannot be empty."

    cleaned = expression.strip()
    try:
        parsed = ast.parse(cleaned, mode="eval")
        result = _eval_ast_node(parsed.body)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except (SyntaxError, ValueError, OverflowError, ZeroDivisionError) as err:
        return f"Calculation Error: {err}"
    except Exception as err:
        return f"Calculation Error: Invalid expression ({err})"


@tool("calculator")
def safe_calculator_tool(expression: str) -> str:
    """Evaluates mathematical arithmetic expressions safely.
    Supports +, -, *, /, //, %, **, sqrt, sin, cos, tan, log, round, abs, and constants pi, e.
    Example input: '25 * 4 + sqrt(144)'
    """
    return calculate_expression(expression)


# ==============================================================================
# 2. DuckDuckGo Web Search Tool
# ==============================================================================

def search_duckduckgo(query: str, max_results: int = 4) -> str:
    """Performs web search via DuckDuckGo and formats the results."""
    if not query or not query.strip():
        return "Error: Search query cannot be empty."

    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query.strip(), max_results=max_results))

        if not results:
            return f"No web search results found for query: '{query}'."

        formatted_lines = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "No Title")
            snippet = r.get("body", "No description")
            url = r.get("href", "")
            formatted_lines.append(f"[{i}] {title}\nSummary: {snippet}\nURL: {url}")

        return "\n\n".join(formatted_lines)

    except Exception as exc:
        return f"Search Error: Unable to complete web search ({exc}). Please try again."


@tool("web_search")
def duckduckgo_search_tool(query: str) -> str:
    """Searches the live web using DuckDuckGo for recent information, facts, or news.
    Example input: 'latest Python release version'
    """
    return search_duckduckgo(query)


# ==============================================================================
# 3. Workspace-Restricted Local File Reader
# ==============================================================================

def read_workspace_file(filename: str) -> str:
    """Reads a text file strictly restricted to the project's designated workspace directory."""
    if not filename or not filename.strip():
        return "Error: Filename cannot be empty."

    # Ensure workspace directory exists
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    # Normalize and resolve requested path
    target = (WORKSPACE_DIR / filename.strip()).resolve()

    # Path traversal protection
    try:
        # target must be within WORKSPACE_DIR
        target.relative_to(WORKSPACE_DIR)
    except ValueError:
        return (
            "Access Denied: Path traversal detected. "
            "You may only read files located inside the designated workspace directory."
        )

    if not target.exists():
        return f"File Error: File '{filename}' not found in workspace."

    if not target.is_file():
        return f"File Error: '{filename}' is a directory, not a readable file."

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return f"File '{filename}' exists but is empty."
        return content
    except Exception as exc:
        return f"File Error: Could not read file '{filename}' ({exc})."


@tool("read_workspace_file")
def workspace_file_reader_tool(filename: str) -> str:
    """Reads the contents of a local file strictly located within the agent's designated workspace directory.
    Access outside this workspace is blocked.
    Example input: 'project_notes.txt'
    """
    return read_workspace_file(filename)


# Tool registry map for dispatching
AVAILABLE_TOOLS = {
    "calculator": safe_calculator_tool,
    "web_search": duckduckgo_search_tool,
    "read_workspace_file": workspace_file_reader_tool,
}
