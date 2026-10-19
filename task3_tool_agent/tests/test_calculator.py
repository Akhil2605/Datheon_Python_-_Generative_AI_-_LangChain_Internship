"""Unit tests for AST-based safe calculator (valid ops, safety bounds, anti-code-exec)."""

from task3_tool_agent.tools import calculate_expression


def test_basic_arithmetic():
    assert calculate_expression("2 + 3 * 4") == "14"
    assert calculate_expression("(10 - 2) / 4") == "2"
    assert calculate_expression("2 ** 5") == "32"
    assert calculate_expression("17 % 5") == "2"
    assert calculate_expression("19 // 4") == "4"


def test_math_functions_and_constants():
    assert calculate_expression("sqrt(144)") == "12"
    assert calculate_expression("abs(-42)") == "42"
    assert calculate_expression("round(3.14159, 2)") == "3.14"
    assert "3.14" in calculate_expression("pi")


def test_division_by_zero():
    res = calculate_expression("100 / 0")
    assert "Division by zero" in res


def test_blocks_arbitrary_code_execution():
    # Attempting os execution
    res1 = calculate_expression("__import__('os').system('dir')")
    assert "Disallowed" in res1 or "Calculation Error" in res1

    # Attempting eval
    res2 = calculate_expression("eval('1 + 1')")
    assert "not in the permitted math whitelist" in res2 or "Calculation Error" in res2

    # Attempting exec
    res3 = calculate_expression("exec('x = 1')")
    assert "Calculation Error" in res3

    # Attempting file open
    res4 = calculate_expression("open('secret.txt').read()")
    assert "not in the permitted math whitelist" in res4 or "Calculation Error" in res4


def test_blocks_invalid_syntax():
    res = calculate_expression("25 ++* 4")
    assert "Calculation Error" in res


def test_empty_expression():
    res = calculate_expression("   ")
    assert "Error: Expression cannot be empty." in res
