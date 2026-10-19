"""Task 3: Console-based AI Agent with Tools.

Equipped with:
1. DuckDuckGo Web Search
2. Restricted Safe Calculator (AST-based)
3. Workspace-Restricted Local File Reader

Features tool invocation, concise action summaries, observable results,
and zero exposure of internal private chain-of-thought.
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Ensure parent directory is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from task3_tool_agent.tools import (
    AVAILABLE_TOOLS,
    calculate_expression,
    search_duckduckgo,
    read_workspace_file,
    WORKSPACE_DIR
)

load_dotenv(dotenv_path=ROOT_DIR / ".env")


AGENT_SYSTEM_PROMPT = """You are an efficient, security-focused AI research agent.
You have access to the following three tools:
1. calculator(expression: str) -> Evaluates mathematical arithmetic expressions safely (e.g. '25 * 4 + 10').
2. web_search(query: str) -> Searches DuckDuckGo for live facts, recent news, or general web information.
3. read_workspace_file(filename: str) -> Reads text files strictly inside your designated workspace directory.

Instructions:
- When a question requires external facts, computation, or file reading, call the appropriate tool.
- To call a tool, respond with:
  Action: <tool_name>
  Action Input: <input_string>
- When you have enough information to answer the user's question, respond with:
  Final Answer: <your concise, complete answer>
- Do NOT expose private internal reasoning, scratchpads, or chain-of-thought.
- Output ONLY the Action/Action Input or Final Answer."""


class AgentRunner:
    """Orchestrates LLM tool selection, execution, and response synthesis."""

    def __init__(self, provider: Optional[str] = None):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        if provider:
            self.provider = provider
        elif self.gemini_key and self.gemini_key.strip() != "your_gemini_api_key_here":
            self.provider = "gemini"
        elif self.openai_key and self.openai_key.strip() != "your_openai_api_key_here":
            self.provider = "openai"
        else:
            self.provider = "offline"

        if self.provider == "gemini":
            from google import genai
            self.gemini_client = genai.Client(api_key=self.gemini_key)
            self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        elif self.provider == "openai":
            import openai
            self.openai_client = openai.OpenAI(api_key=self.openai_key)
            self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def _call_llm(self, prompt: str) -> str:
        """Invokes configured LLM with prompt."""
        if self.provider == "gemini":
            from google.genai import types
            resp = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=AGENT_SYSTEM_PROMPT,
                    temperature=0.0
                )
            )
            return resp.text.strip() if resp.text else ""

        elif self.provider == "openai":
            resp = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return resp.choices[0].message.content.strip()

        raise RuntimeError("No external LLM configured.")

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Executes the specified tool with safe error containment."""
        name = tool_name.strip().lower()
        inp = tool_input.strip()

        if name in ["calculator", "safe_calculator", "calc"]:
            return calculate_expression(inp)
        elif name in ["web_search", "duckduckgo_search", "search"]:
            return search_duckduckgo(inp)
        elif name in ["read_workspace_file", "file_reader", "read_file"]:
            return read_workspace_file(inp)
        else:
            return f"Error: Unknown tool '{tool_name}'. Available tools: calculator, web_search, read_workspace_file."

    def _offline_heuristic_plan(self, query: str) -> List[Dict[str, str]]:
        """Determines tool calls for test suites and offline mode without API keys."""
        lower = query.lower()
        actions = []

        # File reading heuristic
        if "workspace" in lower or ".txt" in lower or "brief" in lower or "budget" in lower or "file" in lower:
            # Extract possible filename
            match = re.search(r"([\w\-]+\.txt)", query)
            fname = match.group(1) if match else ("project_brief.txt" if "brief" in lower else "budget_data.txt")
            actions.append({"tool": "read_workspace_file", "input": fname})

        # Math / calculation heuristic
        math_match = re.search(r"([\d\.\s\+\-\*\/\(\)\%\^]+)", query)
        if any(op in query for op in ["+", "-", "*", "/", "sqrt", "calculate", "sum", "total"]) and math_match:
            expr = math_match.group(1).strip()
            if any(c.isdigit() for c in expr):
                actions.append({"tool": "calculator", "input": expr})

        # Web search heuristic
        if not actions or "search" in lower or "online" in lower or "latest" in lower or "who is" in lower:
            actions.append({"tool": "web_search", "input": query})

        return actions

    def run(self, user_query: str, max_iterations: int = 5, verbose: bool = True) -> str:
        """Executes agent loop to answer a compound research or calculation question."""
        if not user_query or not user_query.strip():
            return "Error: User query cannot be empty."

        conversation_log = [f"User Question: {user_query.strip()}"]

        if self.provider == "offline":
            if verbose:
                print(f"[Agent Mode] Offline / Local Heuristic Dispatcher")

            actions = self._offline_heuristic_plan(user_query)
            tool_results = []

            for act in actions:
                tool_name = act["tool"]
                tool_input = act["input"]

                if verbose:
                    print(f"Action: {tool_name}")
                    print(f"Action Input: {tool_input}")

                res = self._execute_tool(tool_name, tool_input)
                tool_results.append(f"Result from {tool_name}('{tool_input}'):\n{res}")

                if verbose:
                    print(f"Observation: {res}\n")

            # Final answer synthesis
            final_answer = "\n\n".join(tool_results)
            if verbose:
                print(f"Final Answer:\n{final_answer}")
            return final_answer

        # Live LLM Agent ReAct Loop
        for step in range(max_iterations):
            prompt = "\n\n".join(conversation_log)
            try:
                llm_output = self._call_llm(prompt)
            except Exception as exc:
                err_msg = f"Agent API Error: {exc}"
                if verbose:
                    print(err_msg)
                return err_msg

            # Check if final answer is reached
            if "Final Answer:" in llm_output:
                final_part = llm_output.split("Final Answer:", 1)[1].strip()
                if verbose:
                    print(f"Final Answer:\n{final_part}")
                return final_part

            # Parse Action and Action Input
            action_match = re.search(r"Action:\s*([a-zA-Z0-9_\-]+)", llm_output)
            input_match = re.search(r"Action Input:\s*(.+)", llm_output, re.DOTALL)

            if action_match and input_match:
                tool_name = action_match.group(1).strip()
                tool_input = input_match.group(1).strip().split("\n")[0]

                if verbose:
                    print(f"Action: {tool_name}")
                    print(f"Action Input: {tool_input}")

                observation = self._execute_tool(tool_name, tool_input)

                if verbose:
                    print(f"Observation: {observation}\n")

                conversation_log.append(f"Action: {tool_name}\nAction Input: {tool_input}\nObservation: {observation}")
            else:
                # LLM output direct answer without prefix
                if verbose:
                    print(f"Final Answer:\n{llm_output}")
                return llm_output

        return "Agent reached maximum tool invocation limit without producing a final answer."


def main():
    parser = argparse.ArgumentParser(description="Task 3 AI Tool Agent CLI")
    parser.add_argument("--query", "-q", help="Single research or compound calculation query")
    parser.add_argument("--provider", choices=["gemini", "openai", "offline"], help="Force LLM provider")

    args = parser.parse_args()
    agent = AgentRunner(provider=args.provider)

    if args.query:
        print("=" * 60)
        print(f"Query: {args.query}")
        print("=" * 60)
        agent.run(args.query, verbose=True)
    else:
        print("Datheon AI Tool Agent Initialized.")
        print("Available tools: calculator, web_search, read_workspace_file")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                q = input("Agent Prompt > ").strip()
                if not q or q.lower() in ["exit", "quit"]:
                    break
                agent.run(q, verbose=True)
                print("-" * 60)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
