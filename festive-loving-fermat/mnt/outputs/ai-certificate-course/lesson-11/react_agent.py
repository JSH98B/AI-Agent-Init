"""
Lesson 11: What is an AI Agent? — The ReAct Loop
Topics: Perceive → Think → Act loop, tool calling, agent with real tools
Run: python react_agent.py
Requires: pip install anthropic
"""

import anthropic
import json
import math
from datetime import datetime

client = anthropic.Anthropic()

# ─────────────────────────────────────────────
# PART 1: Define Tools (functions the agent can call)
# ─────────────────────────────────────────────

# Tool definitions in Anthropic's JSON schema format
TOOLS = [
    {
        "name": "calculator",
        "description": "Perform mathematical calculations. Use for any arithmetic, including complex expressions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression, e.g. '2 ** 10' or 'math.sqrt(144)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "word_counter",
        "description": "Count the number of words in a text string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to count words in.",
                }
            },
            "required": ["text"],
        },
    },
]


# ─────────────────────────────────────────────
# PART 2: Tool Executor (the "Act" in ReAct)
# ─────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call and return the result as a string."""
    try:
        if tool_name == "calculator":
            # Safe eval using only math functions
            safe_globals = {"math": math, "__builtins__": {}}
            result = eval(tool_input["expression"], safe_globals)
            return str(result)

        elif tool_name == "get_current_time":
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        elif tool_name == "word_counter":
            count = len(tool_input["text"].split())
            return f"{count} words"

        else:
            return f"Error: Unknown tool '{tool_name}'"

    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


# ─────────────────────────────────────────────
# PART 3: The ReAct Agent Loop
# Perceive → Think → Act → Observe → Repeat
# ─────────────────────────────────────────────

def run_agent(user_task: str, max_steps: int = 10, verbose: bool = True) -> str:
    """
    Run the ReAct agent loop until the model stops calling tools.

    Each iteration:
    1. PERCEIVE: Send current messages (including tool results) to the model
    2. THINK:    Model reasons about what to do next
    3. ACT:      Model calls a tool (or stops with final answer)
    4. OBSERVE:  We execute the tool and add result to messages
    """
    messages = [{"role": "user", "content": user_task}]
    step = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {user_task}")
        print('='*60)

    while step < max_steps:
        step += 1

        # THINK — ask the model what to do
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        if verbose:
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\n[THINK] {block.text}")
                elif block.type == "tool_use":
                    print(f"\n[ACT]   Tool: {block.name}")
                    print(f"        Input: {json.dumps(block.input)}")

        # Check if agent is done (no more tool calls)
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "Task complete."

        # ACT + OBSERVE — execute all tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)

                if verbose:
                    print(f"\n[OBS]   Result: {result}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        # Add tool results to message history
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    return "Max steps reached."


# ─────────────────────────────────────────────
# PART 4: Run example tasks
# ─────────────────────────────────────────────

tasks = [
    "What is 2 to the power of 16, and what time is it right now?",
    "How many words are in this sentence: 'The quick brown fox jumps over the lazy dog'? Then calculate that number squared.",
]

for task in tasks:
    final_answer = run_agent(task, verbose=True)
    print(f"\n[FINAL] {final_answer}")
    print()
