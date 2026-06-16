# Lesson 12: Tool Use and Function Calling

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python tool_use.py
```

## What This Code Covers
- Defining tools using JSON Schema (name, description, input_schema)
- Routing tool calls to Python functions via `execute_tool()`
- Handling `stop_reason == "tool_use"` in the response loop
- **Parallel tool calls** — model calls multiple tools in one shot
- Feeding tool results back into the conversation

## Key Pattern
```
User message
  → Model responds with tool_use blocks
  → You execute each tool
  → Send tool_result blocks back
  → Model gives final answer
```

## How to Add a New Tool
1. Add a JSON Schema entry to `TOOLS`
2. Write the Python function
3. Add an `elif` branch in `execute_tool()`
