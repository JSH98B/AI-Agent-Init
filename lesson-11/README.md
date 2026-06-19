# Lesson 11: What is an AI Agent? — The ReAct Loop

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python react_agent.py
```

## What This Code Covers
- **Tool definitions** in Anthropic's JSON schema format
- **Tool executor** — the Python code that actually runs each tool
- **ReAct loop**: Perceive → Think → Act → Observe → repeat until done
- **`stop_reason == "end_turn"`** — how to detect when the agent is finished

## How to Extend This Agent
1. Add a new entry to `TOOLS` with `name`, `description`, `input_schema`
2. Add a matching `elif tool_name == "your_tool"` block in `execute_tool()`
3. That's it — the model automatically learns to use new tools from their descriptions

## Real-World Additions
- Web search tool (SerpAPI, Tavily)
- File read/write tools
- Database query tool
- Email/calendar tools via MCP
