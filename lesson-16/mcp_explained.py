"""
Lesson 16: Model Context Protocol (MCP) Explained
Topics: MCP architecture, building a basic MCP server, tools vs resources vs prompts
Run: python mcp_explained.py   (demonstrates the concepts with a mock MCP server)
Requires: pip install anthropic mcp
"""

# ─────────────────────────────────────────────
# PART 1: What MCP Is (conceptual demo)
# ─────────────────────────────────────────────
# MCP (Model Context Protocol) is an open standard by Anthropic
# that lets LLMs connect to external tools, data sources, and services
# in a standardized way — like USB-C for AI integrations.
#
# Architecture:
#   Host (Claude Desktop / your app)
#     └── MCP Client
#           └── MCP Server (your code)
#                 ├── Tools      → functions the LLM can call
#                 ├── Resources  → data the LLM can read (files, DB rows)
#                 └── Prompts    → reusable prompt templates
# ─────────────────────────────────────────────

import json
import asyncio

# ─────────────────────────────────────────────
# PART 2: A Minimal MCP Server
# This is a real, runnable MCP server you can connect to Claude Desktop
# ─────────────────────────────────────────────

MCP_SERVER_CODE = '''
# Save this as: server.py
# Run with:    python server.py
# Then add to Claude Desktop's claude_desktop_config.json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import json
import datetime

app = Server("my-first-mcp-server")

# ── TOOLS: functions the LLM can call ──

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_time",
            description="Get the current date and time",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="calculate",
            description="Evaluate a mathematical expression",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"]
            }
        ),
        types.Tool(
            name="save_note",
            description="Save a note to memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["title", "content"]
            }
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "get_time":
        return [types.TextContent(type="text", text=str(datetime.datetime.now()))]
    
    elif name == "calculate":
        import math
        result = eval(arguments["expression"], {"math": math, "__builtins__": {}})
        return [types.TextContent(type="text", text=str(result))]
    
    elif name == "save_note":
        # In production: save to a database or file
        note = {"title": arguments["title"], "content": arguments["content"]}
        return [types.TextContent(type="text", text=f"Saved: {json.dumps(note)}")]
    
    raise ValueError(f"Unknown tool: {name}")


# ── RESOURCES: data the LLM can read ──

@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="notes://all",
            name="All Notes",
            description="Read all saved notes",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "notes://all":
        # In production: fetch from database
        return json.dumps([
            {"title": "Meeting notes", "content": "Discuss Q3 roadmap"},
            {"title": "Ideas", "content": "Build an MCP server for Notion"},
        ])
    raise ValueError(f"Unknown resource: {uri}")


# ── PROMPTS: reusable prompt templates ──

@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="summarize_notes",
            description="Summarize all notes into action items",
            arguments=[]
        )
    ]

@app.get_prompt()
async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
    if name == "summarize_notes":
        return types.GetPromptResult(
            description="Summarize notes into action items",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text="Please read the notes resource and summarize them as a bulleted action item list."
                    )
                )
            ]
        )
    raise ValueError(f"Unknown prompt: {name}")


# ── Entry point ──
if __name__ == "__main__":
    asyncio.run(stdio_server(app))
'''

print("=== MCP Server Code ===")
print("This is a complete, runnable MCP server.")
print("Save the code below as 'server.py' and run it:\n")
print(MCP_SERVER_CODE)


# ─────────────────────────────────────────────
# PART 3: Claude Desktop Config
# ─────────────────────────────────────────────

config = {
    "mcpServers": {
        "my-first-mcp-server": {
            "command": "python",
            "args": ["path/to/server.py"],
            "env": {}
        }
    }
}

print("\n=== Claude Desktop Config ===")
print("Add this to: ~/Library/Application Support/Claude/claude_desktop_config.json")
print("(Windows: %APPDATA%\\Claude\\claude_desktop_config.json)\n")
print(json.dumps(config, indent=2))


# ─────────────────────────────────────────────
# PART 4: Calling your MCP server via the Anthropic API
# ─────────────────────────────────────────────

ANTHROPIC_CALL_CODE = '''
import anthropic

client = anthropic.Anthropic()

# When connected via Claude Desktop, tools appear automatically.
# When calling via API, you pass tools manually (same format as Lesson 12).
# MCP just standardizes HOW those tools are discovered and served.

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[
        {
            "name": "get_time",
            "description": "Get the current date and time",
            "input_schema": {"type": "object", "properties": {}}
        }
    ],
    messages=[{"role": "user", "content": "What time is it?"}]
)
print(response.content)
'''

print("\n=== Using Your Server via the API ===")
print(ANTHROPIC_CALL_CODE)

print("\n=== MCP Cheat Sheet ===")
cheatsheet = {
    "Tools":     "Functions the LLM can call (write operations, computations)",
    "Resources": "Data the LLM can read (files, DB rows, API responses)",
    "Prompts":   "Reusable prompt templates exposed to the host",
    "Transport": "stdio (local) or HTTP/SSE (remote)",
    "Config":    "claude_desktop_config.json connects servers to Claude Desktop",
}
for k, v in cheatsheet.items():
    print(f"  {k:<12} → {v}")
