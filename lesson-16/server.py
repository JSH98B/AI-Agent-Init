"""
Lesson 16: Your First MCP Server — save this file and run it!
Run: python server.py
Requires: pip install mcp
Then add to Claude Desktop config (see README.md)
"""

import json
import asyncio
import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("my-first-mcp-server")

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
                    "expression": {"type": "string", "description": "Math expression, e.g. '2 ** 10'"}
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
                    "title":   {"type": "string"},
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
        note = {"title": arguments["title"], "content": arguments["content"]}
        return [types.TextContent(type="text", text=f"Saved: {json.dumps(note)}")]
    raise ValueError(f"Unknown tool: {name}")

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
        return json.dumps([
            {"title": "Meeting notes", "content": "Discuss Q3 roadmap"},
            {"title": "Ideas", "content": "Build an MCP server for Notion"},
        ])
    raise ValueError(f"Unknown resource: {uri}")

if __name__ == "__main__":
    asyncio.run(stdio_server(app))
