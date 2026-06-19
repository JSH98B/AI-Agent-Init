"""
Lesson 17: Building MCP Servers — Tools, Resources, and Prompts in depth
Topics: Full-featured MCP server, real file system resource, prompt templates
Run: python mcp_servers.py    (shows the server; run server_runnable.py to actually serve)
Requires: pip install mcp anthropic
"""

# ─────────────────────────────────────────────
# PART 1: A production-grade MCP server
# with file system tools, database resources,
# and reusable prompt templates
# ─────────────────────────────────────────────

FULL_SERVER = '''
# server_runnable.py — run this with: python server_runnable.py

import json, os, sqlite3, asyncio, datetime, math
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("productivity-mcp")

# In-memory note store (replace with SQLite or Postgres in production)
NOTES: dict[str, dict] = {}

# ── TOOLS ────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="create_note",
            description="Create or update a note by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id":      {"type": "string", "description": "Unique note ID"},
                    "title":   {"type": "string"},
                    "content": {"type": "string"},
                    "tags":    {"type": "array", "items": {"type": "string"}, "default": []}
                },
                "required": ["id", "title", "content"]
            }
        ),
        types.Tool(
            name="search_notes",
            description="Search notes by keyword",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search term"}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="delete_note",
            description="Delete a note by ID",
            inputSchema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"]
            }
        ),
        types.Tool(
            name="read_file",
            description="Read a text file from the filesystem",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"}
                },
                "required": ["path"]
            }
        ),
        types.Tool(
            name="list_directory",
            description="List files in a directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "extension": {"type": "string", "description": "Filter by extension, e.g. .py", "default": ""}
                },
                "required": ["path"]
            }
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "create_note":
        note = {
            "id": arguments["id"],
            "title": arguments["title"],
            "content": arguments["content"],
            "tags": arguments.get("tags", []),
            "created_at": datetime.datetime.now().isoformat(),
        }
        NOTES[arguments["id"]] = note
        return [types.TextContent(type="text", text=f"Note \'{arguments[\'id\']}\' saved.")]

    elif name == "search_notes":
        query = arguments["query"].lower()
        matches = [
            n for n in NOTES.values()
            if query in n["title"].lower() or query in n["content"].lower()
        ]
        return [types.TextContent(type="text", text=json.dumps(matches, indent=2))]

    elif name == "delete_note":
        if arguments["id"] in NOTES:
            del NOTES[arguments["id"]]
            return [types.TextContent(type="text", text=f"Deleted note \'{arguments[\'id\']}\'.")]
        return [types.TextContent(type="text", text="Note not found.")]

    elif name == "read_file":
        path = Path(arguments["path"])
        if not path.exists():
            return [types.TextContent(type="text", text=f"File not found: {path}")]
        return [types.TextContent(type="text", text=path.read_text(encoding="utf-8"))]

    elif name == "list_directory":
        path = Path(arguments["path"])
        ext = arguments.get("extension", "")
        files = [f.name for f in path.iterdir() if ext == "" or f.suffix == ext]
        return [types.TextContent(type="text", text=json.dumps(sorted(files)))]

    raise ValueError(f"Unknown tool: {name}")


# ── RESOURCES ────────────────────────────────

@app.list_resources()
async def list_resources() -> list[types.Resource]:
    resources = [
        types.Resource(
            uri="notes://all",
            name="All Notes",
            description="All notes in the system",
            mimeType="application/json"
        )
    ]
    # Dynamically expose each note as a resource
    for note_id in NOTES:
        resources.append(types.Resource(
            uri=f"notes://{note_id}",
            name=f"Note: {NOTES[note_id][\'title\']}",
            mimeType="text/plain"
        ))
    return resources

@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "notes://all":
        return json.dumps(list(NOTES.values()), indent=2)
    if uri.startswith("notes://"):
        note_id = uri.replace("notes://", "")
        if note_id in NOTES:
            return json.dumps(NOTES[note_id], indent=2)
    raise ValueError(f"Unknown resource: {uri}")


# ── PROMPTS ───────────────────────────────────

@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="daily_review",
            description="Review all notes and create a daily action plan",
            arguments=[]
        ),
        types.Prompt(
            name="summarize_by_tag",
            description="Summarize notes filtered by tag",
            arguments=[
                types.PromptArgument(name="tag", description="Tag to filter by", required=True)
            ]
        ),
    ]

@app.get_prompt()
async def get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
    if name == "daily_review":
        return types.GetPromptResult(
            description="Daily note review",
            messages=[types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text="Read all my notes (notes://all resource) and create a prioritized action plan for today. Group by urgency."
                )
            )]
        )
    if name == "summarize_by_tag":
        tag = arguments.get("tag", "")
        return types.GetPromptResult(
            description=f"Summarize notes tagged: {tag}",
            messages=[types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Search my notes for tag \'{tag}\' and write a 3-sentence summary of the key themes."
                )
            )]
        )
    raise ValueError(f"Unknown prompt: {name}")


if __name__ == "__main__":
    asyncio.run(stdio_server(app))
'''

print("=== Full Production MCP Server ===")
print("Save as 'server_runnable.py' in this folder, then run:\n")
print("  pip install mcp")
print("  python server_runnable.py\n")

# Write the runnable server file
with open("server_runnable.py", "w") as f:
    # Clean up the escaped quotes for actual file writing
    clean = FULL_SERVER.replace("\\'", "'")
    f.write(clean)
print("server_runnable.py has been written — ready to run!")


# ─────────────────────────────────────────────
# PART 2: MCP vs. Direct Tool Use Summary
# ─────────────────────────────────────────────

print("\n\n=== When to Use MCP vs. Direct Tool Use ===")
comparison = [
    ("Quick prototype",         "Direct tool use", "Simpler, no server needed"),
    ("Shared across apps",      "MCP server",      "One server, many clients"),
    ("Claude Desktop plugin",   "MCP server",      "Only way to add tools to Claude Desktop"),
    ("Production API app",      "Either",          "MCP adds overhead; direct is leaner"),
    ("Team shares tools",       "MCP server",      "Standardized discovery and updates"),
]
print(f"\n{'Use Case':<30} {'Approach':<20} {'Why'}")
print("-" * 70)
for use_case, approach, reason in comparison:
    print(f"{use_case:<30} {approach:<20} {reason}")


# ─────────────────────────────────────────────
# PART 3: Debugging MCP Servers
# ─────────────────────────────────────────────

print("\n\n=== Debugging Tips ===")
tips = [
    "Use MCP Inspector: npx @modelcontextprotocol/inspector python server.py",
    "Check Claude Desktop logs: %APPDATA%\\Claude\\logs\\mcp*.log (Windows)",
    "Test tools in isolation before connecting to Claude",
    "Return descriptive error strings (not exceptions) from call_tool",
    "Add logging: import logging; logging.basicConfig(level=logging.DEBUG)",
]
for i, tip in enumerate(tips, 1):
    print(f"  {i}. {tip}")
