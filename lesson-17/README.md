# Lesson 17: Building MCP Servers — Deep Dive

## Setup
```bash
pip install -r requirements.txt
python mcp_servers.py     # prints the server code and writes server_runnable.py
python server_runnable.py # actually starts the MCP server
```

## What This Code Covers
- Full-featured MCP server with 5 tools: create/search/delete notes, read files, list directories
- Dynamic resources — each note exposed as its own `notes://<id>` URI
- Reusable prompt templates: `daily_review`, `summarize_by_tag`
- Debugging tips and MCP Inspector usage

## Connect to Claude Desktop
```json
{
  "mcpServers": {
    "productivity-mcp": {
      "command": "python",
      "args": ["C:\\path\\to\\lesson-17\\server_runnable.py"]
    }
  }
}
```

## Testing with MCP Inspector
```bash
npx @modelcontextprotocol/inspector python server_runnable.py
```
Opens a web UI to test all your tools and resources before connecting to Claude.
