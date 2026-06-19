# Lesson 16: Model Context Protocol (MCP) Explained

## Setup
```bash
pip install -r requirements.txt
python server.py   # starts your MCP server
```

## What This Code Covers
- MCP architecture: Host → Client → Server
- **Tools** — functions the LLM calls (get_time, calculate, save_note)
- **Resources** — data the LLM reads (notes://all)
- **Prompts** — reusable prompt templates
- How to connect to Claude Desktop via `claude_desktop_config.json`

## Connect to Claude Desktop
Add this to `%APPDATA%\Claude\claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "my-first-mcp-server": {
      "command": "python",
      "args": ["C:\\path\\to\\lesson-16\\server.py"]
    }
  }
}
```
Restart Claude Desktop → your tools appear automatically.

## MCP vs. Direct Tool Use (Lesson 12)
| | Direct Tool Use | MCP |
|--|--|--|
| Setup | Hardcode in API call | Separate server process |
| Reusability | One app only | Any MCP-compatible host |
| Discovery | Manual | Automatic |
| Use case | Quick integrations | Shareable tool servers |
