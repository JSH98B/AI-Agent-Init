# AI & LLM Engineering

Hands-on Python implementations covering the full AI/LLM stack — from neural network fundamentals to production agent systems.

## Modules

| Folder | Topic |
|--------|-------|
| `lesson-01/` | What is AI? |
| `lesson-03/` | Neural Networks |
| `lesson-05/` | How LLMs Are Trained |
| `lesson-06/` | Transformer & Self-Attention |
| `lesson-07/` | Tokens, Embeddings & Context Windows |
| `lesson-08/` | Prompt Engineering |
| `lesson-09/` | Advanced Prompting |
| `lesson-10/` | LLM APIs in Practice |
| `lesson-11/` | AI Agents & the ReAct Loop |
| `lesson-12/` | Tool Use & Function Calling |
| `lesson-13/` | The ReAct Pattern |
| `lesson-14/` | Memory Systems |
| `lesson-15/` | Multi-Agent Systems |
| `lesson-16/` | Model Context Protocol (MCP) |
| `lesson-17/` | Building MCP Servers |

## Getting Started

```bash
git clone https://github.com/JSH98B/AI-Agent-Init.git
cd AI-Agent-Init
python -m venv venv
venv\Scripts\activate
cd lesson-11
pip install -r requirements.txt
python react_agent.py
```

## Topics Covered

**Foundations** — Neural networks, activation functions, backpropagation, LLM training (pre-training, fine-tuning, RLHF)

**LLM Internals** — Transformer architecture, self-attention (Q/K/V), tokenization, embeddings, context windows

**Prompt Engineering** — System prompts, temperature, few-shot examples, chain-of-thought, structured JSON output with Pydantic

**APIs & Production** — Anthropic SDK, exponential backoff, streaming, multi-turn conversation management

**Agents** — ReAct loop, tool use, function calling, parallel tool execution, memory systems (in-context, summarization, vector)

**Multi-Agent** — Orchestrator/worker patterns, parallel execution, pipeline handoffs

**MCP** — Model Context Protocol servers, tools vs. resources vs. prompts, Claude Desktop integration
