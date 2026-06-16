# Lesson 13: The ReAct Pattern — Reasoning + Acting

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python react_pattern.py
```

## What This Code Covers
- **Text-based ReAct** — model writes explicit Thought/Action/Observation traces
- Parsing Action lines from model output with regex
- Injecting Observations back into the conversation
- Multi-step research agent with web search + calculator + Wikipedia tools

## ReAct vs. Tool Use (Lesson 12)
| Feature | Tool Use (L12) | ReAct (L13) |
|---------|---------------|-------------|
| Actions | JSON blocks | Text format |
| Reasoning | Implicit | Explicit scratchpad |
| Traceability | Low | High |
| Use case | Production APIs | Research/debugging |

## Key Insight
ReAct's explicit reasoning traces make agent behavior auditable — you can read exactly WHY the model took each action. This is critical for debugging complex multi-step tasks.
