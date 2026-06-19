# Lesson 15: Multi-Agent Systems

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python multi_agent_systems.py
```

## What This Code Covers
- **Orchestrator/Worker pattern** — one agent plans, many workers execute in parallel
- **Parallel execution** with `ThreadPoolExecutor` — 3x faster than sequential
- **Pipeline/Handoff pattern** — agents chained in sequence, each processing the previous output

## When to Use Multi-Agent
| Pattern | Use When |
|---------|---------|
| Orchestrator/Worker | Task can be parallelized into independent subtasks |
| Pipeline | Each step depends on the previous output |
| Single agent | Task is simple enough for one model call |

## Key Rule
Don't over-engineer. Start with a single agent. Add more only when you hit a real bottleneck (token limits, parallelism, specialization).
