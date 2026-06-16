# Lesson 10: LLM APIs in Practice

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python api_in_practice.py
```

## What This Code Covers
- **Exponential backoff with jitter** — handle rate limits without crashing
- **`APIResponse` dataclass** — track tokens and attempts per call
- **`Conversation` class** — stateful multi-turn chat with usage tracking
- **Streaming** — print tokens as they arrive (better UX for long responses)

## Production Checklist
- [ ] Always wrap API calls in retry logic
- [ ] Track token usage per call (budget monitoring)
- [ ] Use `temperature=0` for deterministic tasks
- [ ] Stream responses in any UI that shows text to users
