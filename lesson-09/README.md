# Lesson 9: Advanced Prompting

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python advanced_prompting.py
```

## What This Code Covers
- **Few-shot classification** — give examples in the prompt, model learns the pattern
- **Chain-of-thought (CoT)** — "think step by step" dramatically improves multi-step reasoning
- **Structured JSON output** — force the model to return parseable data
- **Pydantic validation** — catch type errors and bad output before it crashes your app
- **Retry loop** — handle occasional malformed JSON gracefully

## Key Rules
- Few-shot: 3-5 examples is usually enough; more = more tokens = more cost
- CoT: Always use for math, logic, multi-step tasks
- Structured output: Always pair with Pydantic validation + retry
