# Lesson 7: Tokens, Embeddings & Context Windows

## Setup
```bash
pip install -r requirements.txt
python tokens_and_embeddings.py
```

## What This Code Covers
- Tokenizing text with `tiktoken` (same library OpenAI uses)
- Understanding that tokens ≠ words (subword tokenization)
- Estimating API costs before sending requests
- Cosine similarity — the math behind embedding search
- Token budget checker — safely fit messages into context windows

## Key Insight
Context windows are the hard limit on how much an LLM can "see" at once. Always count tokens before sending large prompts. The budget checker pattern is essential for production apps — overflow crashes requests silently.
