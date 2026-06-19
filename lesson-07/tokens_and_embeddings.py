"""
Lesson 7: Tokens, Embeddings & Context Windows
Topics: tiktoken tokenization, cosine similarity, token budget management
Run: python tokens_and_embeddings.py
Requires: pip install tiktoken anthropic
"""

import tiktoken
import json

# ─────────────────────────────────────────────
# PART 1: Tokenization with tiktoken
# ─────────────────────────────────────────────

def tokenize(text: str, model: str = "gpt-4") -> dict:
    """Show how text gets split into tokens."""
    enc = tiktoken.encoding_for_model(model)
    token_ids = enc.encode(text)
    tokens = [enc.decode([t]) for t in token_ids]
    return {
        "text": text,
        "token_ids": token_ids,
        "tokens": tokens,
        "count": len(token_ids),
    }

print("=== Tokenization ===")
examples = [
    "Hello, world!",
    "ChatGPT is an AI assistant.",
    "Unbelievably",       # might split: "Un" + "believ" + "ably"
    "   spaces matter   ",
]

for text in examples:
    result = tokenize(text)
    print(f"\nText:   '{text}'")
    print(f"Tokens: {result['tokens']}")
    print(f"Count:  {result['count']}")


# ─────────────────────────────────────────────
# PART 2: Token costs
# ─────────────────────────────────────────────

PRICE_PER_1M_TOKENS = {
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku":  {"input": 0.80, "output": 4.00},
    "gpt-4o":            {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":       {"input": 0.15, "output": 0.60},
}

def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    prices = PRICE_PER_1M_TOKENS[model]
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000

print("\n\n=== Token Cost Estimator ===")
input_tokens  = 5000
output_tokens = 500

for model, _ in PRICE_PER_1M_TOKENS.items():
    cost = estimate_cost(input_tokens, output_tokens, model)
    print(f"{model:<30} ${cost:.5f} per call")


# ─────────────────────────────────────────────
# PART 3: Cosine Similarity for Embeddings
# ─────────────────────────────────────────────

import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """How similar are two embedding vectors? 1.0 = identical, 0.0 = orthogonal."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n\n=== Cosine Similarity (simulated embeddings) ===")
np.random.seed(42)
dim = 1536  # text-embedding-3-small dimension

# Simulate embeddings: similar sentences → similar vectors
base = np.random.randn(dim)
noise = np.random.randn(dim) * 0.05

cat_1 = base + noise                      # "The cat sat on the mat"
cat_2 = base + np.random.randn(dim)*0.1  # "A cat is resting on the rug"
unrelated = np.random.randn(dim)          # "The stock market crashed today"

sim_cat = cosine_similarity(cat_1, cat_2)
sim_unrelated = cosine_similarity(cat_1, unrelated)

print(f"Cat sentence vs. Cat sentence: {sim_cat:.4f}  (high similarity)")
print(f"Cat sentence vs. Stock news:   {sim_unrelated:.4f}  (low similarity)")
print("\nThis is how RAG retrieval works — find chunks with highest cosine similarity.")


# ─────────────────────────────────────────────
# PART 4: Token Budget Checker
# ─────────────────────────────────────────────

def check_token_budget(
    messages: list[dict],
    system_prompt: str = "",
    model: str = "gpt-4",
    max_context: int = 128_000,
    reserve_for_output: int = 4_000,
) -> dict:
    """
    Before sending to the API, check if your messages fit in the context window.
    Returns status, token counts, and a warning if you're close to the limit.
    """
    enc = tiktoken.encoding_for_model(model)

    system_tokens = len(enc.encode(system_prompt))
    message_tokens = sum(
        len(enc.encode(m.get("content", "")))
        for m in messages
    )
    overhead = len(messages) * 4  # per-message tokens (role, structure)

    total_input = system_tokens + message_tokens + overhead
    available   = max_context - reserve_for_output
    remaining   = available - total_input
    usage_pct   = (total_input / available) * 100

    return {
        "system_tokens":  system_tokens,
        "message_tokens": message_tokens,
        "total_input":    total_input,
        "max_context":    max_context,
        "available":      available,
        "remaining":      remaining,
        "usage_pct":      round(usage_pct, 1),
        "status": "OK" if remaining > 0 else "OVERFLOW",
        "warning": remaining < 5000 and remaining > 0,
    }


print("\n\n=== Token Budget Checker ===")
system = "You are a helpful assistant that answers questions concisely."
msgs = [
    {"role": "user",      "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI where models learn from data."},
    {"role": "user",      "content": "Can you give me a Python example?"},
]

budget = check_token_budget(msgs, system_prompt=system)
print(json.dumps(budget, indent=2))

# Test overflow scenario
huge_content = "word " * 130_000
big_msgs = [{"role": "user", "content": huge_content}]
overflow = check_token_budget(big_msgs)
print(f"\nOverflow test — Status: {overflow['status']}, Remaining: {overflow['remaining']}")
