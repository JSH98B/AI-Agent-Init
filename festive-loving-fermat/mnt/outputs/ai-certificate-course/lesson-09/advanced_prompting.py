"""
Lesson 9: Advanced Prompting
Topics: Few-shot examples, chain-of-thought, structured JSON output with retry
Run: python advanced_prompting.py
Requires: pip install anthropic pydantic
"""

import anthropic
import json
import re
from pydantic import BaseModel, ValidationError
from typing import Optional

client = anthropic.Anthropic()

# ─────────────────────────────────────────────
# PART 1: Few-Shot Classification
# ─────────────────────────────────────────────

FEW_SHOT_SYSTEM = """Classify customer support tickets into one of: 
billing, technical, account, general

Examples:
User: My card was charged twice this month.
Label: billing

User: The app crashes when I upload a file.
Label: technical

User: I need to reset my password.
Label: account

User: What are your business hours?
Label: general

Reply with ONLY the label word. Nothing else."""

def classify_ticket(ticket: str) -> str:
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=10,
        temperature=0,
        system=FEW_SHOT_SYSTEM,
        messages=[{"role": "user", "content": ticket}],
    )
    return response.content[0].text.strip().lower()

print("=== Few-Shot Classification ===")
tickets = [
    "I was billed $99 instead of $9.",
    "The dashboard won't load in Chrome.",
    "Can I change my username?",
    "Do you offer a student discount?",
]
for t in tickets:
    label = classify_ticket(t)
    print(f"  [{label}] {t}")


# ─────────────────────────────────────────────
# PART 2: Chain-of-Thought (CoT)
# ─────────────────────────────────────────────

def solve_with_cot(problem: str) -> str:
    """Ask the model to reason step by step before answering."""
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=512,
        temperature=0,
        system="Think step by step. Show your reasoning, then give a final answer on the last line starting with 'Answer:'",
        messages=[{"role": "user", "content": problem}],
    )
    return response.content[0].text

print("\n=== Chain-of-Thought Reasoning ===")
problem = "A store sells apples for $0.75 each and bags of 6 for $3.50. If I need 20 apples, what's the cheapest way to buy them and how much does it cost?"
result = solve_with_cot(problem)
print(result)


# ─────────────────────────────────────────────
# PART 3: Structured JSON Output with Pydantic + Retry
# ─────────────────────────────────────────────

class ProductReview(BaseModel):
    sentiment: str          # "positive", "negative", "neutral"
    score: int              # 1-5
    key_themes: list[str]   # what the review mentions
    summary: str            # one-sentence summary
    would_recommend: bool

REVIEW_SYSTEM = """Analyze the product review and return a JSON object with this exact schema:
{
  "sentiment": "positive" | "negative" | "neutral",
  "score": 1-5 integer,
  "key_themes": ["theme1", "theme2"],
  "summary": "one sentence",
  "would_recommend": true | false
}
Return ONLY the JSON. No explanation."""

def analyze_review(review: str, max_retries: int = 3) -> Optional[ProductReview]:
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=256,
                temperature=0,
                system=REVIEW_SYSTEM,
                messages=[{"role": "user", "content": review}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown code blocks if present
            raw = re.sub(r"```(?:json)?\n?", "", raw).strip()
            data = json.loads(raw)
            return ProductReview(**data)  # Pydantic validates types
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"  Attempt {attempt+1} failed: {e}")
    return None

print("\n=== Structured Output with Pydantic ===")
reviews = [
    "Absolutely love this blender! Makes perfect smoothies every morning. Super quiet too. 10/10 would buy again.",
    "Terrible. Broke after 2 weeks. Customer service was useless. Never buying from this brand again.",
    "It's okay. Does what it says, nothing special. Delivery was fast.",
]

for review in reviews:
    result = analyze_review(review)
    if result:
        print(f"\nReview: {review[:60]}...")
        print(f"  Sentiment: {result.sentiment} | Score: {result.score}/5")
        print(f"  Themes: {result.key_themes}")
        print(f"  Recommend: {result.would_recommend}")
