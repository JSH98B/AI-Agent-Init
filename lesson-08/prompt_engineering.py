"""
Lesson 8: Prompt Engineering Fundamentals
Topics: System prompts, temperature, top-p, extraction with temperature=0
Run: python prompt_engineering.py
Requires: pip install anthropic
"""

import anthropic
import json

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# ─────────────────────────────────────────────
# PART 1: Basic API call with system prompt
# ─────────────────────────────────────────────

def basic_call(user_message: str, system: str = "", temperature: float = 1.0) -> str:
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=512,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


print("=== Basic API Call ===")
answer = basic_call(
    "What is the capital of France?",
    system="You are a geography tutor. Answer in one sentence.",
)
print(f"Answer: {answer}")


# ─────────────────────────────────────────────
# PART 2: Temperature demo — creative vs. deterministic
# ─────────────────────────────────────────────

print("\n=== Temperature Effect ===")
prompt = "Write a one-sentence tagline for a coffee shop."

for temp in [0.0, 0.5, 1.0]:
    result = basic_call(prompt, temperature=temp)
    print(f"temperature={temp}: {result}")


# ─────────────────────────────────────────────
# PART 3: Structured extraction (temperature=0)
# ─────────────────────────────────────────────

EXTRACT_SYSTEM = """
Extract information from the user's text and return ONLY valid JSON.
No explanation, no markdown — just the JSON object.

Schema:
{
  "name": string,
  "email": string | null,
  "phone": string | null,
  "company": string | null
}
"""

def extract_contact(text: str) -> dict:
    raw = basic_call(text, system=EXTRACT_SYSTEM, temperature=0)
    return json.loads(raw)

print("\n=== Structured Extraction (temperature=0) ===")
samples = [
    "Hi, I'm Sarah Johnson from Acme Corp. Reach me at sarah@acme.com or 555-1234.",
    "My name is Bob. No email yet but you can call 800-555-9876.",
    "Contact Alex at alex.smith@startup.io — no phone on file.",
]

for text in samples:
    contact = extract_contact(text)
    print(f"\nInput:  {text}")
    print(f"Output: {json.dumps(contact, indent=2)}")


# ─────────────────────────────────────────────
# PART 4: Prompt templates (f-string pattern)
# ─────────────────────────────────────────────

def summarize(text: str, max_words: int = 50, tone: str = "neutral") -> str:
    prompt = f"""Summarize the following text in {max_words} words or fewer.
Tone: {tone}

Text:
{text}

Summary:"""
    return basic_call(prompt, temperature=0.3)


print("\n=== Prompt Templates ===")
article = """
The James Webb Space Telescope has captured the deepest infrared image of the universe ever taken.
The image shows thousands of galaxies, some of which formed less than a billion years after the Big Bang.
Scientists say this will help us understand how galaxies form and evolve over cosmic time.
"""

for tone in ["neutral", "excited", "technical"]:
    summary = summarize(article, max_words=30, tone=tone)
    print(f"\n[{tone}] {summary}")
