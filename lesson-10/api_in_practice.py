"""
Lesson 10: LLM APIs in Practice
Topics: Robust API wrapper, exponential backoff, multi-turn conversations
Run: python api_in_practice.py
Requires: pip install anthropic
"""

import anthropic
import time
import random
import json
from dataclasses import dataclass, field
from typing import Optional

client = anthropic.Anthropic()

# ─────────────────────────────────────────────
# PART 1: Bulletproof API Wrapper with Exponential Backoff
# ─────────────────────────────────────────────

@dataclass
class APIResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    attempts: int


def call_llm(
    messages: list[dict],
    system: str = "",
    model: str = "claude-3-5-haiku-20241022",
    max_tokens: int = 1024,
    temperature: float = 1.0,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> APIResponse:
    """
    Production-grade API wrapper with:
    - Exponential backoff for rate limits and server errors
    - Jitter to avoid thundering herd when many calls retry simultaneously
    - Detailed usage tracking
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
            return APIResponse(
                text=response.content[0].text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=response.model,
                attempts=attempt,
            )

        except anthropic.RateLimitError as e:
            last_error = e
            wait = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"  Rate limited. Waiting {wait:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)

        except anthropic.APIStatusError as e:
            if e.status_code >= 500:   # Server errors — retry
                last_error = e
                wait = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(f"  Server error {e.status_code}. Waiting {wait:.1f}s")
                time.sleep(wait)
            else:
                raise  # Client errors (400, 401, 422) — don't retry

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


print("=== Bulletproof API Call ===")
result = call_llm(
    messages=[{"role": "user", "content": "What's 2+2? Answer in one word."}],
    temperature=0,
)
print(f"Answer: {result.text}")
print(f"Tokens: {result.input_tokens} in / {result.output_tokens} out | Attempts: {result.attempts}")


# ─────────────────────────────────────────────
# PART 2: Multi-Turn Conversation Manager
# ─────────────────────────────────────────────

class Conversation:
    """
    Manages a stateful multi-turn conversation.
    Tracks message history and token usage across turns.
    """
    def __init__(self, system: str = "", model: str = "claude-3-5-haiku-20241022"):
        self.system = system
        self.model = model
        self.messages: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, maintaining history."""
        self.messages.append({"role": "user", "content": user_message})

        result = call_llm(
            messages=self.messages,
            system=self.system,
            model=self.model,
            temperature=0.7,
        )

        # Add assistant reply to history for next turn
        self.messages.append({"role": "assistant", "content": result.text})

        self.total_input_tokens  += result.input_tokens
        self.total_output_tokens += result.output_tokens

        return result.text

    def usage_summary(self) -> dict:
        return {
            "turns": len(self.messages) // 2,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": (
                self.total_input_tokens * 0.80 +
                self.total_output_tokens * 4.00
            ) / 1_000_000,
        }


print("\n=== Multi-Turn Conversation ===")
convo = Conversation(
    system="You are a concise Python tutor. Keep answers under 3 sentences."
)

turns = [
    "What is a list comprehension?",
    "Can you show me a quick example?",
    "How is that different from a regular for loop?",
]

for msg in turns:
    print(f"\nUser: {msg}")
    reply = convo.chat(msg)
    print(f"Claude: {reply}")

print(f"\n--- Usage Summary ---")
print(json.dumps(convo.usage_summary(), indent=2))


# ─────────────────────────────────────────────
# PART 3: Streaming responses
# ─────────────────────────────────────────────

print("\n\n=== Streaming (token-by-token output) ===")
print("Claude: ", end="", flush=True)

with client.messages.stream(
    model="claude-3-5-haiku-20241022",
    max_tokens=100,
    messages=[{"role": "user", "content": "Count from 1 to 10, one number per line."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

print()  # newline after stream ends
