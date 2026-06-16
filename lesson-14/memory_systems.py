"""
Lesson 14: Memory Systems for AI Agents
Topics: In-context memory, summarization memory, semantic (vector) memory
Run: python memory_systems.py
Requires: pip install anthropic numpy
"""

import anthropic
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

client = anthropic.Anthropic()

# ─────────────────────────────────────────────
# PART 1: In-Context Memory (simplest)
# Just keep everything in the message list
# ─────────────────────────────────────────────

class InContextMemory:
    """
    The simplest memory: every message stays in the context window.
    Pros: perfect recall, zero complexity
    Cons: hits token limits fast, costs more with each turn
    """
    def __init__(self, system: str = "You are a helpful assistant."):
        self.system = system
        self.messages: list[dict] = []

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=256,
            system=self.system,
            messages=self.messages,
        )
        reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    @property
    def token_estimate(self) -> int:
        """Rough token count (4 chars ≈ 1 token)."""
        total_chars = sum(len(m["content"]) for m in self.messages)
        return total_chars // 4


print("=== In-Context Memory ===")
bot = InContextMemory(system="You are a concise assistant. Keep answers under 2 sentences.")
turns = [
    "My name is Brian and I'm learning about AI agents.",
    "What topic am I studying?",
    "What's my name?",
]
for msg in turns:
    reply = bot.chat(msg)
    print(f"User: {msg}")
    print(f"Bot:  {reply}")
    print(f"[~{bot.token_estimate} tokens in context]\n")


# ─────────────────────────────────────────────
# PART 2: Summarization Memory
# Compress old history to save tokens
# ─────────────────────────────────────────────

class SummarizationMemory:
    """
    When conversation gets long, summarize old messages into a compact summary.
    Keeps a rolling summary + recent messages.
    Pros: scales to long sessions, low cost
    Cons: lossy — details in old messages may be forgotten
    """
    def __init__(self, max_messages: int = 6, system: str = "You are a helpful assistant."):
        self.system = system
        self.max_messages = max_messages
        self.messages: list[dict] = []
        self.summary: str = ""

    def _summarize(self) -> str:
        """Ask the model to compress the oldest half of messages."""
        to_compress = self.messages[:len(self.messages)//2]
        text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in to_compress)
        prompt = f"Summarize this conversation history concisely (2-3 sentences):\n\n{text}"
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def chat(self, user_input: str) -> str:
        # Compress if too long
        if len(self.messages) >= self.max_messages:
            new_summary = self._summarize()
            self.summary = new_summary
            self.messages = self.messages[len(self.messages)//2:]  # keep recent half
            print(f"  [Memory compressed → summary: '{self.summary[:80]}...']")

        # Build system with summary prepended
        system = self.system
        if self.summary:
            system += f"\n\nConversation so far: {self.summary}"

        self.messages.append({"role": "user", "content": user_input})
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=256,
            system=system,
            messages=self.messages,
        )
        reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply})
        return reply


print("\n=== Summarization Memory ===")
bot2 = SummarizationMemory(max_messages=4)
long_convo = [
    "My name is Brian. I work in logistics.",
    "I'm trying to learn Python for AI projects.",
    "My favorite framework so far is FastAPI.",
    "I've been coding for about 6 months.",
    "What do you know about me so far?",  # triggers compression before this
]
for msg in long_convo:
    reply = bot2.chat(msg)
    print(f"User: {msg}")
    print(f"Bot:  {reply}\n")


# ─────────────────────────────────────────────
# PART 3: Semantic (Vector) Memory
# Store memories as embeddings, retrieve by similarity
# ─────────────────────────────────────────────

@dataclass
class Memory:
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


def fake_embed(text: str) -> np.ndarray:
    """
    Simulated embedding (in production: use OpenAI/Anthropic/Voyage embeddings API).
    Creates a deterministic vector based on word overlap — good enough to demo similarity.
    """
    words = set(text.lower().split())
    vocab = ["brian", "python", "ai", "agent", "memory", "learn", "code",
             "project", "tool", "model", "llm", "api", "data", "work", "build"]
    vec = np.array([1.0 if w in words else 0.0 for w in vocab])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class SemanticMemory:
    """
    Stores memories as embeddings. At query time, finds the most relevant memories.
    Pros: scales to thousands of memories, finds semantically related info
    Cons: requires embedding model, approximate retrieval
    """
    def __init__(self):
        self.memories: list[Memory] = []

    def remember(self, text: str):
        """Store a new memory with its embedding."""
        mem = Memory(content=text, embedding=fake_embed(text))
        self.memories.append(mem)

    def recall(self, query: str, top_k: int = 3) -> list[Memory]:
        """Retrieve the top-k most relevant memories for a query."""
        query_vec = fake_embed(query)
        scored = [(cosine_sim(query_vec, m.embedding), m) for m in self.memories]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def chat(self, user_input: str) -> str:
        """Answer using retrieved memories as context."""
        relevant = self.recall(user_input, top_k=3)
        memory_context = "\n".join(f"- {m.content}" for m in relevant)

        system = f"""You are a personal assistant with memory.
Use these relevant memories to personalize your response:
{memory_context}
"""
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": user_input}],
        )
        return response.content[0].text


print("\n=== Semantic (Vector) Memory ===")
mem = SemanticMemory()

# Store facts about Brian
facts = [
    "Brian's name is Brian Jung.",
    "Brian works in logistics and supply chain.",
    "Brian is learning Python and AI agent development.",
    "Brian's GitHub username is JSH98B.",
    "Brian uses Claude for coding projects.",
    "Brian wants to build AI-powered business tools.",
]
for fact in facts:
    mem.remember(fact)
    print(f"  Stored: {fact}")

print()
queries = [
    "What coding tools does Brian use?",
    "What is Brian's professional background?",
    "What are Brian's AI learning goals?",
]
for q in queries:
    relevant = mem.recall(q, top_k=2)
    reply = mem.chat(q)
    print(f"Query: {q}")
    print(f"Retrieved memories: {[r.content[:40]+'...' for r in relevant]}")
    print(f"Answer: {reply}\n")
