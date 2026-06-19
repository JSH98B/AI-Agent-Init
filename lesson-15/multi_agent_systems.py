"""
Lesson 15: Multi-Agent Systems
Topics: Orchestrator/worker pattern, parallel execution, agent handoffs
Run: python multi_agent_systems.py
Requires: pip install anthropic
"""

import anthropic
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

client = anthropic.Anthropic()

# ─────────────────────────────────────────────
# PART 1: Simple Orchestrator / Worker Pattern
# ─────────────────────────────────────────────

def worker_agent(task: str, role: str, context: str = "") -> str:
    """A worker agent with a specific role and task."""
    system = f"You are a {role}. Be concise — 2-3 sentences max."
    prompt = f"{context}\n\nTask: {task}" if context else task
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def orchestrator(topic: str) -> dict:
    """
    Orchestrator breaks a research topic into subtasks,
    dispatches workers, then synthesizes the results.
    """
    print(f"\n[ORCHESTRATOR] Topic: {topic}")

    # Step 1: Plan subtasks
    plan_prompt = f"""Break this research topic into exactly 3 specific subtasks for specialist agents.
Topic: {topic}
Return a JSON array of 3 strings. Example: ["subtask1", "subtask2", "subtask3"]
Return ONLY the JSON array."""
    
    plan_resp = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=256,
        temperature=0,
        messages=[{"role": "user", "content": plan_prompt}],
    )
    subtasks = json.loads(plan_resp.content[0].text)
    print(f"[ORCHESTRATOR] Subtasks: {subtasks}")

    # Step 2: Dispatch workers in parallel
    roles = ["research analyst", "technical expert", "business strategist"]
    results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(worker_agent, task, role): task
            for task, role in zip(subtasks, roles)
        }
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            results[task] = result
            print(f"[WORKER] Done: {task[:50]}...")

    # Step 3: Synthesize
    synthesis_prompt = f"""Synthesize these specialist reports into one cohesive paragraph about: {topic}

Reports:
{json.dumps(results, indent=2)}

Write a 3-4 sentence synthesis."""

    synthesis = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=512,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    return {
        "topic": topic,
        "subtasks": subtasks,
        "worker_outputs": results,
        "synthesis": synthesis.content[0].text,
    }


print("=== Orchestrator / Worker Pattern ===")
result = orchestrator("The impact of AI agents on software development workflows")
print(f"\n[SYNTHESIS]\n{result['synthesis']}")


# ─────────────────────────────────────────────
# PART 2: Agent Handoff (pipeline pattern)
# Each agent passes output to the next
# ─────────────────────────────────────────────

@dataclass
class AgentResult:
    agent: str
    output: str

def pipeline(*agents: tuple[str, str, Callable[[str], str]]) -> list[AgentResult]:
    """
    Run agents in sequence — each receives the previous agent's output.
    agents: list of (name, role_description, transform_fn) tuples
    """
    results = []
    current_input = None

    for name, role, fn in agents:
        print(f"\n[{name}] Processing...")
        output = fn(current_input)
        results.append(AgentResult(agent=name, output=output))
        current_input = output
        print(f"[{name}] Output: {output[:100]}...")

    return results


print("\n\n=== Pipeline / Handoff Pattern ===")

raw_data = "Q1 revenue: $2.3M (+18% YoY). CAC dropped to $42. Churn: 4.2%. New enterprise deals: 7."

def extract_metrics(text: str) -> str:
    resp = client.messages.create(
        model="claude-3-5-haiku-20241022", max_tokens=200, temperature=0,
        messages=[{"role": "user", "content": f"Extract key business metrics as bullet points:\n{text}"}],
    )
    return resp.content[0].text

def analyze_trends(metrics: str) -> str:
    resp = client.messages.create(
        model="claude-3-5-haiku-20241022", max_tokens=200, temperature=0,
        messages=[{"role": "user", "content": f"Identify 2 positive trends and 1 concern from these metrics:\n{metrics}"}],
    )
    return resp.content[0].text

def write_summary(analysis: str) -> str:
    resp = client.messages.create(
        model="claude-3-5-haiku-20241022", max_tokens=200, temperature=0.3,
        messages=[{"role": "user", "content": f"Write a 2-sentence executive summary based on:\n{analysis}"}],
    )
    return resp.content[0].text

# Each agent is (name, fn) — output feeds to next
steps = [
    ("Extractor", "metrics analyst", extract_metrics),
    ("Analyzer",  "trend spotter",  analyze_trends),
    ("Writer",    "exec comms",     write_summary),
]

pipeline_results = pipeline(*steps)
print(f"\n[FINAL EXECUTIVE SUMMARY]\n{pipeline_results[-1].output}")
