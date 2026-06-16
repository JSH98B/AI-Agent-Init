"""
Lesson 13: The ReAct Pattern — Reasoning + Acting
Topics: Interleaved Thought/Action/Observation, search+answer agent, scratchpad tracing
Run: python react_pattern.py
Requires: pip install anthropic
"""

import anthropic
import json
import re

client = anthropic.Anthropic()

# ─────────────────────────────────────────────
# PART 1: ReAct with Explicit Thought Traces
# Unlike lesson-11's implicit loop, here we make
# the model write Thought/Action/Observation explicitly
# ─────────────────────────────────────────────

REACT_SYSTEM = """You are a research assistant that uses tools to answer questions.
Think step by step using this exact format:

Thought: [your reasoning about what to do next]
Action: tool_name({"param": "value"})
Observation: [you will see the tool result here]

... repeat Thought/Action/Observation as needed ...

Thought: I now have enough information to answer.
Final Answer: [your complete answer to the user]

Available tools:
- web_search({"query": "search terms"}) — search the web for information
- calculator({"expression": "2 ** 10"}) — evaluate a math expression
- wikipedia({"topic": "topic name"}) — get a Wikipedia summary

Always start with a Thought. Never skip steps."""


# ─────────────────────────────────────────────
# PART 2: Simulated Tool Implementations
# ─────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "Claude AI": "Claude is an AI assistant made by Anthropic. It uses a Constitutional AI approach for safety alignment. Claude 3.5 Sonnet was released in 2024.",
    "Anthropic": "Anthropic is an AI safety company founded in 2021 by Dario Amodei, Daniela Amodei, and others. It focuses on building safe and steerable AI systems.",
    "transformer": "The Transformer is a neural network architecture introduced in 'Attention Is All You Need' (2017). It uses self-attention instead of recurrence and is the foundation of all modern LLMs.",
    "RAG": "Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model. The retriever finds relevant documents; the LLM synthesizes them into an answer.",
    "GPT-4": "GPT-4 is OpenAI's large multimodal model released in 2023. It accepts image and text inputs and produces text outputs.",
}

SEARCH_RESULTS = {
    "anthropic claude": "Anthropic Claude is a family of AI models. Claude 3.5 Sonnet scored 90.4% on MMLU. Pricing starts at $3/M input tokens.",
    "ai agent frameworks 2024": "Top AI agent frameworks: LangChain (most popular), LlamaIndex (best for RAG), CrewAI (multi-agent), AutoGen (Microsoft). Each has different strengths.",
    "react pattern llm": "ReAct (Reasoning + Acting) was introduced in a 2022 paper. It interleaves reasoning traces with actions, improving accuracy on multi-step tasks by 10-40%.",
}

def web_search(query: str) -> str:
    """Simulated web search."""
    query_lower = query.lower()
    for key, val in SEARCH_RESULTS.items():
        if any(word in query_lower for word in key.split()):
            return val
    return f"Search results for '{query}': No specific results found. Try a more specific query."

def wikipedia(topic: str) -> str:
    """Simulated Wikipedia lookup."""
    for key, val in KNOWLEDGE_BASE.items():
        if key.lower() in topic.lower() or topic.lower() in key.lower():
            return val
    return f"Wikipedia: No article found for '{topic}'. Try a different search term."

def calculator(expression: str) -> str:
    """Safe math evaluator."""
    import math
    try:
        result = eval(expression, {"math": math, "__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def execute_tool_from_text(action_text: str) -> str:
    """Parse 'tool_name({"key": "val"})' from model output and execute."""
    match = re.match(r'(\w+)\((\{.*\})\)', action_text.strip())
    if not match:
        return "Error: Could not parse action. Use format: tool_name({\"param\": \"value\"})"
    tool_name = match.group(1)
    try:
        params = json.loads(match.group(2))
    except json.JSONDecodeError:
        return "Error: Invalid JSON in action parameters."

    if tool_name == "web_search":
        return web_search(**params)
    elif tool_name == "wikipedia":
        return wikipedia(**params)
    elif tool_name == "calculator":
        return calculator(**params)
    return f"Error: Unknown tool '{tool_name}'"


# ─────────────────────────────────────────────
# PART 3: ReAct Execution Loop
# ─────────────────────────────────────────────

def react_agent(question: str, max_steps: int = 8) -> str:
    """
    Run the ReAct loop using text-based Thought/Action/Observation format.
    The model generates text; we parse Action lines and inject Observations.
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)

    # Seed the conversation with the question
    scratchpad = f"Question: {question}\n"
    messages = [{"role": "user", "content": scratchpad}]

    for step in range(max_steps):
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=512,
            system=REACT_SYSTEM,
            messages=messages,
        )
        output = response.content[0].text.strip()
        print(f"\n{output}")

        # Check for final answer
        if "Final Answer:" in output:
            final = output.split("Final Answer:")[-1].strip()
            return final

        # Find Action line and execute it
        action_match = re.search(r'Action:\s*(.+)', output)
        if action_match:
            action = action_match.group(1).strip()
            observation = execute_tool_from_text(action)
            print(f"Observation: {observation}")

            # Append model output + observation back to conversation
            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            # No action found — model is done or confused
            messages.append({"role": "assistant", "content": output})

    return "Max steps reached without a final answer."


# ─────────────────────────────────────────────
# PART 4: Run Example Queries
# ─────────────────────────────────────────────

questions = [
    "What is the ReAct pattern and when was it introduced?",
    "What is 2 to the power of 20, and what company makes Claude AI?",
]

for q in questions:
    answer = react_agent(q)
    print(f"\n✓ FINAL ANSWER: {answer}\n")
