"""
Lesson 12: Tool Use and Function Calling
Topics: JSON schema tool definitions, parallel tool calls, structured tool results
Run: python tool_use.py
Requires: pip install anthropic
"""

import anthropic
import json
import random
from datetime import datetime, timedelta

client = anthropic.Anthropic()

# ─────────────────────────────────────────────
# PART 1: Define Tools via JSON Schema
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city. Returns temperature and conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'New York' or 'Tokyo'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit. Defaults to fahrenheit."
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "search_flights",
        "description": "Search for available flights between two cities on a given date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Departure city"},
                "destination": {"type": "string", "description": "Arrival city"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "passengers": {"type": "integer", "description": "Number of passengers", "default": 1}
            },
            "required": ["origin", "destination", "date"]
        }
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency code, e.g. USD"},
                "to_currency": {"type": "string", "description": "Target currency code, e.g. EUR"}
            },
            "required": ["amount", "from_currency", "to_currency"]
        }
    }
]

# ─────────────────────────────────────────────
# PART 2: Mock Tool Implementations
# ─────────────────────────────────────────────

def get_weather(city: str, unit: str = "fahrenheit") -> dict:
    """Simulated weather API."""
    random.seed(hash(city) % 100)
    temp_f = random.randint(45, 95)
    temp = temp_f if unit == "fahrenheit" else round((temp_f - 32) * 5/9, 1)
    conditions = random.choice(["sunny", "partly cloudy", "cloudy", "rainy"])
    return {"city": city, "temperature": temp, "unit": unit, "conditions": conditions}

def search_flights(origin: str, destination: str, date: str, passengers: int = 1) -> dict:
    """Simulated flight search."""
    random.seed(hash(f"{origin}{destination}") % 100)
    flights = []
    for i in range(3):
        price = random.randint(150, 800)
        hour = random.randint(6, 20)
        flights.append({
            "flight": f"AA{random.randint(100,999)}",
            "departure": f"{hour:02d}:00",
            "arrival": f"{(hour+5)%24:02d}:00",
            "price_per_person": price,
            "total_price": price * passengers
        })
    return {"origin": origin, "destination": destination, "date": date, "flights": sorted(flights, key=lambda x: x["price_per_person"])}

def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Simulated currency conversion."""
    rates = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 149.5, "KRW": 1325.0}
    usd = amount / rates.get(from_currency, 1.0)
    converted = usd * rates.get(to_currency, 1.0)
    return {"amount": amount, "from": from_currency, "to": to_currency, "converted": round(converted, 2)}

def execute_tool(name: str, inputs: dict) -> str:
    """Route tool calls to their implementations."""
    if name == "get_weather":
        return json.dumps(get_weather(**inputs))
    elif name == "search_flights":
        return json.dumps(search_flights(**inputs))
    elif name == "convert_currency":
        return json.dumps(convert_currency(**inputs))
    return json.dumps({"error": f"Unknown tool: {name}"})

# ─────────────────────────────────────────────
# PART 3: Single Tool Call
# ─────────────────────────────────────────────

def ask_with_tools(question: str, verbose: bool = True) -> str:
    """Run one full tool-use exchange."""
    messages = [{"role": "user", "content": question}]

    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        tools=TOOLS,
        messages=messages,
    )

    # Process tool calls if any
    if response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                if verbose:
                    print(f"  → Calling {block.name}({json.dumps(block.input)})")
                result = execute_tool(block.name, block.input)
                if verbose:
                    print(f"  ← Result: {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "user", "content": tool_results})
        final = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        return final.content[0].text

    return response.content[0].text


print("=== Single Tool Call ===")
answer = ask_with_tools("What's the weather like in Seoul right now?")
print(f"\nAnswer: {answer}")

# ─────────────────────────────────────────────
# PART 4: Parallel Tool Calls
# ─────────────────────────────────────────────

print("\n\n=== Parallel Tool Calls ===")
answer = ask_with_tools(
    "I'm planning a trip from New York to London on 2026-07-04 for 2 people. "
    "What's the weather in London, what flights are available, "
    "and how much is $500 in British pounds?"
)
print(f"\nAnswer: {answer}")
