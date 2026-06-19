# Lesson 8: Prompt Engineering Fundamentals

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python prompt_engineering.py
```

## What This Code Covers
- Basic Anthropic API call with system prompts
- Temperature: 0.0 = deterministic, 1.0 = creative
- Structured extraction returning valid JSON (always use temperature=0)
- Reusable prompt templates with f-strings

## Key Rules
1. **temperature=0** for extraction, classification, and anything that needs consistent output
2. **temperature=0.7–1.0** for creative writing, brainstorming
3. **System prompt** = the model's persona and constraints
4. **User prompt** = the actual task/question
