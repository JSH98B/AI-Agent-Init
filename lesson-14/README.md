# Lesson 14: Memory Systems for AI Agents

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python memory_systems.py
```

## What This Code Covers
Three memory strategies every agent builder needs to know:

### 1. In-Context Memory
- Everything stays in the message list
- Perfect recall, zero complexity
- **Use when**: short conversations, < 10K tokens total

### 2. Summarization Memory
- Compresses old messages into a rolling summary when context gets long
- Lossy but scalable
- **Use when**: long chat sessions where exact recall isn't critical

### 3. Semantic (Vector) Memory
- Stores facts as embeddings, retrieves the most relevant at query time
- Scales to unlimited memories, finds semantically related info
- **Use when**: personal assistants, knowledge bases, long-term user memory
- **In production**: replace `fake_embed()` with OpenAI `text-embedding-3-small` or Voyage AI

## Production Upgrade Path
```python
# Replace fake_embed() with real embeddings:
import openai
def embed(text: str) -> list[float]:
    return openai.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding
```
Then store in Pinecone, Chroma, or pgvector instead of a Python list.
