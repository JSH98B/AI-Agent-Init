# AI/LLM Certificate Course — Code Repository

A collection of runnable Python scripts from Brian's AI/LLM Certificate Course. Each lesson folder is self-contained and can be opened directly in Cursor.

## Course Progress

| Folder | Lesson | Topic | Code |
|--------|--------|-------|------|
| `lesson-01/` | Lesson 1 | What is AI? | README only (conceptual) |
| `lesson-03/` | Lesson 3 | Neural Networks | `neural_networks.py` |
| `lesson-05/` | Lesson 5 | How LLMs Are Trained | README only (conceptual) |
| `lesson-06/` | Lesson 6 | Transformer & Self-Attention | `self_attention.py` |
| `lesson-07/` | Lesson 7 | Tokens, Embeddings & Context Windows | `tokens_and_embeddings.py` |
| `lesson-08/` | Lesson 8 | Prompt Engineering Fundamentals | `prompt_engineering.py` |
| `lesson-09/` | Lesson 9 | Advanced Prompting | `advanced_prompting.py` |
| `lesson-10/` | Lesson 10 | LLM APIs in Practice | `api_in_practice.py` |
| `lesson-11/` | Lesson 11 | AI Agents & the ReAct Loop | `react_agent.py` |

> **Note:** Lessons 2 and 4 were not delivered due to a scheduling restart. They cover Machine Learning Basics and "What is an LLM?" respectively.

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-certificate-course.git
cd ai-certificate-course
```

### 2. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Add your API key
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 4. Run any lesson
```bash
cd lesson-11
pip install -r requirements.txt
python react_agent.py
```

## Lessons That Require an API Key

| Lesson | API Needed |
|--------|-----------|
| lesson-01, 03, 05, 06, 07 | None (numpy only or no deps) |
| lesson-08, 09, 10, 11 | `ANTHROPIC_API_KEY` |

Get your key at: https://console.anthropic.com

## Opening in Cursor
1. File → Open Folder → select `ai-certificate-course/`
2. Open any lesson folder
3. Cursor will detect the `requirements.txt` and suggest installing deps
4. Hit Run on any `.py` file

## Course Curriculum (Full 23 Lessons)

**Week 1 — Foundations:** What is AI, ML basics, Neural Networks, LLMs, LLM Training  
**Week 2 — LLMs Deep Dive:** Transformers, Tokens/Embeddings, Prompt Engineering, Advanced Prompting, APIs  
**Week 3 — AI Agents:** Agent loops, Tool use, ReAct, Memory, Multi-agent systems  
**Week 4 — MCP & Infrastructure:** MCP explained, MCP servers, MCP clients, RAG, Vector databases  
**Week 5 — Production:** Agent frameworks, Build your first agent, Evaluation & safety  
