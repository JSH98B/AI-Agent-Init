# Lesson 5: How LLMs Are Trained

## Learning Objectives
- Understand pre-training on raw text
- Understand fine-tuning and instruction tuning
- Understand RLHF (Reinforcement Learning from Human Feedback)
- Know why alignment matters

## Key Concepts

### Stage 1: Pre-training
The model reads hundreds of billions of tokens from the internet, books, and code. 
Its only job: **predict the next token**. 

This single objective, repeated trillions of times, forces the model to learn:
- Grammar and syntax
- Facts about the world
- Reasoning patterns
- Code structure

**Cost**: GPT-4 pre-training reportedly cost ~$100M in compute.

### Stage 2: Fine-tuning (SFT)
After pre-training, the model is good at completing text — but not at following instructions.

**Supervised Fine-Tuning (SFT)**: Train on curated (prompt, ideal-response) pairs.
- Input: "Summarize this article: ..."
- Target: A high-quality human-written summary

This teaches the model to be an *assistant*, not just a text predictor.

### Stage 3: RLHF
**Reinforcement Learning from Human Feedback** aligns the model with human preferences:

1. Generate multiple responses to the same prompt
2. Human raters rank them (A > B > C)
3. Train a **reward model** to predict human rankings
4. Use RL (PPO) to nudge the LLM toward responses the reward model scores higher

Result: A model that's helpful, harmless, and honest — not just statistically likely.

### Why Alignment Matters
Without alignment:
- A model optimized purely for "human preference" could learn to be sycophantic
- It might tell users what they want to hear, not what's true
- Edge cases could produce harmful outputs

Anthropic's Constitutional AI (CAI) and OpenAI's RLHF are both attempts to solve this.

## No Code This Lesson
Training LLMs requires hundreds of GPUs and months of time. Instead, your job is to use these already-trained models efficiently via APIs.

## Knowledge Check
1. What is the training objective during pre-training?
2. What problem does SFT solve that pre-training doesn't?
3. Why is RLHF necessary even after SFT?
