# Lesson 6: The Transformer Architecture — Self-Attention

## Setup
```bash
pip install -r requirements.txt
python self_attention.py
```

## What This Code Covers
- Scaled dot-product attention (Q, K, V matrices)
- Attention score computation and softmax normalization
- Causal masking (so decoders can't peek at future tokens)
- Multi-head attention with parallel heads

## Key Insight
Self-attention lets every token "look at" every other token in the sequence simultaneously. This is what gives Transformers their power — and why they replaced RNNs, which had to process tokens one at a time.
