"""
Lesson 6: The Transformer Architecture — Self-Attention
Topics: Q/K/V vectors, attention scores, softmax, weighted value sum
Run: python self_attention.py
"""

import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ─────────────────────────────────────────────
# PART 1: Scaled Dot-Product Attention
# ─────────────────────────────────────────────

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Core attention mechanism from 'Attention Is All You Need' (Vaswani et al., 2017).

    Args:
        Q: Query matrix  (seq_len, d_k)
        K: Key matrix    (seq_len, d_k)
        V: Value matrix  (seq_len, d_v)
        mask: Optional causal mask to prevent attending to future tokens

    Returns:
        output: Attended values  (seq_len, d_v)
        weights: Attention weights (seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # Step 1: Compute raw attention scores — how relevant is each token to each other?
    scores = Q @ K.T / np.sqrt(d_k)   # (seq_len, seq_len)

    # Step 2: Apply causal mask (decoder only — can't look at future tokens)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Step 3: Softmax → probabilities that sum to 1 per row
    weights = softmax(scores, axis=-1)  # (seq_len, seq_len)

    # Step 4: Weighted sum of Values
    output = weights @ V               # (seq_len, d_v)

    return output, weights


# ─────────────────────────────────────────────
# PART 2: Demo with toy tokens
# ─────────────────────────────────────────────

np.random.seed(42)

# Imagine we have 4 tokens, each embedded as a 8-dim vector
seq_len = 4
d_model = 8
d_k = 4   # query/key dimension
d_v = 4   # value dimension

# Token embeddings (randomly initialized for demo)
X = np.random.randn(seq_len, d_model)

# Learned projection matrices (in a real Transformer, these are trained)
W_Q = np.random.randn(d_model, d_k) * 0.1
W_K = np.random.randn(d_model, d_k) * 0.1
W_V = np.random.randn(d_model, d_v) * 0.1

# Project input into Q, K, V spaces
Q = X @ W_Q   # (4, 4)
K = X @ W_K   # (4, 4)
V = X @ W_V   # (4, 4)

print("=== Scaled Dot-Product Attention ===")
print(f"Sequence length: {seq_len}, d_k: {d_k}, d_v: {d_v}")

output, weights = scaled_dot_product_attention(Q, K, V)

print("\nAttention weights (each row sums to 1):")
print(np.round(weights, 3))
print("\nOutput (attended value vectors):")
print(np.round(output, 3))

# ─────────────────────────────────────────────
# PART 3: Causal (masked) attention — decoder style
# ─────────────────────────────────────────────

# Create causal mask: token i can only attend to tokens 0..i
causal_mask = np.tril(np.ones((seq_len, seq_len)))
print("\n=== Causal Mask ===")
print(causal_mask)

_, causal_weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print("\nCausal attention weights (future tokens masked):")
print(np.round(causal_weights, 3))
print("Note: upper triangle is ~0 — future tokens are invisible.")

# ─────────────────────────────────────────────
# PART 4: Multi-Head Attention
# ─────────────────────────────────────────────

def multi_head_attention(X, num_heads=2):
    """
    Multi-head attention: run attention H times in parallel with different projections,
    then concatenate and project back.
    """
    d_model = X.shape[1]
    d_head = d_model // num_heads
    heads = []

    for h in range(num_heads):
        np.random.seed(h)
        Wq = np.random.randn(d_model, d_head) * 0.1
        Wk = np.random.randn(d_model, d_head) * 0.1
        Wv = np.random.randn(d_model, d_head) * 0.1

        Q_h = X @ Wq
        K_h = X @ Wk
        V_h = X @ Wv

        out_h, _ = scaled_dot_product_attention(Q_h, K_h, V_h)
        heads.append(out_h)

    # Concatenate all heads
    concat = np.concatenate(heads, axis=-1)  # (seq_len, d_model)

    # Final linear projection
    W_O = np.random.randn(d_model, d_model) * 0.1
    return concat @ W_O


print("\n=== Multi-Head Attention (2 heads) ===")
mha_output = multi_head_attention(X, num_heads=2)
print(f"Input shape:  {X.shape}")
print(f"Output shape: {mha_output.shape}")
print("Shapes match — multi-head attention is a drop-in replacement for single-head.")
