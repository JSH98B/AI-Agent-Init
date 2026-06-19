# Lesson 3: Neural Networks

## Setup
```bash
pip install -r requirements.txt
python neural_networks.py
```

## What This Code Covers
- Activation functions: ReLU, Sigmoid, Tanh
- A single artificial neuron (weighted sum + activation)
- Forward pass through a 2-layer network
- Backpropagation and gradient descent training loop

## Key Insight
Every modern LLM is a neural network — just much deeper and trained on text instead of labeled tabular data. The core math (matrix multiply → activation → loss → gradient → update) is identical.
