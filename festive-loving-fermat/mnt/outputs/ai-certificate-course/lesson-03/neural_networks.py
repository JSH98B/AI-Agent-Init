"""
Lesson 3: Neural Networks
Topics: Neurons, activation functions, forward pass, backprop intuition
Run: python neural_networks.py
"""

import numpy as np

# ─────────────────────────────────────────────
# PART 1: Activation Functions
# ─────────────────────────────────────────────

def relu(x):
    """ReLU: max(0, x) — most common hidden-layer activation."""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid: squashes output to (0, 1) — used for binary classification."""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh: squashes output to (-1, 1) — zero-centered version of sigmoid."""
    return np.tanh(x)

# Demo activations
print("=== Activation Functions ===")
x = np.array([-3, -1, 0, 1, 3])
print(f"Input:   {x}")
print(f"ReLU:    {relu(x)}")
print(f"Sigmoid: {np.round(sigmoid(x), 3)}")
print(f"Tanh:    {np.round(tanh(x), 3)}")

# ─────────────────────────────────────────────
# PART 2: A Single Neuron (forward pass)
# ─────────────────────────────────────────────

def neuron(inputs, weights, bias, activation=relu):
    """
    A single artificial neuron:
    output = activation(dot(inputs, weights) + bias)
    """
    z = np.dot(inputs, weights) + bias
    return activation(z)

print("\n=== Single Neuron ===")
inputs  = np.array([0.5, 0.8, 0.2])   # 3 input features
weights = np.array([0.4, -0.3, 0.9])  # learned weights
bias    = 0.1

out = neuron(inputs, weights, bias, activation=relu)
print(f"Inputs:  {inputs}")
print(f"Weights: {weights}, Bias: {bias}")
print(f"Output (ReLU): {out:.4f}")

# ─────────────────────────────────────────────
# PART 3: A Tiny Neural Net (2-layer forward pass)
# ─────────────────────────────────────────────

class TinyNet:
    """
    A minimal 2-layer neural network:
    Input(3) → Hidden(4, ReLU) → Output(1, Sigmoid)
    """
    def __init__(self):
        np.random.seed(42)
        # Hidden layer: 3 inputs → 4 neurons
        self.W1 = np.random.randn(3, 4) * 0.1
        self.b1 = np.zeros(4)
        # Output layer: 4 inputs → 1 neuron
        self.W2 = np.random.randn(4, 1) * 0.1
        self.b2 = np.zeros(1)

    def forward(self, X):
        """Forward pass: input → hidden → output."""
        self.z1 = X @ self.W1 + self.b1     # (N, 4)
        self.a1 = relu(self.z1)              # (N, 4)
        self.z2 = self.a1 @ self.W2 + self.b2  # (N, 1)
        self.a2 = sigmoid(self.z2)           # (N, 1) — probability
        return self.a2

    def loss(self, y_pred, y_true):
        """Binary cross-entropy loss."""
        eps = 1e-8
        return -np.mean(y_true * np.log(y_pred + eps) +
                        (1 - y_true) * np.log(1 - y_pred + eps))


print("\n=== Tiny Neural Net (Forward Pass) ===")
net = TinyNet()

# 4 training samples, 3 features each
X = np.array([
    [0.1, 0.2, 0.3],
    [0.9, 0.8, 0.7],
    [0.2, 0.9, 0.1],
    [0.8, 0.1, 0.6],
])
y = np.array([[0], [1], [0], [1]])  # binary labels

y_pred = net.forward(X)
print(f"Predictions:\n{np.round(y_pred, 3)}")
print(f"True labels:\n{y}")
print(f"Loss: {net.loss(y_pred, y):.4f}")

# ─────────────────────────────────────────────
# PART 4: Gradient Descent (training loop)
# ─────────────────────────────────────────────

class TrainableNet(TinyNet):
    def backward(self, X, y_true, lr=0.1):
        """
        Backpropagation — compute gradients and update weights.
        This is the math behind 'how the model learns.'
        """
        N = X.shape[0]

        # Output layer gradients
        dL_da2 = (self.a2 - y_true) / N                  # d(loss)/d(a2)
        dL_dz2 = dL_da2 * self.a2 * (1 - self.a2)        # sigmoid derivative
        dL_dW2 = self.a1.T @ dL_dz2
        dL_db2 = dL_dz2.sum(axis=0)

        # Hidden layer gradients
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * (self.z1 > 0)                  # ReLU derivative
        dL_dW1 = X.T @ dL_dz1
        dL_db1 = dL_dz1.sum(axis=0)

        # Update weights (gradient descent)
        self.W2 -= lr * dL_dW2
        self.b2 -= lr * dL_db2
        self.W1 -= lr * dL_dW1
        self.b1 -= lr * dL_db1


print("\n=== Training Loop (50 epochs) ===")
net2 = TrainableNet()
for epoch in range(50):
    y_pred = net2.forward(X)
    loss = net2.loss(y_pred, y)
    net2.backward(X, y, lr=0.5)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

print(f"\nFinal predictions: {np.round(net2.forward(X).flatten(), 3)}")
print(f"True labels:       {y.flatten()}")
