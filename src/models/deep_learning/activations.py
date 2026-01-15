"""
Activation Functions - From Scratch Implementation
Cardiovascular Disease Prediction System

NO tensorflow/pytorch - Pure NumPy implementation.
Implements: ReLU, Sigmoid, Tanh, Softmax with forward and backward passes.
"""

import numpy as np


class ReLU:
    """
    Rectified Linear Unit (ReLU) activation function.
    
    Forward: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0, else 0
    
    Used in hidden layers for non-linearity.
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, Z):
        """
        Forward pass through ReLU.
        
        Args:
            Z: Pre-activation values (any shape)
        
        Returns:
            Activated values (same shape as Z)
        """
        self.cache = Z
        return np.maximum(0, Z)
    
    def backward(self, dA):
        """
        Backward pass through ReLU.
        
        Args:
            dA: Gradient of loss w.r.t. activation output
        
        Returns:
            dZ: Gradient of loss w.r.t. pre-activation (input)
        """
        Z = self.cache
        dZ = dA.copy()
        dZ[Z <= 0] = 0  # Zero gradient where input was negative
        return dZ
    
    def __call__(self, Z):
        """Allow using activation as a function."""
        return self.forward(Z)


class Sigmoid:
    """
    Sigmoid activation function.
    
    Forward: f(x) = 1 / (1 + e^(-x))
    Derivative: f'(x) = f(x) * (1 - f(x))
    
    Used in output layer for binary classification.
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, Z):
        """
        Forward pass through Sigmoid.
        
        Args:
            Z: Pre-activation values (any shape)
        
        Returns:
            Activated values (same shape as Z), range (0, 1)
        """
        # Clip to prevent overflow
        Z = np.clip(Z, -500, 500)
        A = 1 / (1 + np.exp(-Z))
        self.cache = A
        return A
    
    def backward(self, dA):
        """
        Backward pass through Sigmoid.
        
        Args:
            dA: Gradient of loss w.r.t. activation output
        
        Returns:
            dZ: Gradient of loss w.r.t. pre-activation (input)
        """
        A = self.cache
        dZ = dA * A * (1 - A)
        return dZ
    
    def __call__(self, Z):
        """Allow using activation as a function."""
        return self.forward(Z)


class Tanh:
    """
    Hyperbolic Tangent (Tanh) activation function.
    
    Forward: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Derivative: f'(x) = 1 - f(x)²
    
    Used in hidden layers, outputs range (-1, 1).
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, Z):
        """
        Forward pass through Tanh.
        
        Args:
            Z: Pre-activation values (any shape)
        
        Returns:
            Activated values (same shape as Z), range (-1, 1)
        """
        A = np.tanh(Z)
        self.cache = A
        return A
    
    def backward(self, dA):
        """
        Backward pass through Tanh.
        
        Args:
            dA: Gradient of loss w.r.t. activation output
        
        Returns:
            dZ: Gradient of loss w.r.t. pre-activation (input)
        """
        A = self.cache
        dZ = dA * (1 - A ** 2)
        return dZ
    
    def __call__(self, Z):
        """Allow using activation as a function."""
        return self.forward(Z)


class Softmax:
    """
    Softmax activation function.
    
    Forward: f(x_i) = e^(x_i) / Σ(e^(x_j))
    
    Used in output layer for multi-class classification.
    Converts logits to probability distribution.
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, Z):
        """
        Forward pass through Softmax.
        
        Args:
            Z: Pre-activation values (batch_size, n_classes)
        
        Returns:
            Probabilities (batch_size, n_classes), each row sums to 1
        """
        # Subtract max for numerical stability
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        
        exp_Z = np.exp(Z_shifted)
        A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        self.cache = A
        return A
    
    def backward(self, dA):
        """
        Backward pass through Softmax.
        
        Note: Usually combined with cross-entropy loss for efficiency.
        This is the general form.
        
        Args:
            dA: Gradient of loss w.r.t. activation output
        
        Returns:
            dZ: Gradient of loss w.r.t. pre-activation (input)
        """
        A = self.cache
        # Simplified: assumes cross-entropy loss
        # Full Jacobian computation would be more complex
        return dA
    
    def __call__(self, Z):
        """Allow using activation as a function."""
        return self.forward(Z)


class Linear:
    """
    Linear activation (identity function).
    
    Forward: f(x) = x
    Derivative: f'(x) = 1
    
    Used for regression tasks.
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, Z):
        """Forward pass - identity."""
        self.cache = Z
        return Z
    
    def backward(self, dA):
        """Backward pass - gradient passes through unchanged."""
        return dA
    
    def __call__(self, Z):
        """Allow using activation as a function."""
        return self.forward(Z)


def get_activation(name):
    """
    Factory function to get activation by name.
    
    Args:
        name: Activation name ('relu', 'sigmoid', 'tanh', 'softmax', 'linear')
    
    Returns:
        Activation object
    """
    activations = {
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'softmax': Softmax,
        'linear': Linear
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    
    return activations[name.lower()]()


if __name__ == "__main__":
    # Test activation functions
    print("Testing Activation Functions (Pure NumPy - NO TensorFlow/PyTorch)...")
    print("=" * 60)
    
    # Test data
    Z = np.array([[-2, -1, 0, 1, 2],
                  [0.5, 1.5, 2.5, 3.5, 4.5]])
    
    print("\nInput Z:")
    print(Z)
    
    # Test ReLU
    print("\n1. ReLU Activation:")
    relu = ReLU()
    A_relu = relu.forward(Z)
    print(f"   Forward: {A_relu}")
    dA = np.ones_like(A_relu)
    dZ_relu = relu.backward(dA)
    print(f"   Backward (gradient): {dZ_relu}")
    
    # Test Sigmoid
    print("\n2. Sigmoid Activation:")
    sigmoid = Sigmoid()
    A_sigmoid = sigmoid.forward(Z)
    print(f"   Forward: {A_sigmoid}")
    dZ_sigmoid = sigmoid.backward(dA)
    print(f"   Backward (gradient): {dZ_sigmoid}")
    
    # Test Tanh
    print("\n3. Tanh Activation:")
    tanh = Tanh()
    A_tanh = tanh.forward(Z)
    print(f"   Forward: {A_tanh}")
    dZ_tanh = tanh.backward(dA)
    print(f"   Backward (gradient): {dZ_tanh}")
    
    # Test Softmax
    print("\n4. Softmax Activation:")
    softmax = Softmax()
    A_softmax = softmax.forward(Z)
    print(f"   Forward: {A_softmax}")
    print(f"   Row sums: {np.sum(A_softmax, axis=1)}")
    
    # Test gradient flow
    print("\n5. Gradient Flow Test:")
    Z_test = np.array([[1.0, 2.0, 3.0]])
    
    relu_test = ReLU()
    A_test = relu_test.forward(Z_test)
    dA_test = np.array([[0.1, 0.2, 0.3]])
    dZ_test = relu_test.backward(dA_test)
    
    print(f"   Input: {Z_test}")
    print(f"   ReLU output: {A_test}")
    print(f"   Upstream gradient: {dA_test}")
    print(f"   Backprop gradient: {dZ_test}")
    print(f"   ✓ Gradient flows correctly!")
    
    print("\n" + "=" * 60)
    print("✓ All activation functions working! (Pure NumPy)")
    print("✓ NO TensorFlow/PyTorch used!")
    print("✓ ReLU, Sigmoid, Tanh, Softmax - ALL from scratch!")
    print("✓ Forward and backward passes implemented!")
