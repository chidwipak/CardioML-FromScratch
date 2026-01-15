"""
Neural Network Layers - From Scratch Implementation
Cardiovascular Disease Prediction System

NO tensorflow/pytorch - Pure NumPy implementation.
Implements: Dense (fully connected) layer with forward and backward passes.
"""

import numpy as np


class DenseLayer:
    """
    Fully connected (dense) neural network layer.
    
    Computes: Z = W·X + b
    
    Where:
    - W: weight matrix (n_output, n_input)
    - X: input (batch_size, n_input)
    - b: bias vector (n_output,)
    - Z: output (batch_size, n_output)
    
    NO TensorFlow/PyTorch - Pure NumPy implementation.
    """
    
    def __init__(self, n_input, n_output, activation=None, 
                 weight_init='he', random_state=None):
        """
        Initialize dense layer.
        
        Args:
            n_input: Number of input features
            n_output: Number of output neurons
            activation: Activation function object (optional)
            weight_init: Weight initialization method ('he', 'xavier', 'random')
            random_state: Random seed
        """
        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation
        
        # Random number generator
        self.rng = np.random.RandomState(random_state)
        
        # Initialize weights
        self.W, self.b = self._initialize_weights(weight_init)
        
        # Cache for backward pass
        self.cache = {}
        
        # Gradients (computed during backward pass)
        self.dW = None
        self.db = None
    
    def _initialize_weights(self, method):
        """
        Initialize weights using specified method.
        
        Args:
            method: 'he' (He initialization), 'xavier' (Xavier/Glorot), or 'random'
        
        Returns:
            W, b: Initialized weight matrix and bias vector
        """
        if method == 'he':
            # He initialization (good for ReLU)
            W = self.rng.randn(self.n_output, self.n_input) * np.sqrt(2.0 / self.n_input)
        
        elif method == 'xavier':
            # Xavier initialization (good for Sigmoid/Tanh)
            W = self.rng.randn(self.n_output, self.n_input) * np.sqrt(1.0 / self.n_input)
        
        elif method == 'random':
            # Small random values
            W = self.rng.randn(self.n_output, self.n_input) * 0.01
        
        else:
            raise ValueError(f"Unknown initialization: {method}")
        
        # Bias initialized to zeros
        b = np.zeros((self.n_output, 1))
        
        return W, b
    
    def forward(self, X):
        """
        Forward pass through layer.
        
        Computes: Z = W·X + b, then applies activation if present.
        
        Args:
            X: Input (batch_size, n_input)
        
        Returns:
            A: Activated output (batch_size, n_output)
        """
        # Ensure X is (batch_size, n_input)
        if X.shape[1] != self.n_input:
            X = X.T
        
        # Transpose for matrix multiplication: (n_output, n_input) @ (n_input, batch_size)
        X_T = X.T  # (n_input, batch_size)
        
        # Linear transformation: Z = W·X + b
        Z = np.dot(self.W, X_T) + self.b  # (n_output, batch_size)
        
        # Apply activation if present
        if self.activation is not None:
            A = self.activation.forward(Z)
        else:
            A = Z
        
        # Cache for backward pass
        self.cache['X'] = X_T  # Store as (n_input, batch_size)
        self.cache['Z'] = Z    # (n_output, batch_size)
        self.cache['A'] = A    # (n_output, batch_size)
        
        return A.T  # Return as (batch_size, n_output)
    
    def backward(self, dA):
        """
        Backward pass through layer.
        
        Computes gradients w.r.t. weights, bias, and input.
        
        Args:
            dA: Gradient of loss w.r.t. layer output (batch_size, n_output)
        
        Returns:
            dX: Gradient of loss w.r.t. layer input (batch_size, n_input)
        """
        # Get cached values
        X = self.cache['X']  # (n_input, batch_size)
        Z = self.cache['Z']  # (n_output, batch_size)
        
        # Ensure dA has correct shape: (batch_size, n_output) -> (n_output, batch_size)
        if dA.ndim == 1:
            dA = dA.reshape(-1, 1)
        if dA.shape[0] == X.shape[1]:  # If (batch_size, n_output)
            dA = dA.T
        
        batch_size = X.shape[1]
        
        # Backprop through activation
        if self.activation is not None:
            dZ = self.activation.backward(dA)
        else:
            dZ = dA
        
        # Compute gradients
        # dW = dZ · X^T / batch_size
        self.dW = np.dot(dZ, X.T) / batch_size
        
        # db = sum(dZ) / batch_size
        self.db = np.sum(dZ, axis=1, keepdims=True) / batch_size
        
        # dX = W^T · dZ
        dX = np.dot(self.W.T, dZ)
        
        return dX.T  # Return as (batch_size, n_input)
    
    def update_weights(self, learning_rate):
        """
        Update weights using computed gradients.
        
        Args:
            learning_rate: Learning rate (step size)
        """
        if self.dW is None or self.db is None:
            raise ValueError("Gradients not computed! Call backward() first.")
        
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
    
    def get_weights(self):
        """Get current weights and bias."""
        return self.W.copy(), self.b.copy()
    
    def set_weights(self, W, b):
        """Set weights and bias."""
        if W.shape != (self.n_output, self.n_input):
            raise ValueError(f"Weight shape mismatch. Expected {(self.n_output, self.n_input)}, got {W.shape}")
        if b.shape != (self.n_output, 1):
            raise ValueError(f"Bias shape mismatch. Expected {(self.n_output, 1)}, got {b.shape}")
        
        self.W = W.copy()
        self.b = b.copy()


if __name__ == "__main__":
    # Test Dense Layer
    print("Testing Dense Layer (Pure NumPy - NO TensorFlow/PyTorch)...")
    print("=" * 60)
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from activations import ReLU, Sigmoid
    
    # Create test data
    np.random.seed(42)
    batch_size = 3
    n_input = 4
    n_output = 2
    
    X = np.random.randn(batch_size, n_input)
    
    print(f"\n1. Test Data:")
    print(f"   Batch size: {batch_size}")
    print(f"   Input features: {n_input}")
    print(f"   Output neurons: {n_output}")
    print(f"   Input X shape: {X.shape}")
    print(f"   Input X:\n{X}")
    
    # Test forward pass without activation
    print(f"\n2. Dense Layer (no activation):")
    layer1 = DenseLayer(n_input, n_output, activation=None, random_state=42)
    Z = layer1.forward(X)
    print(f"   Output Z shape: {Z.shape}")
    print(f"   Output Z:\n{Z}")
    
    # Test forward pass with ReLU
    print(f"\n3. Dense Layer (ReLU activation):")
    layer2 = DenseLayer(n_input, n_output, activation=ReLU(), random_state=42)
    A_relu = layer2.forward(X)
    print(f"   Output A shape: {A_relu.shape}")
    print(f"   Output A:\n{A_relu}")
    
    # Test backward pass
    print(f"\n4. Backward Pass:")
    dA = np.ones_like(A_relu)
    dX = layer2.backward(dA)
    print(f"   Upstream gradient dA shape: {dA.shape}")
    print(f"   Computed dX shape: {dX.shape}")
    print(f"   Weight gradient dW shape: {layer2.dW.shape}")
    print(f"   Bias gradient db shape: {layer2.db.shape}")
    print(f"   ✓ Gradients computed!")
    
    # Test weight update
    print(f"\n5. Weight Update:")
    W_before, b_before = layer2.get_weights()
    print(f"   W before update (first 2 values): {W_before[0, :2]}")
    print(f"   b before update: {b_before.flatten()}")
    
    layer2.update_weights(learning_rate=0.01)
    W_after, b_after = layer2.get_weights()
    print(f"   W after update (first 2 values): {W_after[0, :2]}")
    print(f"   b after update: {b_after.flatten()}")
    print(f"   ✓ Weights updated!")
    
    # Test He initialization
    print(f"\n6. Weight Initialization Methods:")
    layer_he = DenseLayer(10, 5, weight_init='he', random_state=42)
    layer_xavier = DenseLayer(10, 5, weight_init='xavier', random_state=42)
    
    W_he, _ = layer_he.get_weights()
    W_xavier, _ = layer_xavier.get_weights()
    
    print(f"   He init - Weight std: {np.std(W_he):.4f}")
    print(f"   Xavier init - Weight std: {np.std(W_xavier):.4f}")
    print(f"   ✓ Different initializations working!")
    
    # Test gradient flow through multiple layers
    print(f"\n7. Multi-Layer Gradient Flow:")
    layer_a = DenseLayer(4, 3, activation=ReLU(), random_state=42)
    layer_b = DenseLayer(3, 2, activation=Sigmoid(), random_state=42)
    
    # Forward
    X_test = np.random.randn(2, 4)
    A1 = layer_a.forward(X_test)
    A2 = layer_b.forward(A1)
    
    # Backward
    dA2 = np.ones_like(A2)
    dA1 = layer_b.backward(dA2)
    dX_test = layer_a.backward(dA1)
    
    print(f"   Input shape: {X_test.shape}")
    print(f"   Hidden layer output shape: {A1.shape}")
    print(f"   Final output shape: {A2.shape}")
    print(f"   Gradient flows back to input shape: {dX_test.shape}")
    print(f"   ✓ Multi-layer gradient flow working!")
    
    print("\n" + "=" * 60)
    print("✓ Dense Layer working! (Pure NumPy)")
    print("✓ NO TensorFlow/PyTorch used!")
    print("✓ Forward pass, backward pass, weight updates - ALL from scratch!")
    print("✓ He and Xavier initialization implemented!")
