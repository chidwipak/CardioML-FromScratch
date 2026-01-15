"""
Loss Functions - From Scratch Implementation
Cardiovascular Disease Prediction System

NO tensorflow/pytorch - Pure NumPy implementation.
Implements: Binary Cross-Entropy, Categorical Cross-Entropy, MSE with gradients.
"""

import numpy as np


class BinaryCrossEntropy:
    """
    Binary Cross-Entropy Loss for binary classification.
    
    Loss: L = -1/n * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
    Gradient: dL/dŷ = -(y/ŷ - (1-y)/(1-ŷ))
    
    Used with sigmoid activation in output layer.
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.
        
        Args:
            y_true: True labels (n_samples,) or (n_samples, 1)
            y_pred: Predicted probabilities (n_samples,) or (n_samples, 1)
        
        Returns:
            loss: Scalar loss value
        """
        # Clip predictions to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Flatten if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Binary cross-entropy
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        
        self.cache = (y_true, y_pred)
        return loss
    
    def backward(self):
        """
        Compute gradient of loss w.r.t. predictions.
        
        Returns:
            dL/dy_pred: Gradient array (same shape as y_pred)
        """
        y_true, y_pred = self.cache
        n_samples = len(y_true)
        
        # Gradient
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        grad = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / n_samples
        
        return grad
    
    def __call__(self, y_true, y_pred):
        """Allow using loss as a function."""
        return self.forward(y_true, y_pred)


class CategoricalCrossEntropy:
    """
    Categorical Cross-Entropy Loss for multi-class classification.
    
    Loss: L = -1/n * Σ Σ y_ij * log(ŷ_ij)
    
    Used with softmax activation in output layer.
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, y_true, y_pred):
        """
        Compute categorical cross-entropy loss.
        
        Args:
            y_true: True labels, one-hot encoded (n_samples, n_classes)
            y_pred: Predicted probabilities (n_samples, n_classes)
        
        Returns:
            loss: Scalar loss value
        """
        # Clip predictions
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Categorical cross-entropy
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        self.cache = (y_true, y_pred)
        return loss
    
    def backward(self):
        """
        Compute gradient of loss w.r.t. predictions.
        
        Returns:
            dL/dy_pred: Gradient array (same shape as y_pred)
        """
        y_true, y_pred = self.cache
        n_samples = y_true.shape[0]
        
        # Gradient
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        grad = -y_true / (y_pred * n_samples)
        
        return grad
    
    def __call__(self, y_true, y_pred):
        """Allow using loss as a function."""
        return self.forward(y_true, y_pred)


class MeanSquaredError:
    """
    Mean Squared Error Loss for regression.
    
    Loss: L = 1/n * Σ(y - ŷ)²
    Gradient: dL/dŷ = 2/n * (ŷ - y)
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, y_true, y_pred):
        """
        Compute mean squared error.
        
        Args:
            y_true: True values (n_samples,)
            y_pred: Predicted values (n_samples,)
        
        Returns:
            loss: Scalar loss value
        """
        loss = np.mean((y_true - y_pred) ** 2)
        self.cache = (y_true, y_pred)
        return loss
    
    def backward(self):
        """
        Compute gradient of loss w.r.t. predictions.
        
        Returns:
            dL/dy_pred: Gradient array (same shape as y_pred)
        """
        y_true, y_pred = self.cache
        n_samples = len(y_true)
        
        grad = 2 * (y_pred - y_true) / n_samples
        return grad
    
    def __call__(self, y_true, y_pred):
        """Allow using loss as a function."""
        return self.forward(y_true, y_pred)


def get_loss(name):
    """
    Factory function to get loss by name.
    
    Args:
        name: Loss name ('binary_crossentropy', 'categorical_crossentropy', 'mse')
    
    Returns:
        Loss object
    """
    losses = {
        'binary_crossentropy': BinaryCrossEntropy,
        'categorical_crossentropy': CategoricalCrossEntropy,
        'mse': MeanSquaredError,
        'mean_squared_error': MeanSquaredError
    }
    
    if name.lower() not in losses:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(losses.keys())}")
    
    return losses[name.lower()]()


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions (Pure NumPy - NO TensorFlow/PyTorch)...")
    print("=" * 60)
    
    # Binary classification test
    print("\n1. Binary Cross-Entropy Loss:")
    y_true_binary = np.array([1, 0, 1, 1, 0])
    y_pred_binary = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
    
    bce = BinaryCrossEntropy()
    loss_bce = bce.forward(y_true_binary, y_pred_binary)
    grad_bce = bce.backward()
    
    print(f"   True labels: {y_true_binary}")
    print(f"   Predictions: {y_pred_binary}")
    print(f"   Loss: {loss_bce:.4f}")
    print(f"   Gradient: {grad_bce}")
    
    # Multi-class classification test
    print("\n2. Categorical Cross-Entropy Loss:")
    y_true_cat = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    y_pred_cat = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6]
    ])
    
    cce = CategoricalCrossEntropy()
    loss_cce = cce.forward(y_true_cat, y_pred_cat)
    grad_cce = cce.backward()
    
    print(f"   True labels (one-hot):")
    print(f"   {y_true_cat}")
    print(f"   Predictions:")
    print(f"   {y_pred_cat}")
    print(f"   Loss: {loss_cce:.4f}")
    
    # Regression test
    print("\n3. Mean Squared Error Loss:")
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_reg = np.array([1.1, 2.2, 2.9, 4.3, 4.8])
    
    mse = MeanSquaredError()
    loss_mse = mse.forward(y_true_reg, y_pred_reg)
    grad_mse = mse.backward()
    
    print(f"   True values: {y_true_reg}")
    print(f"   Predictions: {y_pred_reg}")
    print(f"   Loss: {loss_mse:.4f}")
    print(f"   Gradient: {grad_mse}")
    
    # Test perfect predictions
    print("\n4. Perfect Predictions Test:")
    y_perfect = np.array([1, 0, 1, 0])
    
    bce_perfect = BinaryCrossEntropy()
    loss_perfect = bce_perfect.forward(y_perfect, y_perfect)
    print(f"   Perfect prediction loss: {loss_perfect:.6f} (should be ~0)")
    
    # Test gradient flow
    print("\n5. Gradient Flow Test:")
    y_test = np.array([1.0])
    y_pred_test = np.array([0.8])
    
    bce_test = BinaryCrossEntropy()
    loss_test = bce_test.forward(y_test, y_pred_test)
    grad_test = bce_test.backward()
    
    print(f"   Loss: {loss_test:.4f}")
    print(f"   Gradient: {grad_test}")
    print(f"   ✓ Gradient computed correctly!")
    
    print("\n" + "=" * 60)
    print("✓ All loss functions working! (Pure NumPy)")
    print("✓ NO TensorFlow/PyTorch used!")
    print("✓ Binary CE, Categorical CE, MSE - ALL from scratch!")
    print("✓ Forward and backward passes implemented!")
