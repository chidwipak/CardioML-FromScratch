"""
Multi-Layer Perceptron (MLP) Neural Network
============================================

A complete from-scratch implementation of a feedforward neural network
with backpropagation for binary classification.

Features:
- Configurable hidden layer architecture
- Multiple activation functions (ReLU, Sigmoid, Tanh)
- Binary cross-entropy loss with gradient descent
- Mini-batch training with shuffling

Implementation: Pure NumPy (no TensorFlow/PyTorch/Keras)

Author: Mohanganesh
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from src.models.deep_learning.layers import DenseLayer
from src.models.deep_learning.activations import ReLU, Sigmoid, Tanh, get_activation
from src.models.deep_learning.losses import BinaryCrossEntropy, get_loss


class MLPClassifier:
    """
    Multi-Layer Perceptron for binary/multi-class classification.
    
    Architecture:
    - Input layer
    - Multiple hidden layers with configurable activation
    - Output layer with sigmoid (binary) or softmax (multi-class)
    
    NO TensorFlow/PyTorch - Pure NumPy implementation per project requirements.
    
    Parameters:
    -----------
    hidden_layers : list of int
        Number of neurons in each hidden layer. E.g., [64, 32, 16]
    activation : str, default='relu'
        Activation for hidden layers ('relu', 'sigmoid', 'tanh')
    learning_rate : float, default=0.001
        Learning rate for gradient descent
    n_epochs : int, default=100
        Number of training epochs
    batch_size : int, default=32
        Mini-batch size (None = full batch)
    loss : str, default='binary_crossentropy'
        Loss function
    verbose : bool, default=False
        Print training progress
    random_state : int, default=None
        Random seed
    """
    
    def __init__(self, hidden_layers=[64, 32], activation='relu',
                 learning_rate=0.001, n_epochs=100, batch_size=32,
                 loss='binary_crossentropy', verbose=False, random_state=None):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.loss_name = loss
        self.verbose = verbose
        self.random_state = random_state
        
        # Network layers (built during fit)
        self.layers = []
        
        # Loss function
        self.loss_fn = get_loss(loss)
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': []
        }
        
        # Random state
        self.rng = np.random.RandomState(random_state)
    
    def _build_network(self, n_input, n_output):
        """
        Build neural network architecture.
        
        Args:
            n_input: Number of input features
            n_output: Number of output classes
        """
        self.layers = []
        
        # Input -> First hidden layer
        prev_size = n_input
        
        # Hidden layers
        for i, hidden_size in enumerate(self.hidden_layers):
            activation_fn = get_activation(self.activation)
            
            layer = DenseLayer(
                n_input=prev_size,
                n_output=hidden_size,
                activation=activation_fn,
                weight_init='he' if self.activation == 'relu' else 'xavier',
                random_state=self.rng.randint(0, 10000)
            )
            
            self.layers.append(layer)
            prev_size = hidden_size
        
        # Output layer
        if n_output == 1 or n_output == 2:
            # Binary classification - single neuron with sigmoid
            output_activation = Sigmoid()
            output_size = 1
        else:
            # Multi-class - softmax
            output_activation = get_activation('softmax')
            output_size = n_output
        
        output_layer = DenseLayer(
            n_input=prev_size,
            n_output=output_size,
            activation=output_activation,
            weight_init='xavier',
            random_state=self.rng.randint(0, 10000)
        )
        
        self.layers.append(output_layer)
    
    def forward(self, X):
        """
        Forward pass through entire network.
        
        Args:
            X: Input (batch_size, n_features)
        
        Returns:
            Output from final layer (batch_size, n_output)
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def backward(self, y_true, y_pred):
        """
        Backward pass through entire network.
        
        Computes gradients for all layers using backpropagation.
        
        Args:
            y_true: True labels
            y_pred: Predicted values
        """
        # Compute loss gradient
        self.loss_fn.forward(y_true, y_pred)
        dA = self.loss_fn.backward()
        
        # Ensure correct shape
        if len(dA.shape) == 1:
            dA = dA.reshape(-1, 1).T
        
        # Backpropagate through layers in reverse
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
    
    def update_weights(self):
        """
        Update all layer weights using computed gradients.
        """
        for layer in self.layers:
            layer.update_weights(self.learning_rate)
    
    def fit(self, X, y):
        """
        Train MLP using mini-batch gradient descent.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        
        Returns:
            self
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Determine number of output classes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        # Build network
        self._build_network(n_features, n_classes)
        
        # Training loop
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            n_batches = 0
            
            # Mini-batch training
            if self.batch_size is None:
                batch_size = n_samples
            else:
                batch_size = self.batch_size
            
            # Shuffle data
            indices = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process mini-batches
            for i in range(0, n_samples, batch_size):
                # Get batch
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = self.loss_fn.forward(y_batch, y_pred)
                epoch_loss += loss
                
                # Compute accuracy
                if n_classes == 2:
                    predictions = (y_pred.flatten() >= 0.5).astype(int)
                else:
                    predictions = np.argmax(y_pred, axis=1)
                
                epoch_correct += np.sum(predictions == y_batch)
                
                # Backward pass
                self.backward(y_batch, y_pred)
                
                # Update weights
                self.update_weights()
                
                n_batches += 1
            
            # Average loss and accuracy
            avg_loss = epoch_loss / n_batches
            accuracy = epoch_correct / n_samples
            
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(accuracy)
            
            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        if self.verbose:
            print(f"Training complete! Final loss: {self.history['loss'][-1]:.4f}, Final accuracy: {self.history['accuracy'][-1]:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Probabilities (n_samples,) for binary, (n_samples, n_classes) for multi-class
        """
        if not self.layers:
            raise ValueError("Model not trained! Call fit() first.")
        
        X = np.array(X)
        probabilities = self.forward(X)
        
        return probabilities.flatten() if probabilities.shape[1] == 1 else probabilities
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Args:
            X: Features (n_samples, n_features)
            threshold: Decision threshold for binary classification
        
        Returns:
            Predicted labels (n_samples,)
        """
        probabilities = self.predict_proba(X)
        
        if len(probabilities.shape) == 1:
            # Binary classification
            predictions = (probabilities >= threshold).astype(int)
        else:
            # Multi-class
            predictions = np.argmax(probabilities, axis=1)
        
        return predictions
    
    def get_architecture(self):
        """
        Get network architecture summary.
        
        Returns:
            List of layer configurations
        """
        arch = []
        for i, layer in enumerate(self.layers):
            arch.append({
                'layer': i,
                'type': 'Dense',
                'input_size': layer.n_input,
                'output_size': layer.n_output,
                'activation': layer.activation.__class__.__name__ if layer.activation else 'None'
            })
        return arch


if __name__ == "__main__":
    # Test MLP
    print("Testing MLP Neural Network (Pure NumPy - NO TensorFlow/PyTorch)...")
    print("=" * 60)
    
    # Generate synthetic binary classification data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create non-linearly separable data
    X = np.random.randn(n_samples, n_features)
    
    # Complex decision boundary
    y = (
        (np.sum(X[:, :5] ** 2, axis=1) > 5) |
        (np.sum(X[:, 5:10], axis=1) > 2)
    ).astype(int)
    
    print(f"\n1. Generated synthetic data:")
    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Class distribution: {np.bincount(y)}")
    
    # Split data
    from src.utils.train_test_split import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    print(f"\n2. Data split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Features standardized: ✓")
    
    # Build and train MLP
    print(f"\n3. Building MLP Neural Network...")
    mlp = MLPClassifier(
        hidden_layers=[64, 32],
        activation='relu',
        learning_rate=0.1,
        n_epochs=100,
        batch_size=64,
        loss='binary_crossentropy',
        verbose=True,
        random_state=42
    )
    
    print(f"\n4. Network Architecture:")
    mlp._build_network(n_features, 2)
    arch = mlp.get_architecture()
    for layer_info in arch:
        print(f"   Layer {layer_info['layer']}: {layer_info['input_size']} -> {layer_info['output_size']} ({layer_info['activation']})")
    
    print(f"\n5. Training MLP...")
    mlp.fit(X_train, y_train)
    
    # Make predictions
    print(f"\n6. Making predictions...")
    y_pred_proba = mlp.predict_proba(X_test)
    y_pred = mlp.predict(X_test)
    
    print(f"   Sample probabilities: {y_pred_proba[:5]}")
    print(f"   Sample predictions: {y_pred[:5]}")
    print(f"   True labels: {y_test[:5]}")
    
    # Evaluate
    from src.core.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n7. Test Set Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ MLP Neural Network working! (Pure NumPy)")
    print("✓ NO TensorFlow/PyTorch/Keras used!")
    print("✓ Multi-layer architecture with backpropagation - ALL from scratch!")
    print("✓ TIER 1 Deep Learning component complete!")
