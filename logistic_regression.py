"""
Logistic Regression Classifier - From Scratch Implementation
Cardiovascular Disease Prediction System

NO sklearn.linear_model - Pure NumPy implementation with gradient descent.
Implements: Sigmoid transformation, L2 regularization, probabilistic predictions.
"""

import numpy as np


class LogisticRegression:
    """
    Logistic Regression for binary classification.
    
    Uses gradient descent optimization with L2 regularization.
    NO sklearn - Pure NumPy implementation per project requirements.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Step size for gradient descent
    n_iterations : int, default=1000
        Number of gradient descent iterations
    regularization : float, default=0.01
        L2 regularization strength (lambda)
    verbose : bool, default=False
        Print training progress
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 regularization=0.01, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization  # L2 penalty
        self.verbose = verbose
        
        # Model parameters (learned during training)
        self.weights = None
        self.bias = None
        
        # Training history
        self.loss_history = []
    
    def sigmoid(self, z):
        """
        Sigmoid activation function.
        
        Formula: σ(z) = 1 / (1 + e^(-z))
        
        Args:
            z: Linear combination (numpy array)
        
        Returns:
            Sigmoid output between 0 and 1
        """
        # Clip z to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_true, y_pred_proba):
        """
        Compute binary cross-entropy loss with L2 regularization.
        
        Loss = -1/n * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)] + λ/2 * ||w||²
        
        Args:
            y_true: True labels (n_samples,)
            y_pred_proba: Predicted probabilities (n_samples,)
        
        Returns:
            Total loss (float)
        """
        n_samples = len(y_true)
        
        # Clip predictions to prevent log(0)
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        
        # Binary cross-entropy loss
        bce_loss = -np.mean(
            y_true * np.log(y_pred_proba) + 
            (1 - y_true) * np.log(1 - y_pred_proba)
        )
        
        # L2 regularization term
        l2_penalty = (self.regularization / 2) * np.sum(self.weights ** 2)
        
        total_loss = bce_loss + l2_penalty
        
        return total_loss
    
    def fit(self, X, y):
        """
        Train logistic regression using gradient descent.
        
        Updates weights and bias to minimize loss function.
        NO sklearn - Pure NumPy gradient descent implementation.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        
        Returns:
            self
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Gradient descent optimization
        for iteration in range(self.n_iterations):
            # Forward pass
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred_proba = self.sigmoid(linear_output)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred_proba)
            self.loss_history.append(loss)
            
            # Backward pass - compute gradients
            error = y_pred_proba - y
            
            # Gradient for weights (with L2 regularization)
            dw = (1 / n_samples) * np.dot(X.T, error) + self.regularization * self.weights
            
            # Gradient for bias (no regularization on bias)
            db = (1 / n_samples) * np.sum(error)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Loss: {loss:.4f}")
        
        if self.verbose:
            print(f"Training complete! Final loss: {self.loss_history[-1]:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Probability of positive class (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model not trained yet! Call fit() first.")
        
        X = np.array(X)
        linear_output = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(linear_output)
        
        return probabilities
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Args:
            X: Features (n_samples, n_features)
            threshold: Decision threshold (default 0.5)
        
        Returns:
            Predicted class labels (n_samples,)
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions
    
    def get_coefficients(self):
        """
        Get model coefficients (weights and bias).
        
        Returns:
            dict: {'weights': array, 'bias': float}
        """
        if self.weights is None:
            raise ValueError("Model not trained yet!")
        
        return {
            'weights': self.weights.copy(),
            'bias': self.bias
        }
    
    def feature_importance(self, feature_names=None):
        """
        Get feature importance based on absolute weight values.
        
        Args:
            feature_names: List of feature names (optional)
        
        Returns:
            Sorted list of (feature, importance) tuples
        """
        if self.weights is None:
            raise ValueError("Model not trained yet!")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.weights))]
        
        # Absolute weight values as importance
        importances = np.abs(self.weights)
        
        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1]
        
        feature_importance_list = [
            (feature_names[i], importances[i]) 
            for i in sorted_indices
        ]
        
        return feature_importance_list


if __name__ == "__main__":
    # Test Logistic Regression
    print("Testing Logistic Regression (Pure NumPy - NO sklearn)...")
    print("=" * 60)
    
    # Generate synthetic binary classification data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Create linearly separable data
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([1.5, -2.0, 0.5, -1.0, 0.8])
    true_bias = 0.3
    
    # Generate labels
    linear_combination = np.dot(X, true_weights) + true_bias
    probabilities = 1 / (1 + np.exp(-linear_combination))
    y = (probabilities > 0.5).astype(int)
    
    print(f"\n1. Generated synthetic data:")
    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Class distribution: {np.bincount(y)}")
    
    # Split data
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from utils.train_test_split import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n2. Data split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Train model
    print(f"\n3. Training Logistic Regression...")
    model = LogisticRegression(
        learning_rate=0.01,
        n_iterations=1000,
        regularization=0.01,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    print(f"\n4. Making predictions...")
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    print(f"   Sample probabilities: {y_pred_proba[:5]}")
    print(f"   Sample predictions: {y_pred[:5]}")
    print(f"   True labels: {y_test[:5]}")
    
    # Evaluate
    from core.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n5. Test Set Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Check learned weights
    print(f"\n6. Learned Parameters:")
    coeffs = model.get_coefficients()
    print(f"   Weights: {coeffs['weights']}")
    print(f"   Bias: {coeffs['bias']:.4f}")
    print(f"\n   True weights: {true_weights}")
    print(f"   True bias: {true_bias:.4f}")
    
    # Feature importance
    print(f"\n7. Feature Importance:")
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    importances = model.feature_importance(feature_names)
    for name, importance in importances[:5]:
        print(f"   {name}: {importance:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Logistic Regression working! (Pure NumPy)")
    print("✓ NO sklearn.linear_model used!")
    print("✓ Sigmoid, Gradient Descent, L2 Regularization - ALL from scratch!")
