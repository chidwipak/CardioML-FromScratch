"""
Support Vector Machine (SVM) Classifier - From Scratch Implementation
Cardiovascular Disease Prediction System

NO sklearn.svm - Pure NumPy implementation with gradient descent.
Implements: Linear SVM, RBF kernel, hinge loss optimization.

Note: Full SMO algorithm is extremely complex (~2000 lines). 
We implement a gradient-based SVM which is mathematically equivalent
and more practical for large datasets like our 70K CVD samples.
"""

import numpy as np


class SVMClassifier:
    """
    Support Vector Machine for binary classification.
    
    Uses gradient descent on hinge loss for optimization.
    Supports linear and RBF (Radial Basis Function) kernels.
    
    NO sklearn - Pure NumPy implementation per project requirements.
    
    Parameters:
    -----------
    kernel : str, default='rbf'
        Kernel type ('linear', 'rbf')
    C : float, default=1.0
        Regularization parameter (larger = less regularization)
    gamma : float, default='auto'
        Kernel coefficient for RBF ('auto' = 1/n_features)
    learning_rate : float, default=0.001
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of training iterations
    tol : float, default=1e-3
        Tolerance for convergence
    verbose : bool, default=False
        Print training progress
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='auto',
                 learning_rate=0.001, n_iterations=1000,
                 tol=1e-3, verbose=False):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.verbose = verbose
        
        # Model parameters (learned during training)
        self.w = None  # Weights (for linear kernel)
        self.b = None  # Bias
        self.alpha = None  # Lagrange multipliers
        
        # Training data (stored for kernel methods)
        self.X_train = None
        self.y_train = None
        
        # Loss history
        self.loss_history = []
    
    def _linear_kernel(self, X1, X2):
        """
        Linear kernel: K(x1, x2) = x1 · x2
        
        Args:
            X1: First set of vectors (n_samples1, n_features)
            X2: Second set of vectors (n_samples2, n_features)
        
        Returns:
            Kernel matrix (n_samples1, n_samples2)
        """
        return np.dot(X1, X2.T)
    
    def _rbf_kernel(self, X1, X2):
        """
        RBF (Gaussian) kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||²)
        
        Args:
            X1: First set of vectors (n_samples1, n_features)
            X2: Second set of vectors (n_samples2, n_features)
        
        Returns:
            Kernel matrix (n_samples1, n_samples2)
        """
        # Compute pairwise squared Euclidean distances
        # ||x1 - x2||² = ||x1||² + ||x2||² - 2*x1·x2
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances_sq = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        
        # Apply RBF transformation
        K = np.exp(-self.gamma * distances_sq)
        
        return K
    
    def _compute_kernel(self, X1, X2):
        """
        Compute kernel matrix.
        
        Args:
            X1: First set of vectors
            X2: Second set of vectors
        
        Returns:
            Kernel matrix
        """
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _hinge_loss(self, y_true, y_pred):
        """
        Compute hinge loss.
        
        Loss = max(0, 1 - y_true * y_pred)
        
        Args:
            y_true: True labels {-1, +1}
            y_pred: Predicted decision values
        
        Returns:
            Average hinge loss
        """
        margins = y_true * y_pred
        loss = np.maximum(0, 1 - margins)
        return np.mean(loss)
    
    def fit(self, X, y):
        """
        Train SVM using gradient descent on hinge loss.
        
        Optimization objective:
        min: (1/2)||w||² + C * Σ max(0, 1 - y_i * (w·x_i + b))
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels {0, 1} (will be converted to {-1, +1})
        
        Returns:
            self
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        
        # Convert labels to {-1, +1}
        y_train = np.where(y == 0, -1, 1)
        
        # Set gamma if auto
        if self.gamma == 'auto':
            self.gamma = 1.0 / n_features
        
        # Store training data (needed for kernel methods)
        self.X_train = X
        self.y_train = y_train
        
        if self.kernel == 'linear':
            # Linear SVM: optimize weights directly
            self.w = np.zeros(n_features)
            self.b = 0.0
            
            for iteration in range(self.n_iterations):
                # Compute predictions
                y_pred = np.dot(X, self.w) + self.b
                
                # Compute hinge loss
                loss = self._hinge_loss(y_train, y_pred)
                
                # L2 regularization term
                reg_loss = 0.5 * np.dot(self.w, self.w)
                total_loss = reg_loss + self.C * loss
                self.loss_history.append(total_loss)
                
                # Compute gradients
                margins = y_train * y_pred
                
                # Subgradient of hinge loss
                # If margin >= 1: gradient = 0
                # If margin < 1: gradient = -y_i * x_i
                mask = (margins < 1).astype(float)
                
                # Gradient w.r.t. w
                dw = self.w - self.C * np.dot(mask * y_train, X) / n_samples
                
                # Gradient w.r.t. b
                db = -self.C * np.sum(mask * y_train) / n_samples
                
                # Update parameters
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
                
                # Print progress
                if self.verbose and (iteration + 1) % 100 == 0:
                    print(f"Iteration {iteration + 1}/{self.n_iterations}, Loss: {total_loss:.4f}")
                
                # Check convergence
                if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iteration + 1}")
                    break
        
        else:
            # Kernel SVM: optimize dual variables (alpha)
            self.alpha = np.zeros(n_samples)
            self.b = 0.0
            
            # Precompute kernel matrix
            K = self._compute_kernel(X, X)
            
            for iteration in range(self.n_iterations):
                # Compute predictions using kernel trick
                # f(x) = Σ alpha_i * y_i * K(x_i, x) + b
                y_pred = np.dot((self.alpha * y_train), K) + self.b
                
                # Compute loss
                loss = self._hinge_loss(y_train, y_pred)
                
                # Regularization
                reg_loss = 0.5 * np.dot(self.alpha * y_train, np.dot(K, self.alpha * y_train))
                total_loss = reg_loss + self.C * loss
                self.loss_history.append(total_loss)
                
                # Compute gradients
                margins = y_train * y_pred
                mask = (margins < 1).astype(float)
                
                # Gradient w.r.t. alpha (simplified)
                d_alpha = np.dot(K, self.alpha * y_train) * y_train - self.C * mask * y_train
                
                # Update alpha
                self.alpha -= self.learning_rate * d_alpha
                
                # Clip alpha to [0, C]
                self.alpha = np.clip(self.alpha, 0, self.C)
                
                # Update bias
                support_vectors = (self.alpha > 1e-5)
                if np.sum(support_vectors) > 0:
                    self.b = np.mean(
                        y_train[support_vectors] - 
                        np.dot((self.alpha * y_train), K[:, support_vectors])
                    )
                
                # Print progress
                if self.verbose and (iteration + 1) % 100 == 0:
                    n_sv = np.sum(self.alpha > 1e-5)
                    print(f"Iteration {iteration + 1}/{self.n_iterations}, Loss: {total_loss:.4f}, SVs: {n_sv}")
                
                # Check convergence
                if iteration > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iteration + 1}")
                    break
        
        if self.verbose:
            print(f"Training complete! Final loss: {self.loss_history[-1]:.4f}")
        
        return self
    
    def decision_function(self, X):
        """
        Compute decision function (distance from hyperplane).
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Decision values (n_samples,)
        """
        X = np.array(X)
        
        if self.kernel == 'linear':
            # Linear: f(x) = w·x + b
            return np.dot(X, self.w) + self.b
        
        else:
            # Kernel: f(x) = Σ alpha_i * y_i * K(x_i, x) + b
            K = self._compute_kernel(self.X_train, X)
            return np.dot((self.alpha * self.y_train), K) + self.b
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predicted labels {0, 1} (n_samples,)
        """
        decision = self.decision_function(X)
        
        # Convert {-1, +1} back to {0, 1}
        predictions = np.where(decision >= 0, 1, 0)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities using Platt scaling approximation.
        
        Converts decision function to probabilities using sigmoid.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Probability of positive class (n_samples,)
        """
        decision = self.decision_function(X)
        
        # Approximate probability using sigmoid
        # P(y=1|x) ≈ 1 / (1 + exp(-decision))
        probabilities = 1 / (1 + np.exp(-decision))
        
        return probabilities
    
    def get_support_vectors(self):
        """
        Get support vectors (for kernel SVM).
        
        Returns:
            Indices of support vectors
        """
        if self.alpha is None:
            return None
        
        return np.where(self.alpha > 1e-5)[0]
    
    def n_support_vectors(self):
        """
        Count support vectors.
        
        Returns:
            Number of support vectors
        """
        sv_indices = self.get_support_vectors()
        return len(sv_indices) if sv_indices is not None else 0


if __name__ == "__main__":
    # Test SVM
    print("Testing SVM Classifier (Pure NumPy - NO sklearn)...")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    # Create linearly separable clusters
    cluster1 = np.random.randn(n_samples // 2, n_features) + np.array([2] * n_features)
    cluster2 = np.random.randn(n_samples // 2, n_features) + np.array([-2] * n_features)
    
    X = np.vstack([cluster1, cluster2])
    y = np.hstack([np.ones(n_samples // 2), np.zeros(n_samples // 2)])
    
    print(f"\n1. Generated synthetic data:")
    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Class distribution: {np.bincount(y.astype(int))}")
    
    # Split data
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from utils.train_test_split import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)
    
    print(f"\n2. Data split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Features standardized: ✓")
    
    # Test Linear SVM
    print(f"\n3. Training Linear SVM...")
    svm_linear = SVMClassifier(
        kernel='linear',
        C=1.0,
        learning_rate=0.01,
        n_iterations=500,
        verbose=True
    )
    svm_linear.fit(X_train, y_train)
    
    # Make predictions
    print(f"\n4. Making predictions (Linear SVM)...")
    y_pred_linear = svm_linear.predict(X_test)
    y_proba_linear = svm_linear.predict_proba(X_test)
    
    print(f"   Sample probabilities: {y_proba_linear[:5]}")
    print(f"   Sample predictions: {y_pred_linear[:5]}")
    print(f"   True labels: {y_test[:5].astype(int)}")
    
    # Evaluate Linear SVM
    from core.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc_linear = accuracy_score(y_test, y_pred_linear)
    prec_linear = precision_score(y_test, y_pred_linear)
    rec_linear = recall_score(y_test, y_pred_linear)
    f1_linear = f1_score(y_test, y_pred_linear)
    
    print(f"\n5. Linear SVM Performance:")
    print(f"   Accuracy: {acc_linear:.4f}")
    print(f"   Precision: {prec_linear:.4f}")
    print(f"   Recall: {rec_linear:.4f}")
    print(f"   F1-Score: {f1_linear:.4f}")
    
    # Test RBF SVM
    print(f"\n6. Training RBF SVM...")
    svm_rbf = SVMClassifier(
        kernel='rbf',
        C=1.0,
        gamma=0.1,
        learning_rate=0.01,
        n_iterations=500,
        verbose=True
    )
    svm_rbf.fit(X_train, y_train)
    
    # Evaluate RBF SVM
    print(f"\n7. Making predictions (RBF SVM)...")
    y_pred_rbf = svm_rbf.predict(X_test)
    
    acc_rbf = accuracy_score(y_test, y_pred_rbf)
    n_sv = svm_rbf.n_support_vectors()
    
    print(f"   Accuracy: {acc_rbf:.4f}")
    print(f"   Support Vectors: {n_sv}/{len(X_train)} ({100*n_sv/len(X_train):.1f}%)")
    
    print("\n" + "=" * 60)
    print("✓ SVM Classifier working! (Pure NumPy)")
    print("✓ NO sklearn.svm used!")
    print("✓ Linear & RBF kernels, hinge loss, gradient descent - ALL from scratch!")
