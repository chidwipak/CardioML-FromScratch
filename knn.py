"""
K-Nearest Neighbors Classifier - From Scratch Implementation
Cardiovascular Disease Prediction System

NO sklearn.neighbors - Pure NumPy implementation with distance metrics.
Implements: Euclidean distance, k-nearest search, majority voting.
"""

import numpy as np
from collections import Counter


class KNNClassifier:
    """
    K-Nearest Neighbors classifier for binary/multiclass classification.
    
    Uses Euclidean distance and majority voting.
    NO sklearn - Pure NumPy implementation per project requirements.
    
    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use
    weights : str, default='uniform'
        Weight function: 'uniform' (all equal) or 'distance' (inverse distance)
    metric : str, default='euclidean'
        Distance metric to use
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        
        # Training data (stored for predictions)
        self.X_train = None
        self.y_train = None
    
    def euclidean_distance(self, x1, x2):
        """
        Calculate Euclidean distance between two vectors.
        
        Formula: d = sqrt(Σ(x1_i - x2_i)²)
        
        Args:
            x1: First vector (n_features,)
            x2: Second vector (n_features,)
        
        Returns:
            Euclidean distance (float)
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def manhattan_distance(self, x1, x2):
        """
        Calculate Manhattan distance between two vectors.
        
        Formula: d = Σ|x1_i - x2_i|
        
        Args:
            x1: First vector (n_features,)
            x2: Second vector (n_features,)
        
        Returns:
            Manhattan distance (float)
        """
        return np.sum(np.abs(x1 - x2))
    
    def compute_distance(self, x1, x2):
        """
        Compute distance based on selected metric.
        
        Args:
            x1: First vector
            x2: Second vector
        
        Returns:
            Distance value
        """
        if self.metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def fit(self, X, y):
        """
        Store training data for KNN.
        
        KNN is a lazy learner - no training phase, just stores data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        
        Returns:
            self
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        For each sample:
        1. Calculate distances to all training samples
        2. Find k nearest neighbors
        3. Apply majority voting
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predicted labels (n_samples,)
        """
        if self.X_train is None:
            raise ValueError("Model not trained yet! Call fit() first.")
        
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Calculate distances to all training samples
            distances = np.array([
                self.compute_distance(x, x_train) 
                for x_train in self.X_train
            ])
            
            # Find k nearest neighbors
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_train[k_indices]
            
            # Apply weighting
            if self.weights == 'uniform':
                # Simple majority voting
                most_common = Counter(k_nearest_labels).most_common(1)
                prediction = most_common[0][0]
            
            elif self.weights == 'distance':
                # Distance-weighted voting
                k_distances = distances[k_indices]
                
                # Avoid division by zero
                k_distances = np.where(k_distances == 0, 1e-10, k_distances)
                
                # Inverse distance weights
                weights = 1 / k_distances
                
                # Weighted vote for each class
                unique_labels = np.unique(k_nearest_labels)
                weighted_votes = {}
                
                for label in unique_labels:
                    mask = (k_nearest_labels == label)
                    weighted_votes[label] = np.sum(weights[mask])
                
                # Select class with highest weighted vote
                prediction = max(weighted_votes, key=weighted_votes.get)
            
            else:
                raise ValueError(f"Unknown weight type: {self.weights}")
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Probability = (count of neighbors in class) / k
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Probability of positive class (n_samples,)
        """
        if self.X_train is None:
            raise ValueError("Model not trained yet! Call fit() first.")
        
        X = np.array(X)
        probabilities = []
        
        for x in X:
            # Calculate distances to all training samples
            distances = np.array([
                self.compute_distance(x, x_train) 
                for x_train in self.X_train
            ])
            
            # Find k nearest neighbors
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_train[k_indices]
            
            if self.weights == 'uniform':
                # Simple probability = count / k
                probability = np.sum(k_nearest_labels == 1) / self.n_neighbors
            
            elif self.weights == 'distance':
                # Distance-weighted probability
                k_distances = distances[k_indices]
                k_distances = np.where(k_distances == 0, 1e-10, k_distances)
                weights = 1 / k_distances
                
                # Weighted count for positive class
                positive_mask = (k_nearest_labels == 1)
                positive_weight = np.sum(weights[positive_mask])
                total_weight = np.sum(weights)
                
                probability = positive_weight / total_weight
            
            probabilities.append(probability)
        
        return np.array(probabilities)
    
    def get_neighbors(self, x):
        """
        Get k nearest neighbors for a single sample.
        
        Args:
            x: Sample features (n_features,)
        
        Returns:
            List of (index, distance, label) tuples
        """
        if self.X_train is None:
            raise ValueError("Model not trained yet!")
        
        # Calculate distances
        distances = np.array([
            self.compute_distance(x, x_train) 
            for x_train in self.X_train
        ])
        
        # Find k nearest
        k_indices = np.argsort(distances)[:self.n_neighbors]
        
        neighbors = [
            (idx, distances[idx], self.y_train[idx]) 
            for idx in k_indices
        ]
        
        return neighbors


if __name__ == "__main__":
    # Test KNN Classifier
    print("Testing K-Nearest Neighbors (Pure NumPy - NO sklearn)...")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 4
    
    # Create two clusters
    cluster1 = np.random.randn(n_samples // 2, n_features) + np.array([2, 2, 2, 2])
    cluster2 = np.random.randn(n_samples // 2, n_features) + np.array([-2, -2, -2, -2])
    
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
    
    print(f"\n2. Data split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Test different k values
    print(f"\n3. Testing different k values...")
    
    for k in [3, 5, 7]:
        model = KNNClassifier(n_neighbors=k, weights='uniform')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        from core.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   k={k}: Accuracy = {accuracy:.4f}")
    
    # Train final model
    print(f"\n4. Training KNN with k=5...")
    model = KNNClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
    model.fit(X_train, y_train)
    
    # Make predictions
    print(f"\n5. Making predictions...")
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
    
    print(f"\n6. Test Set Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Test distance weighting
    print(f"\n7. Testing distance-weighted KNN...")
    model_weighted = KNNClassifier(n_neighbors=5, weights='distance')
    model_weighted.fit(X_train, y_train)
    y_pred_weighted = model_weighted.predict(X_test)
    accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
    print(f"   Distance-weighted accuracy: {accuracy_weighted:.4f}")
    
    # Show nearest neighbors for first test sample
    print(f"\n8. Nearest neighbors for first test sample:")
    neighbors = model.get_neighbors(X_test[0])
    for idx, dist, label in neighbors:
        print(f"   Neighbor {idx}: distance={dist:.4f}, label={int(label)}")
    
    print("\n" + "=" * 60)
    print("✓ K-Nearest Neighbors working! (Pure NumPy)")
    print("✓ NO sklearn.neighbors used!")
    print("✓ Euclidean distance, majority voting, distance weighting - ALL from scratch!")
