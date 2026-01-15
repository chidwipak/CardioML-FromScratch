"""
Voting Ensemble Classifiers - From Scratch Implementation
Cardiovascular Disease Prediction System


NO sklearn.ensemble - Pure NumPy implementation.
Implements: Hard Voting (majority), Soft Voting (probability averaging).
"""

import numpy as np


class VotingClassifier:
    """
    Voting ensemble that combines multiple classifiers.
    
    Two voting strategies:
    - Hard Voting: Majority vote on predicted class labels
    - Soft Voting: Average predicted probabilities
    
    NO sklearn - Pure NumPy implementation per project requirements.
    
    Parameters:
    -----------
    estimators : list of (name, estimator) tuples
        Base classifiers to ensemble
    voting : str, default='hard'
        Voting strategy ('hard' or 'soft')
    weights : array-like, default=None
        Weights for each estimator (None = equal weights)
    """
    
    def __init__(self, estimators, voting='hard', weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        
        # Validate weights
        if self.weights is not None:
            if len(self.weights) != len(self.estimators):
                raise ValueError("Number of weights must match number of estimators")
            # Normalize weights
            self.weights = np.array(self.weights) / np.sum(self.weights)
        else:
            # Equal weights
            self.weights = np.ones(len(self.estimators)) / len(self.estimators)
    
    def fit(self, X, y):
        """
        Train all base estimators.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        
        Returns:
            self
        """
        X = np.array(X)
        y = np.array(y)
        
        # Train each estimator
        for name, estimator in self.estimators:
            print(f"Training {name}...")
            estimator.fit(X, y)
        
        return self
    
    def predict(self, X):
        """
        Predict using voting ensemble.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predicted labels (n_samples,)
        """
        X = np.array(X)
        
        if self.voting == 'hard':
            # Hard voting: majority vote
            predictions = []
            
            for name, estimator in self.estimators:
                pred = estimator.predict(X)
                predictions.append(pred)
            
            predictions = np.array(predictions)  # (n_estimators, n_samples)
            
            # Weighted majority vote
            n_samples = X.shape[0]
            final_predictions = []
            
            for i in range(n_samples):
                # Get predictions for this sample
                sample_preds = predictions[:, i]
                
                # Weighted vote
                unique_labels = np.unique(sample_preds)
                votes = {}
                
                for label in unique_labels:
                    mask = (sample_preds == label)
                    votes[label] = np.sum(self.weights[mask])
                
                # Select label with highest weighted vote
                final_pred = max(votes, key=votes.get)
                final_predictions.append(final_pred)
            
            return np.array(final_predictions)
        
        elif self.voting == 'soft':
            # Soft voting: average probabilities
            probabilities = []
            
            for name, estimator in self.estimators:
                if not hasattr(estimator, 'predict_proba'):
                    raise ValueError(f"Estimator {name} does not support predict_proba")
                
                proba = estimator.predict_proba(X)
                probabilities.append(proba)
            
            probabilities = np.array(probabilities)  # (n_estimators, n_samples)
            
            # Weighted average of probabilities
            weighted_proba = np.average(probabilities, axis=0, weights=self.weights)
            
            # Predict class with highest probability
            predictions = (weighted_proba >= 0.5).astype(int)
            
            return predictions
        
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting}")
    
    def predict_proba(self, X):
        """
        Predict class probabilities (soft voting only).
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Probability of positive class (n_samples,)
        """
        X = np.array(X)
        
        probabilities = []
        
        for name, estimator in self.estimators:
            if not hasattr(estimator, 'predict_proba'):
                raise ValueError(f"Estimator {name} does not support predict_proba")
            
            proba = estimator.predict_proba(X)
            probabilities.append(proba)
        
        probabilities = np.array(probabilities)  # (n_estimators, n_samples)
        
        # Weighted average
        weighted_proba = np.average(probabilities, axis=0, weights=self.weights)
        
        return weighted_proba
    
    def get_estimator_predictions(self, X):
        """
        Get predictions from each individual estimator.
        
        Useful for analysis and debugging.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            dict: {estimator_name: predictions}
        """
        X = np.array(X)
        
        results = {}
        for name, estimator in self.estimators:
            predictions = estimator.predict(X)
            results[name] = predictions
        
        return results


if __name__ == "__main__":
    # Test Voting Ensemble
    print("Testing Voting Ensemble (Pure NumPy - NO sklearn)...")
    print("=" * 60)
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from classical.logistic_regression import LogisticRegression
    from classical.knn import KNNClassifier
    from classical.decision_tree import DecisionTreeClassifier
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
    
    print(f"\n1. Generated synthetic data:")
    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Class distribution: {np.bincount(y)}")
    
    # Split data
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from utils.train_test_split import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n2. Data split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Create base estimators
    print(f"\n3. Creating base estimators...")
    estimators = [
        ('logistic', LogisticRegression(n_iterations=500, learning_rate=0.1, verbose=False)),
        ('knn', KNNClassifier(n_neighbors=5)),
        ('tree', DecisionTreeClassifier(max_depth=5))
    ]
    
    # Test Hard Voting
    print(f"\n4. Training Hard Voting Ensemble...")
    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    voting_hard.fit(X_train, y_train)
    
    print(f"\n5. Making predictions (Hard Voting)...")
    y_pred_hard = voting_hard.predict(X_test)
    
    from core.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc_hard = accuracy_score(y_test, y_pred_hard)
    prec_hard = precision_score(y_test, y_pred_hard)
    rec_hard = recall_score(y_test, y_pred_hard)
    f1_hard = f1_score(y_test, y_pred_hard)
    
    print(f"\n6. Hard Voting Performance:")
    print(f"   Accuracy: {acc_hard:.4f}")
    print(f"   Precision: {prec_hard:.4f}")
    print(f"   Recall: {rec_hard:.4f}")
    print(f"   F1-Score: {f1_hard:.4f}")
    
    # Test Soft Voting
    print(f"\n7. Training Soft Voting Ensemble...")
    voting_soft = VotingClassifier(estimators=estimators, voting='soft')
    voting_soft.fit(X_train, y_train)
    
    print(f"\n8. Making predictions (Soft Voting)...")
    y_pred_soft = voting_soft.predict(X_test)
    y_proba_soft = voting_soft.predict_proba(X_test)
    
    acc_soft = accuracy_score(y_test, y_pred_soft)
    
    print(f"   Sample probabilities: {y_proba_soft[:5]}")
    print(f"   Accuracy: {acc_soft:.4f}")
    
    # Compare individual models
    print(f"\n9. Individual Model Performance:")
    individual_preds = voting_hard.get_estimator_predictions(X_test)
    
    for name, preds in individual_preds.items():
        acc = accuracy_score(y_test, preds)
        print(f"   {name}: {acc:.4f}")
    
    print(f"   Hard Voting: {acc_hard:.4f}")
    print(f"   Soft Voting: {acc_soft:.4f}")
    
    # Test weighted voting
    print(f"\n10. Weighted Voting (weights=[0.5, 0.3, 0.2])...")
    voting_weighted = VotingClassifier(
        estimators=estimators, 
        voting='soft',
        weights=[0.5, 0.3, 0.2]
    )
    voting_weighted.fit(X_train, y_train)
    y_pred_weighted = voting_weighted.predict(X_test)
    acc_weighted = accuracy_score(y_test, y_pred_weighted)
    print(f"   Weighted Voting Accuracy: {acc_weighted:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Voting Ensemble working! (Pure NumPy)")
    print("✓ NO sklearn.ensemble used!")
    print("✓ Hard voting, soft voting, weighted voting - ALL from scratch!")
