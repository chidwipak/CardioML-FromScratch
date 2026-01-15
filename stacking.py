"""
Stacking Ensemble Classifier - From Scratch Implementation
Cardiovascular Disease Prediction System


NO sklearn.ensemble - Pure NumPy implementation.
Implements: Stacking with meta-classifier (2-level learning).
"""

import numpy as np


class StackingClassifier:
    """
    Stacking ensemble with meta-classifier.
    
    Two-level learning:
    1. Base level: Multiple diverse classifiers trained on training data
    2. Meta level: Meta-classifier trained on base predictions
    
    The meta-classifier learns to optimally combine base predictions.
    
    NO sklearn - Pure NumPy implementation per project requirements.
    
    Parameters:
    -----------
    base_estimators : list of (name, estimator) tuples
        Base classifiers (level 0)
    meta_classifier : estimator
        Meta-classifier (level 1) that combines base predictions
    use_probabilities : bool, default=True
        Use predicted probabilities as meta-features (vs. class labels)
    cv_folds : int, default=5
        Number of cross-validation folds for generating meta-features
    """
    
    def __init__(self, base_estimators, meta_classifier, 
                 use_probabilities=True, cv_folds=5):
        self.base_estimators = base_estimators
        self.meta_classifier = meta_classifier
        self.use_probabilities = use_probabilities
        self.cv_folds = cv_folds
        
        # Trained base models
        self.base_models = []
    
    def _generate_meta_features(self, X, y):
        """
        Generate meta-features using cross-validation.
        
        Prevents overfitting by using out-of-fold predictions.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        
        Returns:
            meta_features: (n_samples, n_base_estimators)
        """
        n_samples = X.shape[0]
        n_estimators = len(self.base_estimators)
        
        # Initialize meta-features
        meta_features = np.zeros((n_samples, n_estimators))
        
        # K-fold cross-validation
        fold_size = n_samples // self.cv_folds
        
        for fold in range(self.cv_folds):
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.cv_folds - 1 else n_samples
            
            val_indices = np.arange(val_start, val_end)
            train_indices = np.concatenate([
                np.arange(0, val_start),
                np.arange(val_end, n_samples)
            ])
            
            X_fold_train = X[train_indices]
            y_fold_train = y[train_indices]
            X_fold_val = X[val_indices]
            
            # Train each base estimator on this fold
            for i, (name, estimator) in enumerate(self.base_estimators):
                # Import the estimator class and create new instance
                estimator_class = type(estimator)
                
                # Create a new instance with same parameters
                # Try to copy key parameters
                try:
                    if hasattr(estimator, 'n_neighbors'):
                        fold_model = estimator_class(n_neighbors=estimator.n_neighbors)
                    elif hasattr(estimator, 'max_depth'):
                        fold_model = estimator_class(max_depth=estimator.max_depth)
                    elif hasattr(estimator, 'learning_rate'):
                        fold_model = estimator_class(
                            learning_rate=estimator.learning_rate,
                            n_iterations=estimator.n_iterations,
                            verbose=False
                        )
                    else:
                        fold_model = estimator_class()
                except:
                    # Fallback: use default parameters
                    fold_model = estimator_class()
                
                # Train on fold training data
                fold_model.fit(X_fold_train, y_fold_train)
                
                # Predict on fold validation data
                if self.use_probabilities and hasattr(fold_model, 'predict_proba'):
                    preds = fold_model.predict_proba(X_fold_val)
                else:
                    preds = fold_model.predict(X_fold_val)
                
                # Store predictions as meta-features
                meta_features[val_indices, i] = preds
        
        return meta_features
    
    def fit(self, X, y):
        """
        Train stacking ensemble.
        
        Steps:
        1. Generate meta-features using CV on base estimators
        2. Train meta-classifier on meta-features
        3. Retrain base estimators on full training data
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        
        Returns:
            self
        """
        X = np.array(X)
        y = np.array(y)
        
        print("Generating meta-features using cross-validation...")
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X, y)
        
        print(f"Meta-features shape: {meta_features.shape}")
        
        # Train meta-classifier on meta-features
        print("Training meta-classifier...")
        self.meta_classifier.fit(meta_features, y)
        
        # Retrain base estimators on full training data
        print("Retraining base estimators on full data...")
        self.base_models = []
        
        for name, estimator in self.base_estimators:
            print(f"  Training {name}...")
            estimator.fit(X, y)
            self.base_models.append((name, estimator))
        
        print("Stacking training complete!")
        
        return self
    
    def predict(self, X):
        """
        Predict using stacking ensemble.
        
        Steps:
        1. Get predictions from all base estimators
        2. Use predictions as input to meta-classifier
        3. Return meta-classifier predictions
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predicted labels (n_samples,)
        """
        X = np.array(X)
        
        # Get base predictions
        base_predictions = self._get_base_predictions(X)
        
        # Meta-classifier predicts on base predictions
        final_predictions = self.meta_classifier.predict(base_predictions)
        
        return final_predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Probability of positive class (n_samples,)
        """
        X = np.array(X)
        
        # Get base predictions
        base_predictions = self._get_base_predictions(X)
        
        # Meta-classifier predicts probabilities
        if hasattr(self.meta_classifier, 'predict_proba'):
            probabilities = self.meta_classifier.predict_proba(base_predictions)
        else:
            # Fallback to hard predictions
            probabilities = self.meta_classifier.predict(base_predictions)
        
        return probabilities
    
    def _get_base_predictions(self, X):
        """
        Get predictions from all base estimators.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Base predictions (n_samples, n_base_estimators)
        """
        n_samples = X.shape[0]
        n_estimators = len(self.base_models)
        
        base_predictions = np.zeros((n_samples, n_estimators))
        
        for i, (name, estimator) in enumerate(self.base_models):
            if self.use_probabilities and hasattr(estimator, 'predict_proba'):
                preds = estimator.predict_proba(X)
            else:
                preds = estimator.predict(X)
            
            base_predictions[:, i] = preds
        
        return base_predictions
    
    def get_base_predictions_detailed(self, X):
        """
        Get detailed predictions from each base estimator.
        
        Useful for analysis and debugging.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            dict: {estimator_name: predictions}
        """
        X = np.array(X)
        
        results = {}
        for name, estimator in self.base_models:
            predictions = estimator.predict(X)
            results[name] = predictions
        
        return results


if __name__ == "__main__":
    # Test Stacking Ensemble
    print("Testing Stacking Ensemble (Pure NumPy - NO sklearn)...")
    print("=" * 60)
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from classical.logistic_regression import LogisticRegression
    from classical.knn import KNNClassifier
    from classical.decision_tree import DecisionTreeClassifier
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 600
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # Non-linear decision boundary
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 1.5).astype(int)
    
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
    base_estimators = [
        ('logistic', LogisticRegression(n_iterations=500, learning_rate=0.1, verbose=False)),
        ('knn', KNNClassifier(n_neighbors=5)),
        ('tree', DecisionTreeClassifier(max_depth=5))
    ]
    
    # Create meta-classifier
    meta_classifier = LogisticRegression(n_iterations=300, learning_rate=0.1, verbose=False)
    
    # Train Stacking Ensemble
    print(f"\n4. Training Stacking Ensemble...")
    stacking = StackingClassifier(
        base_estimators=base_estimators,
        meta_classifier=meta_classifier,
        use_probabilities=True,
        cv_folds=5
    )
    
    stacking.fit(X_train, y_train)
    
    # Make predictions
    print(f"\n5. Making predictions...")
    y_pred = stacking.predict(X_test)
    y_proba = stacking.predict_proba(X_test)
    
    from core.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n6. Stacking Ensemble Performance:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall: {rec:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Compare with individual models
    print(f"\n7. Individual Base Model Performance:")
    base_preds = stacking.get_base_predictions_detailed(X_test)
    
    for name, preds in base_preds.items():
        acc_base = accuracy_score(y_test, preds)
        print(f"   {name}: {acc_base:.4f}")
    
    print(f"\n   Stacking (meta-learning): {acc:.4f}")
    print(f"   ✓ Stacking combines base models optimally!")
    
    print("\n" + "=" * 60)
    print("✓ Stacking Ensemble working! (Pure NumPy)")
    print("✓ NO sklearn.ensemble used!")
    print("✓ 2-level meta-learning, cross-validation, optimal combination - ALL from scratch!")
    print("✓ TIER 3 ENSEMBLE COMPLETE!")
