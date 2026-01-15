"""
Random Forest Classifier
Cardiovascular Disease Prediction System
Implementation: Complete Random Forest with bootstrap aggregating and random feature selection
"""

import numpy as np
from collections import Counter

class Node:
    """Decision tree node for Random Forest"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier:
    """Decision tree classifier for Random Forest ensemble"""
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
        
    def fit(self, X, y):
        """Build decision tree"""
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the decision tree"""
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        # Check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split using random feature subset
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)
        
        # Create child splits
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)
    
    def _best_split(self, X, y, feat_idxs):
        """Find the best split using Gini impurity"""
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
                    
        return split_idx, split_threshold
    
    def _information_gain(self, y, X_column, threshold):
        """Calculate information gain using Gini impurity"""
        parent_gini = self._gini_impurity(y)
        
        left_idxs, right_idxs = self._split(X_column, threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        g_l, g_r = self._gini_impurity(y[left_idxs]), self._gini_impurity(y[right_idxs])
        child_gini = (n_l / n) * g_l + (n_r / n) * g_r
        
        return parent_gini - child_gini
    
    def _split(self, X_column, split_thresh):
        """Split data based on threshold"""
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        gini = 1 - sum(probabilities**2)
        return gini
    
    def _most_common_label(self, y):
        """Return most common label"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """Predict class labels"""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction"""
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForestClassifier:
    """Random Forest classifier with bootstrap aggregating"""
    def __init__(self, n_trees=150, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
    def fit(self, X, y):
        """Build forest of trees"""
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def _bootstrap_samples(self, X, y):
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def predict(self, X):
        """Predict using majority voting"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        probabilities = np.array([np.mean(pred) for pred in tree_preds])
        return probabilities
    
    def _most_common_label(self, y):
        """Return most common label"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_trees=100, max_depth=10)
    rf.fit(X_train, y_train)
    
    # Make predictions
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Random Forest Accuracy: {accuracy:.4f}")
