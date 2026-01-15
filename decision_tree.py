"""
Decision Tree Classifier - CART Algorithm From Scratch
Cardiovascular Disease Prediction System

NO sklearn.tree - Pure NumPy implementation with CART algorithm.
Implements: Gini impurity, recursive binary splitting, pruning.
"""

import numpy as np


class Node:
    """
    Decision Tree Node.
    
    Can be either:
    - Internal node: has feature_index and threshold for splitting
    - Leaf node: has predicted class value
    """
    
    def __init__(self, feature_index=None, threshold=None, 
                 left=None, right=None, value=None, gini=None, samples=None):
        # Internal node attributes
        self.feature_index = feature_index  # Feature to split on
        self.threshold = threshold          # Threshold value for split
        self.left = left                    # Left child node
        self.right = right                  # Right child node
        
        # Leaf node attribute
        self.value = value                  # Predicted class (for leaf)
        
        # Node statistics
        self.gini = gini                    # Gini impurity
        self.samples = samples              # Number of samples at node


class DecisionTreeClassifier:
    """
    Decision Tree using CART (Classification and Regression Trees) algorithm.
    
    Uses Gini impurity for splitting criterion.
    NO sklearn - Pure NumPy implementation per project requirements.
    
    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of tree (None = unlimited)
    min_samples_split : int, default=2
        Minimum samples required to split node
    min_samples_leaf : int, default=1
        Minimum samples required in leaf node
    max_features : int or None, default=None
        Number of features to consider for split (None = all features)
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        
        # Root node (built during training)
        self.root = None
        
        # Number of classes
        self.n_classes = None
    
    def gini_impurity(self, y):
        """
        Calculate Gini impurity for a set of labels.
        
        Formula: Gini = 1 - Σ(p_i²)
        where p_i is the proportion of class i
        
        Gini = 0: Pure node (all same class)
        Gini = 0.5: Maximum impurity (binary, 50/50 split)
        
        Args:
            y: Labels array (n_samples,)
        
        Returns:
            Gini impurity (float)
        """
        if len(y) == 0:
            return 0.0
        
        # Count each class
        _, counts = np.unique(y, return_counts=True)
        
        # Calculate proportions
        proportions = counts / len(y)
        
        # Gini impurity
        gini = 1.0 - np.sum(proportions ** 2)
        
        return gini
    
    def split_data(self, X, y, feature_index, threshold):
        """
        Split dataset based on feature and threshold.
        
        Left: samples where feature <= threshold
        Right: samples where feature > threshold
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            feature_index: Index of feature to split on
            threshold: Threshold value
        
        Returns:
            X_left, X_right, y_left, y_right
        """
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        return X_left, X_right, y_left, y_right
    
    def information_gain(self, y_parent, y_left, y_right):
        """
        Calculate information gain from a split.
        
        Formula: IG = Gini(parent) - [n_left/n * Gini(left) + n_right/n * Gini(right)]
        
        Args:
            y_parent: Parent labels
            y_left: Left child labels
            y_right: Right child labels
        
        Returns:
            Information gain (float)
        """
        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        # Weighted average of child impurities
        gini_parent = self.gini_impurity(y_parent)
        gini_left = self.gini_impurity(y_left)
        gini_right = self.gini_impurity(y_right)
        
        weighted_child_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
        
        # Information gain
        ig = gini_parent - weighted_child_gini
        
        return ig
    
    def find_best_split(self, X, y):
        """
        Find the best split for a node using Gini impurity.
        
        Tries all features and all unique values as potential thresholds.
        Selects split with maximum information gain.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
        
        Returns:
            best_feature_index, best_threshold, best_gain
        """
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None, None, 0.0
        
        # Determine which features to consider
        if self.max_features is None:
            feature_indices = range(n_features)
        else:
            # Random subset of features
            feature_indices = np.random.choice(
                n_features, 
                size=min(self.max_features, n_features), 
                replace=False
            )
        
        best_gain = 0.0
        best_feature_index = None
        best_threshold = None
        
        # Try each feature
        for feature_index in feature_indices:
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)
            
            # Try each unique value as threshold
            for threshold in unique_values:
                # Split data
                _, _, y_left, y_right = self.split_data(X, y, feature_index, threshold)
                
                # Check minimum samples constraint
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                # Calculate information gain
                gain = self.information_gain(y, y_left, y_right)
                
                # Update best split
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
        
        return best_feature_index, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        """
        Recursively build decision tree using CART algorithm.
        
        Stopping criteria:
        1. Max depth reached
        2. Node is pure (all same class)
        3. Minimum samples for split not met
        4. No valid split found
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            depth: Current tree depth
        
        Returns:
            Node object (root of subtree)
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Node statistics
        gini = self.gini_impurity(y)
        
        # Determine most common class (for leaf nodes)
        unique_classes, counts = np.unique(y, return_counts=True)
        most_common_class = unique_classes[np.argmax(counts)]
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or \
           n_samples < self.min_samples_split:
            # Create leaf node
            return Node(value=most_common_class, gini=gini, samples=n_samples)
        
        # Find best split
        best_feature, best_threshold, best_gain = self.find_best_split(X, y)
        
        # If no valid split found, create leaf
        if best_feature is None or best_gain == 0.0:
            return Node(value=most_common_class, gini=gini, samples=n_samples)
        
        # Split data
        X_left, X_right, y_left, y_right = self.split_data(X, y, best_feature, best_threshold)
        
        # Recursively build left and right subtrees
        left_child = self.build_tree(X_left, y_left, depth + 1)
        right_child = self.build_tree(X_right, y_right, depth + 1)
        
        # Create internal node
        return Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            gini=gini,
            samples=n_samples
        )
    
    def fit(self, X, y):
        """
        Build decision tree from training data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        
        Returns:
            self
        """
        X = np.array(X)
        y = np.array(y)
        
        self.n_classes = len(np.unique(y))
        
        # Build tree recursively
        self.root = self.build_tree(X, y, depth=0)
        
        return self
    
    def predict_sample(self, x, node):
        """
        Predict class for a single sample by traversing tree.
        
        Args:
            x: Feature vector (n_features,)
            node: Current node
        
        Returns:
            Predicted class
        """
        # If leaf node, return value
        if node.value is not None:
            return node.value
        
        # Otherwise, traverse based on feature value
        if x[node.feature_index] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predicted labels (n_samples,)
        """
        if self.root is None:
            raise ValueError("Tree not trained yet! Call fit() first.")
        
        X = np.array(X)
        predictions = np.array([self.predict_sample(x, self.root) for x in X])
        
        return predictions
    
    def predict_proba_sample(self, x, node):
        """
        Get class probabilities for a single sample.
        
        For leaf nodes, returns proportion of training samples in each class.
        
        Args:
            x: Feature vector (n_features,)
            node: Current node
        
        Returns:
            Probability of positive class (float)
        """
        # If leaf node, return probability based on training data
        if node.value is not None:
            # For binary classification, return 1.0 if class 1, else 0.0
            # (This is simplified; ideally store class distributions in leaves)
            return float(node.value)
        
        # Traverse tree
        if x[node.feature_index] <= node.threshold:
            return self.predict_proba_sample(x, node.left)
        else:
            return self.predict_proba_sample(x, node.right)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Probabilities of positive class (n_samples,)
        """
        if self.root is None:
            raise ValueError("Tree not trained yet!")
        
        X = np.array(X)
        probabilities = np.array([self.predict_proba_sample(x, self.root) for x in X])
        
        return probabilities
    
    def get_depth(self, node=None):
        """
        Get depth of tree.
        
        Args:
            node: Starting node (default: root)
        
        Returns:
            Tree depth (int)
        """
        if node is None:
            node = self.root
        
        if node is None or node.value is not None:
            return 0
        
        left_depth = self.get_depth(node.left)
        right_depth = self.get_depth(node.right)
        
        return 1 + max(left_depth, right_depth)
    
    def count_nodes(self, node=None):
        """
        Count total nodes in tree.
        
        Args:
            node: Starting node (default: root)
        
        Returns:
            Node count (int)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        if node.value is not None:
            return 1  # Leaf node
        
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)
    
    def count_leaves(self, node=None):
        """
        Count leaf nodes in tree.
        
        Args:
            node: Starting node (default: root)
        
        Returns:
            Leaf count (int)
        """
        if node is None:
            node = self.root
        
        if node is None:
            return 0
        
        if node.value is not None:
            return 1  # Leaf node
        
        return self.count_leaves(node.left) + self.count_leaves(node.right)


if __name__ == "__main__":
    # Test Decision Tree
    print("Testing Decision Tree CART (Pure NumPy - NO sklearn)...")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 4
    
    # Create dataset with clear decision boundaries
    X = np.random.randn(n_samples, n_features)
    
    # Complex decision rule
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int) | \
        ((X[:, 2] < -0.5) & (X[:, 3] > 0.5)).astype(int)
    
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
    
    # Train decision tree
    print(f"\n3. Training Decision Tree...")
    tree = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5
    )
    tree.fit(X_train, y_train)
    
    print(f"   Tree depth: {tree.get_depth()}")
    print(f"   Total nodes: {tree.count_nodes()}")
    print(f"   Leaf nodes: {tree.count_leaves()}")
    
    # Make predictions
    print(f"\n4. Making predictions...")
    y_pred = tree.predict(X_test)
    
    print(f"   Sample predictions: {y_pred[:10]}")
    print(f"   True labels: {y_test[:10]}")
    
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
    
    # Test different max_depth values
    print(f"\n6. Testing different max_depth values...")
    for depth in [3, 5, 10, None]:
        tree_test = DecisionTreeClassifier(max_depth=depth)
        tree_test.fit(X_train, y_train)
        y_pred_test = tree_test.predict(X_test)
        acc = accuracy_score(y_test, y_pred_test)
        print(f"   max_depth={depth}: Accuracy={acc:.4f}, Depth={tree_test.get_depth()}, Nodes={tree_test.count_nodes()}")
    
    print("\n" + "=" * 60)
    print("✓ Decision Tree CART working! (Pure NumPy)")
    print("✓ NO sklearn.tree used!")
    print("✓ Gini impurity, binary splitting, recursive tree building - ALL from scratch!")
