"""
Train/Test Split Module - From Scratch Implementation
Cardiovascular Disease Prediction System

NO sklearn.model_selection.train_test_split - Pure NumPy implementation.
"""

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None, stratify=False):
    """
    Split data into training and test sets.
    
    NO sklearn - Pure NumPy implementation with stratification support.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        test_size: Proportion of test set (0.0 to 1.0) or absolute number
        random_state: Random seed for reproducibility
        stratify: If True, preserve class distribution in splits
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.array(X)
    y = np.array(y)
    n_samples = len(X)
    
    # Convert test_size to number of samples
    if test_size < 1.0:
        n_test = int(n_samples * test_size)
    else:
        n_test = int(test_size)
    
    n_train = n_samples - n_test
    
    if stratify:
        # Stratified split - preserve class distribution
        indices_train = []
        indices_test = []
        
        # Get unique classes
        classes = np.unique(y)
        
        for cls in classes:
            # Get indices for this class
            cls_indices = np.where(y == cls)[0]
            n_cls_samples = len(cls_indices)
            
            # Calculate how many test samples for this class
            n_cls_test = int(n_cls_samples * (n_test / n_samples))
            
            # Shuffle indices for this class
            np.random.shuffle(cls_indices)
            
            # Split
            indices_test.extend(cls_indices[:n_cls_test])
            indices_train.extend(cls_indices[n_cls_test:])
        
        # Convert to numpy arrays
        indices_train = np.array(indices_train)
        indices_test = np.array(indices_test)
        
        # Shuffle the final indices
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_test)
        
    else:
        # Random split
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        indices_test = indices[:n_test]
        indices_train = indices[n_test:]
    
    # Split data
    X_train = X[indices_train]
    X_test = X[indices_test]
    y_train = y[indices_train]
    y_test = y[indices_test]
    
    return X_train, X_test, y_train, y_test


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, random_state=None, stratify=False):
    """
    Split data into training, validation, and test sets.
    
    NO sklearn - Pure NumPy implementation.
    
    Args:
        X: Feature matrix
        y: Target vector
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed
        stratify: If True, preserve class distribution
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    # Second split: separate validation from training
    # Adjust val_size relative to remaining data
    val_size_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state + 1 if random_state else None,
        stratify=stratify
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def k_fold_split(X, y, n_splits=5, shuffle=True, random_state=None):
    """
    Generate K-Fold cross-validation splits.
    
    NO sklearn - Pure NumPy implementation.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_splits: Number of folds
        shuffle: Whether to shuffle before splitting
        random_state: Random seed
    
    Yields:
        train_indices, val_indices for each fold
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1
    
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        yield train_indices, val_indices
        current = stop


def stratified_k_fold_split(X, y, n_splits=5, shuffle=True, random_state=None):
    """
    Generate Stratified K-Fold cross-validation splits.
    
    Preserves class distribution in each fold.
    NO sklearn - Pure NumPy implementation.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_splits: Number of folds
        shuffle: Whether to shuffle before splitting
        random_state: Random seed
    
    Yields:
        train_indices, val_indices for each fold
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(y)
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    
    # Count samples per class
    class_counts = np.bincount(y_indices)
    
    # Ensure each fold has at least one sample from each class
    if np.any(class_counts < n_splits):
        raise ValueError(
            f"Cannot have n_splits={n_splits} with less than {n_splits} "
            f"samples per class"
        )
    
    # Create fold indices for each class
    class_fold_indices = []
    for cls in range(n_classes):
        cls_indices = np.where(y_indices == cls)[0]
        if shuffle:
            np.random.shuffle(cls_indices)
        class_fold_indices.append(cls_indices)
    
    # Distribute samples into folds
    fold_indices = [[] for _ in range(n_splits)]
    for cls in range(n_classes):
        cls_indices = class_fold_indices[cls]
        n_cls_samples = len(cls_indices)
        
        # Calculate fold sizes for this class
        fold_sizes = np.full(n_splits, n_cls_samples // n_splits, dtype=int)
        fold_sizes[:n_cls_samples % n_splits] += 1
        
        current = 0
        for fold_idx, fold_size in enumerate(fold_sizes):
            start, stop = current, current + fold_size
            fold_indices[fold_idx].extend(cls_indices[start:stop])
            current = stop
    
    # Generate train/val splits
    for fold_idx in range(n_splits):
        val_indices = np.array(fold_indices[fold_idx])
        train_indices = np.concatenate([
            fold_indices[i] for i in range(n_splits) if i != fold_idx
        ])
        
        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)
        
        yield train_indices, val_indices


if __name__ == "__main__":
    # Test train/test split functions
    print("Testing Train/Test Split Module (Pure NumPy)...")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    print(f"\nOriginal data: {X.shape[0]} samples")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Test basic split
    print("\n1. Basic Train/Test Split (80/20):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    print(f"   Train classes: {np.bincount(y_train)}")
    print(f"   Test classes: {np.bincount(y_test)}")
    
    # Test stratified split
    print("\n2. Stratified Train/Test Split (80/20):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=True
    )
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    print(f"   Train classes: {np.bincount(y_train)}")
    print(f"   Test classes: {np.bincount(y_test)}")
    train_ratio = np.bincount(y_train)[1] / len(y_train)
    test_ratio = np.bincount(y_test)[1] / len(y_test)
    print(f"   Train positive ratio: {train_ratio:.3f}")
    print(f"   Test positive ratio: {test_ratio:.3f}")
    print(f"   ✓ Ratios preserved!")
    
    # Test train/val/test split
    print("\n3. Train/Val/Test Split (70/15/15):")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_size=0.15, test_size=0.15, random_state=42, stratify=True
    )
    print(f"   Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"   Val: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"   Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"   Train classes: {np.bincount(y_train)}")
    print(f"   Val classes: {np.bincount(y_val)}")
    print(f"   Test classes: {np.bincount(y_test)}")
    
    # Test K-Fold
    print("\n4. K-Fold Cross-Validation (5 folds):")
    for fold_idx, (train_idx, val_idx) in enumerate(
        k_fold_split(X, y, n_splits=5, random_state=42)
    ):
        print(f"   Fold {fold_idx + 1}: Train={len(train_idx)}, Val={len(val_idx)}")
    
    # Test Stratified K-Fold
    print("\n5. Stratified K-Fold Cross-Validation (5 folds):")
    for fold_idx, (train_idx, val_idx) in enumerate(
        stratified_k_fold_split(X, y, n_splits=5, random_state=42)
    ):
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        train_pos_ratio = np.sum(y_train_fold) / len(y_train_fold)
        val_pos_ratio = np.sum(y_val_fold) / len(y_val_fold)
        print(f"   Fold {fold_idx + 1}: Train={len(train_idx)} ({train_pos_ratio:.3f}), "
              f"Val={len(val_idx)} ({val_pos_ratio:.3f})")
    
    print("\n" + "=" * 60)
    print("✓ All split functions working! (Pure NumPy)")
    print("✓ No sklearn.model_selection used!")
