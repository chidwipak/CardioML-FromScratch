"""
Data Loading Module - From Scratch Implementation
Cardiovascular Disease Prediction System

NO sklearn - Pure Pandas/NumPy for loading and preparing data.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_cvd_dataset(filepath, apply_feature_engineering=True):
    """
    Load the Cardiovascular Disease dataset from CSV.
    
    Args:
        filepath: Path to CSV file (string or Path object)
        apply_feature_engineering: Whether to apply feature engineering
    
    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array)
        feature_names: List of feature names
    """
    # Load CSV using pandas
    df = pd.read_csv(filepath)
    
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Apply feature engineering if requested
    if apply_feature_engineering:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from preprocessing.feature_engineering import engineer_all_features
        df = engineer_all_features(df)
        print(f"After feature engineering: {df.shape[1]} features")
    
    # Separate features and target
    if 'cardio' in df.columns:
        y = df['cardio'].values
        X = df.drop('cardio', axis=1)
    else:
        raise ValueError("Target column 'cardio' not found in dataset")
    
    # Remove ID column if present
    if 'id' in X.columns:
        X = X.drop('id', axis=1)
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Convert to numpy array
    X = X.values
    
    # Check for missing values
    n_missing = np.sum(np.isnan(X))
    if n_missing > 0:
        print(f"Warning: {n_missing} missing values detected")
    
    return X, y, feature_names


def handle_missing_values(X, strategy='mean'):
    """
    Handle missing values in feature matrix.
    
    NO sklearn.impute.SimpleImputer - Pure NumPy implementation.
    
    Args:
        X: Feature matrix with potential NaN values
        strategy: 'mean', 'median', or 'mode'
    
    Returns:
        X_imputed: Feature matrix with missing values filled
    """
    X = np.array(X, dtype=float)
    
    # Check if there are any missing values
    if not np.any(np.isnan(X)):
        return X
    
    X_imputed = X.copy()
    
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        missing_mask = np.isnan(col)
        
        if np.any(missing_mask):
            if strategy == 'mean':
                fill_value = np.nanmean(col)
            elif strategy == 'median':
                fill_value = np.nanmedian(col)
            elif strategy == 'mode':
                # Get most frequent value
                values, counts = np.unique(col[~missing_mask], return_counts=True)
                fill_value = values[np.argmax(counts)]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            X_imputed[missing_mask, col_idx] = fill_value
    
    return X_imputed


def validate_data(X, y, feature_names=None):
    """
    Validate data integrity and print summary statistics.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names (optional)
    
    Returns:
        bool: True if data is valid
    """
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    
    # Check dimensions
    print(f"\n1. Dimensions:")
    print(f"   Features (X): {X.shape}")
    print(f"   Target (y): {y.shape}")
    
    if len(X) != len(y):
        print("   ❌ ERROR: X and y have different number of samples!")
        return False
    print("   ✓ Dimensions match")
    
    # Check for NaN/Inf
    print(f"\n2. Data Quality:")
    n_nan_X = np.sum(np.isnan(X))
    n_inf_X = np.sum(np.isinf(X))
    n_nan_y = np.sum(np.isnan(y))
    
    print(f"   NaN in X: {n_nan_X}")
    print(f"   Inf in X: {n_inf_X}")
    print(f"   NaN in y: {n_nan_y}")
    
    if n_nan_X > 0 or n_inf_X > 0 or n_nan_y > 0:
        print("   ⚠️  WARNING: Data contains NaN or Inf values!")
    else:
        print("   ✓ No NaN or Inf values")
    
    # Check target distribution
    print(f"\n3. Target Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for val, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        print(f"   Class {int(val)}: {count:,} samples ({percentage:.2f}%)")
    
    # Check class balance
    balance_ratio = max(counts) / min(counts)
    if balance_ratio > 1.5:
        print(f"   ⚠️  WARNING: Imbalanced classes (ratio: {balance_ratio:.2f})")
    else:
        print(f"   ✓ Balanced classes (ratio: {balance_ratio:.2f})")
    
    # Feature statistics
    print(f"\n4. Feature Statistics:")
    print(f"   Number of features: {X.shape[1]}")
    print(f"   Feature ranges:")
    for i in range(min(5, X.shape[1])):  # Show first 5 features
        feature_name = feature_names[i] if feature_names else f"Feature {i}"
        print(f"      {feature_name}: [{np.min(X[:, i]):.2f}, {np.max(X[:, i]):.2f}]")
    if X.shape[1] > 5:
        print(f"      ... and {X.shape[1] - 5} more features")
    
    print("\n" + "=" * 60)
    print("✓ Validation complete!")
    print("=" * 60 + "\n")
    
    return True


def save_processed_data(X, y, feature_names, output_dir='data/processed'):
    """
    Save processed data to files.
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy arrays
    np.save(output_path / 'X.npy', X)
    np.save(output_path / 'y.npy', y)
    
    # Save feature names
    with open(output_path / 'feature_names.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    print(f"✓ Saved processed data to {output_dir}/")


def load_processed_data(input_dir='data/processed'):
    """
    Load previously processed data.
    
    Args:
        input_dir: Directory containing processed files
    
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
    """
    input_path = Path(input_dir)
    
    # Load numpy arrays
    X = np.load(input_path / 'X.npy')
    y = np.load(input_path / 'y.npy')
    
    # Load feature names
    with open(input_path / 'feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"✓ Loaded processed data from {input_dir}/")
    print(f"   Shape: {X.shape}, Target: {y.shape}")
    
    return X, y, feature_names


if __name__ == "__main__":
    # Test data loading
    print("Testing Data Loader Module (Pure Pandas/NumPy)...")
    print("=" * 60)
    
    # Test with actual CVD dataset if available
    try:
        print("\n1. Loading CVD dataset...")
        X, y, feature_names = load_cvd_dataset('cardio_train_cleaned.csv')
        
        print(f"\n2. Dataset loaded successfully!")
        print(f"   Samples: {X.shape[0]:,}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Feature names: {feature_names[:5]}... (showing first 5)")
        
        print(f"\n3. Validating data...")
        validate_data(X, y, feature_names)
        
    except FileNotFoundError:
        print("   CSV file not found. Creating synthetic data for testing...")
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        y = np.random.randint(0, 2, 1000)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        print(f"\n2. Synthetic data created:")
        print(f"   Samples: {X.shape[0]:,}")
        print(f"   Features: {X.shape[1]}")
    
    # Test missing value handling
    print(f"\n4. Testing missing value handling...")
    X_with_missing = X.copy()
    X_with_missing[0:5, 0] = np.nan
    X_with_missing[10:15, 1] = np.nan
    
    print(f"   Added {np.sum(np.isnan(X_with_missing))} missing values")
    X_imputed = handle_missing_values(X_with_missing, strategy='mean')
    print(f"   After imputation: {np.sum(np.isnan(X_imputed))} missing values")
    print(f"   ✓ Missing values handled!")
    
    print("\n" + "=" * 60)
    print("✓ All data loading functions working! (Pure Pandas/NumPy)")
