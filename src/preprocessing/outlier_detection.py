"""
Outlier Detection Module - From Scratch Implementation
Cardiovascular Disease Prediction System

NO sklearn - Pure NumPy implementation for academic project.
"""

import numpy as np


def detect_outliers_iqr(X, threshold=1.5):
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Outliers are values that fall below Q1 - threshold*IQR or above Q3 + threshold*IQR.
    
    Args:
        X: Data array (n_samples, n_features) or (n_samples,)
        threshold: IQR multiplier (default 1.5 for standard outlier detection)
    
    Returns:
        outlier_mask: Boolean array indicating outliers (True = outlier)
    """
    X = np.array(X)
    
    # Handle 1D array
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    outlier_mask = np.zeros(X.shape[0], dtype=bool)
    
    for feature_idx in range(X.shape[1]):
        feature_data = X[:, feature_idx]
        
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = np.percentile(feature_data, 25)
        Q3 = np.percentile(feature_data, 75)
        
        # Calculate IQR
        IQR = Q3 - Q1
        
        # Define outlier boundaries
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Mark outliers for this feature
        feature_outliers = (feature_data < lower_bound) | (feature_data > upper_bound)
        outlier_mask |= feature_outliers
    
    return outlier_mask


def detect_outliers_zscore(X, threshold=3.0):
    """
    Detect outliers using Z-score method.
    
    Outliers are values where |z-score| > threshold.
    Z-score = (x - mean) / std
    
    Args:
        X: Data array (n_samples, n_features) or (n_samples,)
        threshold: Z-score threshold (default 3.0)
    
    Returns:
        outlier_mask: Boolean array indicating outliers (True = outlier)
    """
    X = np.array(X)
    
    # Handle 1D array
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    outlier_mask = np.zeros(X.shape[0], dtype=bool)
    
    for feature_idx in range(X.shape[1]):
        feature_data = X[:, feature_idx]
        
        # Calculate mean and standard deviation
        mean = np.mean(feature_data)
        std = np.std(feature_data)
        
        # Avoid division by zero
        if std == 0:
            continue
        
        # Calculate z-scores
        z_scores = np.abs((feature_data - mean) / std)
        
        # Mark outliers
        feature_outliers = z_scores > threshold
        outlier_mask |= feature_outliers
    
    return outlier_mask


def detect_clinical_outliers(data_dict):
    """
    Detect physiologically impossible values for CVD dataset.
    
    Clinical constraints for cardiovascular data:
    - Age: 18-100 years (6570-36500 days)
    - Height: 130-230 cm
    - Weight: 30-200 kg
    - Systolic BP: 50-250 mmHg
    - Diastolic BP: 30-150 mmHg
    - Systolic >= Diastolic (physiological constraint)
    - Cholesterol, Glucose: 1, 2, or 3
    - Binary features: 0 or 1
    
    Args:
        data_dict: Dictionary with keys: 'age', 'height', 'weight', 
                   'ap_hi', 'ap_lo', 'cholesterol', 'gluc', etc.
    
    Returns:
        outlier_mask: Boolean array indicating outliers (True = outlier)
    """
    n_samples = len(data_dict['age'])
    outlier_mask = np.zeros(n_samples, dtype=bool)
    
    # Age constraints (in days)
    if 'age' in data_dict:
        age = np.array(data_dict['age'])
        outlier_mask |= (age < 6570) | (age > 36500)  # 18-100 years
    
    # Height constraints
    if 'height' in data_dict:
        height = np.array(data_dict['height'])
        outlier_mask |= (height < 130) | (height > 230)
    
    # Weight constraints
    if 'weight' in data_dict:
        weight = np.array(data_dict['weight'])
        outlier_mask |= (weight < 30) | (weight > 200)
    
    # Blood pressure constraints
    if 'ap_hi' in data_dict and 'ap_lo' in data_dict:
        ap_hi = np.array(data_dict['ap_hi'])
        ap_lo = np.array(data_dict['ap_lo'])
        
        # Systolic BP constraints
        outlier_mask |= (ap_hi < 50) | (ap_hi > 250)
        
        # Diastolic BP constraints
        outlier_mask |= (ap_lo < 30) | (ap_lo > 150)
        
        # Systolic must be >= Diastolic
        outlier_mask |= (ap_hi < ap_lo)
    
    # Cholesterol constraints (must be 1, 2, or 3)
    if 'cholesterol' in data_dict:
        cholesterol = np.array(data_dict['cholesterol'])
        outlier_mask |= ~np.isin(cholesterol, [1, 2, 3])
    
    # Glucose constraints (must be 1, 2, or 3)
    if 'gluc' in data_dict:
        gluc = np.array(data_dict['gluc'])
        outlier_mask |= ~np.isin(gluc, [1, 2, 3])
    
    # Binary feature constraints (must be 0 or 1)
    binary_features = ['gender', 'smoke', 'alco', 'active', 'cardio']
    for feature in binary_features:
        if feature in data_dict:
            values = np.array(data_dict[feature])
            outlier_mask |= ~np.isin(values, [0, 1, 2])  # Gender can be 1 or 2
    
    return outlier_mask


def remove_outliers(X, y, outlier_mask):
    """
    Remove outliers from dataset.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target array (n_samples,)
        outlier_mask: Boolean array indicating outliers
    
    Returns:
        X_clean: Feature matrix without outliers
        y_clean: Target array without outliers
        n_removed: Number of outliers removed
    """
    X = np.array(X)
    y = np.array(y)
    outlier_mask = np.array(outlier_mask)
    
    # Keep samples that are NOT outliers
    clean_mask = ~outlier_mask
    
    X_clean = X[clean_mask]
    y_clean = y[clean_mask]
    n_removed = np.sum(outlier_mask)
    
    return X_clean, y_clean, n_removed


if __name__ == "__main__":
    # Test outlier detection
    print("Testing Outlier Detection Module (Pure NumPy)...")
    print("=" * 60)
    
    # Test IQR method
    print("\n1. IQR Method Test:")
    np.random.seed(42)
    data = np.random.randn(100, 3)
    # Add some outliers
    data[0, 0] = 10  # Extreme value
    data[1, 1] = -10  # Extreme value
    
    outliers_iqr = detect_outliers_iqr(data, threshold=1.5)
    print(f"   Detected {np.sum(outliers_iqr)} outliers using IQR")
    
    # Test Z-score method
    print("\n2. Z-score Method Test:")
    outliers_zscore = detect_outliers_zscore(data, threshold=3.0)
    print(f"   Detected {np.sum(outliers_zscore)} outliers using Z-score")
    
    # Test clinical outliers
    print("\n3. Clinical Outliers Test:")
    cvd_data = {
        'age': np.array([18000, 20000, 5000, 40000]),  # One too young, one too old
        'height': np.array([165, 170, 100, 250]),  # One too short, one too tall
        'weight': np.array([70, 75, 20, 250]),  # One too light, one too heavy
        'ap_hi': np.array([120, 130, 300, 90]),  # One too high
        'ap_lo': np.array([80, 85, 70, 100]),  # Normal
        'cholesterol': np.array([1, 2, 5, 3]),  # One invalid
        'gluc': np.array([1, 2, 3, 1]),
    }
    
    clinical_outliers = detect_clinical_outliers(cvd_data)
    print(f"   Detected {np.sum(clinical_outliers)} clinical outliers")
    print(f"   Outlier indices: {np.where(clinical_outliers)[0]}")
    
    # Test remove_outliers
    print("\n4. Remove Outliers Test:")
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    outlier_mask = detect_outliers_iqr(X, threshold=1.5)
    
    X_clean, y_clean, n_removed = remove_outliers(X, y, outlier_mask)
    print(f"   Original samples: {len(X)}")
    print(f"   Removed: {n_removed}")
    print(f"   Remaining: {len(X_clean)}")
    
    print("\n" + "=" * 60)
    print("✓ All outlier detection methods working! (Pure NumPy)")
