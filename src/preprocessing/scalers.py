"""
Data Scaling Module - From Scratch Implementation
Cardiovascular Disease Prediction System

Scalers implemented using only numpy - no sklearn dependencies.
"""

import numpy as np


class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance.
    
    Formula: z = (x - μ) / σ
    where μ is the mean and σ is the standard deviation.
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_ = None
    
    def fit(self, X):
        """
        Compute the mean and std to be used for later scaling.
        
        Args:
            X: Training data (n_samples, n_features)
        
        Returns:
            self
        """
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.n_features_ = X.shape[1]
        
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        
        return self
    
    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        
        Args:
            X: Data to transform (n_samples, n_features)
        
        Returns:
            X_scaled: Standardized data
        """
        if self.mean_ is None:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        
        X = np.array(X)
        X_scaled = (X - self.mean_) / self.std_
        
        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Args:
            X: Training data
        
        Returns:
            X_scaled: Standardized data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Scale back the data to the original representation.
        
        Args:
            X_scaled: Scaled data
        
        Returns:
            X: Original scale data
        """
        if self.mean_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        X_scaled = np.array(X_scaled)
        X = X_scaled * self.std_ + self.mean_
        
        return X


class MinMaxScaler:
    """
    Transform features by scaling each feature to [0, 1] range.
    
    Formula: x_scaled = (x - x_min) / (x_max - x_min)
    """
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_features_ = None
    
    def fit(self, X):
        """
        Compute the min and max to be used for later scaling.
        
        Args:
            X: Training data
        
        Returns:
            self
        """
        X = np.array(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Avoid division by zero
        self.data_range_[self.data_range_ == 0] = 1.0
        
        self.n_features_ = X.shape[1]
        self.min_, self.max_ = self.feature_range
        
        return self
    
    def transform(self, X):
        """
        Scale features to the specified range.
        
        Args:
            X: Data to transform
        
        Returns:
            X_scaled: Scaled data
        """
        if self.data_min_ is None:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        
        X = np.array(X)
        
        # Scale to [0, 1]
        X_std = (X - self.data_min_) / self.data_range_
        
        # Scale to [min, max]
        X_scaled = X_std * (self.max_ - self.min_) + self.min_
        
        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Args:
            X: Training data
        
        Returns:
            X_scaled: Scaled data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Undo the scaling of X according to feature_range.
        
        Args:
            X_scaled: Scaled data
        
        Returns:
            X: Original scale data
        """
        if self.data_min_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        X_scaled = np.array(X_scaled)
        
        # Scale from [min, max] to [0, 1]
        X_std = (X_scaled - self.min_) / (self.max_ - self.min_)
        
        # Scale from [0, 1] to original range
        X = X_std * self.data_range_ + self.data_min_
        
        return X


class RobustScaler:
    """
    Scale features using statistics that are robust to outliers.
    
    Uses median and interquartile range (IQR) instead of mean and std.
    Formula: x_scaled = (x - median) / IQR
    """
    
    def __init__(self):
        self.center_ = None
        self.scale_ = None
        self.n_features_ = None
    
    def fit(self, X):
        """
        Compute the median and IQR to be used for later scaling.
        
        Args:
            X: Training data
        
        Returns:
            self
        """
        X = np.array(X)
        
        self.center_ = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        self.scale_ = q75 - q25
        
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        
        self.n_features_ = X.shape[1]
        
        return self
    
    def transform(self, X):
        """
        Center and scale the data.
        
        Args:
            X: Data to transform
        
        Returns:
            X_scaled: Scaled data
        """
        if self.center_ is None:
            raise ValueError("Scaler has not been fitted yet. Call fit() first.")
        
        X = np.array(X)
        X_scaled = (X - self.center_) / self.scale_
        
        return X_scaled
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Args:
            X: Training data
        
        Returns:
            X_scaled: Scaled data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Scale back the data to the original representation.
        
        Args:
            X_scaled: Scaled data
        
        Returns:
            X: Original scale data
        """
        if self.center_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        X_scaled = np.array(X_scaled)
        X = X_scaled * self.scale_ + self.center_
        
        return X


if __name__ == "__main__":
    # Test the scalers
    print("Testing Scalers Module...")
    
    np.random.seed(42)
    X_train = np.random.randn(100, 3) * 10 + 50
    X_test = np.random.randn(20, 3) * 10 + 50
    
    print("\n1. StandardScaler:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   Original mean: {X_train.mean(axis=0)}")
    print(f"   Original std:  {X_train.std(axis=0)}")
    print(f"   Scaled mean:   {X_train_scaled.mean(axis=0)}")
    print(f"   Scaled std:    {X_train_scaled.std(axis=0)}")
    
    print("\n2. MinMaxScaler:")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   Original min:  {X_train.min(axis=0)}")
    print(f"   Original max:  {X_train.max(axis=0)}")
    print(f"   Scaled min:    {X_train_scaled.min(axis=0)}")
    print(f"   Scaled max:    {X_train_scaled.max(axis=0)}")
    
    print("\n3. RobustScaler:")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   Original median: {np.median(X_train, axis=0)}")
    print(f"   Scaled median:   {np.median(X_train_scaled, axis=0)}")
    
    print("\n✓ All scalers working correctly!")
