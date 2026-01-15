# 📋 CardioML-FromScratch: End-to-End Project Documentation

> **Author**: Chidwipak Kuppani (S20230010131)  
> **Institution**: IIIT Sri City  
> **Course**: Cyber Security Academic Project  
> **Development Period**: November 2025  
> **Repository**: [CardioML-FromScratch](https://github.com/chidwipak/CardioML-FromScratch)

---

## Table of Contents

1. [Project Idea & Motivation](#1-project-idea--motivation)
2. [What We Have Done](#2-what-we-have-done)
3. [How We Have Done It](#3-how-we-have-done-it)
4. [Tech Stack & Justifications](#4-tech-stack--justifications)
5. [Phase-wise Development](#5-phase-wise-development)
6. [Algorithm Deep Dives](#6-algorithm-deep-dives)
7. [Results & Analysis](#7-results--analysis)
8. [Challenges & Solutions](#8-challenges--solutions)
9. [References](#9-references)
10. [Future Scope](#10-future-scope)

---

## 1. Project Idea & Motivation

### 1.1 The Problem

Cardiovascular diseases (CVDs) are the **leading cause of death globally**, claiming approximately 17.9 million lives each year according to WHO. Early detection and prediction of CVD risk can significantly reduce mortality rates through timely intervention.

### 1.2 Why Machine Learning?

Traditional diagnosis relies on:
- Medical expertise availability
- Expensive diagnostic tests
- Time-consuming evaluation processes

**Machine learning offers**:
- Rapid, automated risk assessment
- Pattern recognition from large datasets
- Objective, reproducible predictions
- Scalable healthcare solutions

### 1.3 Why From-Scratch Implementation?

Instead of using libraries like scikit-learn, we chose to implement every algorithm from mathematical foundations because:

1. **Deep Understanding**: Writing algorithms from scratch forces us to understand every mathematical concept
2. **Academic Value**: Demonstrates true knowledge of ML internals, not just API usage
3. **Customization**: Full control over every hyperparameter and optimization technique
4. **Portfolio Differentiation**: Shows advanced skills beyond typical ML projects
5. **Debugging Skills**: When things go wrong, we know exactly where to look

### 1.4 Project Objectives

| Objective | Description |
|-----------|-------------|
| **Primary** | Build a CVD prediction system achieving >70% accuracy |
| **Technical** | Implement 8 ML algorithms from scratch using only NumPy |
| **Educational** | Document mathematical foundations of each algorithm |
| **Practical** | Create a reusable, modular codebase |

---

## 2. What We Have Done

### 2.1 Complete Pipeline Implementation

We built an end-to-end machine learning pipeline consisting of:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                          │
│  • CSV loading with Pandas                                       │
│  • Data validation and cleaning                                  │
│  • Missing value handling                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING LAYER                            │
│  • Outlier detection (IQR, Z-score methods)                     │
│  • Feature engineering (BMI calculation, age conversion)        │
│  • Feature scaling (StandardScaler, MinMaxScaler)               │
│  • Train-Validation-Test splitting (stratified)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL LAYER (8 Classifiers)                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│  │  Logistic   │ │    KNN      │ │  Decision   │                │
│  │ Regression  │ │  Classifier │ │    Tree     │                │
│  └─────────────┘ └─────────────┘ └─────────────┘                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│  │   Random    │ │     SVM     │ │     MLP     │                │
│  │   Forest    │ │  Classifier │ │   Network   │                │
│  └─────────────┘ └─────────────┘ └─────────────┘                │
│  ┌─────────────┐ ┌─────────────┐                                │
│  │   Voting    │ │  Stacking   │                                │
│  │  Ensemble   │ │  Ensemble   │                                │
│  └─────────────┘ └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION LAYER                              │
│  • Accuracy, Precision, Recall, F1-Score                        │
│  • ROC-AUC curves                                                │
│  • Confusion matrices                                            │
│  • Model comparison visualizations                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementation Statistics

| Category | Count | Details |
|----------|-------|---------|
| **Python Files** | 20+ | Classifiers, utilities, modules |
| **Lines of Code** | ~6,000+ | Pure NumPy implementations |
| **ML Algorithms** | 8 | From basic to advanced |
| **Support Modules** | 10 | Preprocessing, metrics, visualization |
| **Dataset Samples** | 70,000 | Real patient records |

### 2.3 Key Deliverables

1. **8 From-Scratch ML Classifiers**
   - Logistic Regression
   - K-Nearest Neighbors
   - Decision Tree (CART)
   - Random Forest
   - Support Vector Machine
   - Multi-Layer Perceptron
   - Voting Ensemble
   - Stacking Ensemble

2. **Supporting Infrastructure**
   - Data preprocessing pipeline
   - Feature engineering utilities
   - Evaluation metrics (all from scratch)
   - Visualization generation

3. **Documentation**
   - Comprehensive README
   - This detailed documentation
   - Inline code documentation

---

## 3. How We Have Done It

### 3.1 Development Approach

We followed a **bottom-up development approach**:

```
Week 1: Data preprocessing and utility functions
    ↓
Week 2: Classical ML models (Logistic Regression, KNN)
    ↓
Week 3: Tree-based models (Decision Tree, Random Forest)
    ↓
Week 4: Advanced models (SVM, MLP Neural Network)
    ↓
Week 5: Ensemble methods (Voting, Stacking)
    ↓
Week 6: Evaluation, visualization, and documentation
```

### 3.2 Code Architecture Principles

**1. Modular Design**
```
src/
├── preprocessing/     # Data cleaning and transformation
├── models/           # Neural network components
├── core/             # Evaluation metrics
├── utils/            # Helper functions
└── visualization/    # Plotting utilities
```

**2. Consistent API Design**

Every classifier follows the scikit-learn inspired interface:
```python
class Classifier:
    def __init__(self, **hyperparameters):
        """Initialize with hyperparameters"""
        
    def fit(self, X, y):
        """Train on data"""
        return self
        
    def predict(self, X):
        """Return class predictions"""
        
    def predict_proba(self, X):
        """Return probability estimates"""
```

**3. Pure NumPy Implementation**

All mathematical operations use NumPy vectorization:
- Matrix multiplications for efficiency
- Broadcasting for element-wise operations
- No loops where vectorization is possible

### 3.3 Testing Methodology

| Test Type | Description |
|-----------|-------------|
| **Unit Tests** | Individual function correctness |
| **Integration Tests** | End-to-end pipeline validation |
| **Comparison Tests** | Results compared with scikit-learn |
| **Performance Tests** | Timing and memory usage |

---

## 4. Tech Stack & Justifications

### 4.1 Core Technologies

| Technology | Version | Purpose | Why This Choice |
|------------|---------|---------|-----------------|
| **Python** | 3.8+ | Primary language | Industry standard for ML, excellent ecosystem |
| **NumPy** | ≥1.21.0 | Numerical computing | Foundation for matrix operations, vectorization |
| **Pandas** | ≥1.3.0 | Data manipulation | Efficient CSV handling, data cleaning |
| **Matplotlib** | ≥3.4.0 | Visualization | Publication-quality plots |
| **Seaborn** | ≥0.11.0 | Statistical visualization | Beautiful default styling |

### 4.2 Why NOT These Technologies

| Technology | Reason for Exclusion |
|------------|---------------------|
| **scikit-learn** | Would defeat the from-scratch purpose |
| **TensorFlow/PyTorch** | Too high-level, hides implementations |
| **Keras** | Same as TensorFlow |
| **XGBoost/LightGBM** | Pre-built implementations |

### 4.3 Development Environment

- **IDE**: VS Code with Python extensions
- **Remote System**: IIIT Sri City SSH server
- **Version Control**: Git (local)
- **Python Environment**: venv

---

## 5. Phase-wise Development

### Phase 1: Project Setup & Data Preprocessing (Week 1)

**Objective**: Establish project structure and data pipeline

**Completed Tasks**:
- [x] Project directory structure setup
- [x] Data loading utilities (`data_loader.py`)
- [x] Outlier detection methods (`outlier_detection.py`)
  - IQR-based detection
  - Z-score based detection
- [x] Feature scaling (`scalers.py`)
  - StandardScaler (z-score normalization)
  - MinMaxScaler (0-1 normalization)
- [x] Train-test-validation split (`train_test_split.py`)
  - Stratified sampling for balanced classes
- [x] Feature engineering (`feature_engineering.py`)
  - BMI calculation from height/weight
  - Age conversion from days to years

**Key Files Created**:
```
src/
├── preprocessing/
│   ├── scalers.py
│   ├── outlier_detection.py
│   └── feature_engineering.py
└── utils/
    ├── data_loader.py
    └── train_test_split.py
```

---

### Phase 2: Classical ML Models (Week 2)

**Objective**: Implement foundational classification algorithms

#### 2.1 Logistic Regression

**Mathematical Foundation**:

Sigmoid function:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Binary cross-entropy loss with L2 regularization:
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})] + \frac{\lambda}{2}||\theta||^2$$

Gradient descent update:
$$\theta := \theta - \alpha \nabla J(\theta)$$

**Implementation Details**:
- Gradient descent optimization
- L2 regularization to prevent overfitting
- Learning rate scheduling
- Convergence monitoring

#### 2.2 K-Nearest Neighbors

**Mathematical Foundation**:

Euclidean distance:
$$d(x, x') = \sqrt{\sum_{i=1}^{n}(x_i - x'_i)^2}$$

**Implementation Details**:
- Multiple distance metrics (Euclidean, Manhattan)
- Uniform and distance-weighted voting
- Efficient neighbor search

**Key Files Created**:
```
logistic_regression.py
knn.py
```

---

### Phase 3: Tree-Based Models (Week 3)

**Objective**: Implement decision tree and ensemble tree methods

#### 3.1 Decision Tree (CART Algorithm)

**Mathematical Foundation**:

Gini impurity:
$$Gini(D) = 1 - \sum_{i=1}^{c} p_i^2$$

Information gain:
$$IG = Gini_{parent} - \frac{n_{left}}{n}Gini_{left} - \frac{n_{right}}{n}Gini_{right}$$

**Implementation Details**:
- Recursive binary splitting
- Best split selection using Gini impurity
- Pruning via max_depth, min_samples_split
- Leaf node probability estimation

#### 3.2 Random Forest

**Mathematical Foundation**:

Bootstrap aggregating (Bagging):
- Random sampling with replacement
- Random feature subset selection
- Majority voting for final prediction

**Implementation Details**:
- Ensemble of decision trees
- Out-of-bag error estimation
- Feature importance calculation

**Key Files Created**:
```
decision_tree.py
random_forest.py
```

---

### Phase 4: Advanced Models (Week 4)

**Objective**: Implement SVM and neural network

#### 4.1 Support Vector Machine

**Mathematical Foundation**:

Linear SVM objective:
$$\min \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\max(0, 1 - y_i(w \cdot x_i + b))$$

RBF kernel:
$$K(x, x') = \exp(-\gamma||x - x'||^2)$$

**Implementation Details**:
- Gradient descent on hinge loss
- Linear and RBF kernel support
- Platt scaling for probability estimates

#### 4.2 Multi-Layer Perceptron

**Mathematical Foundation**:

Forward propagation:
$$a^{[l]} = g(W^{[l]} \cdot a^{[l-1]} + b^{[l]})$$

Backpropagation:
$$\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \odot g'(z^{[l]})$$

Weight update:
$$W^{[l]} := W^{[l]} - \alpha \frac{\partial J}{\partial W^{[l]}}$$

**Implementation Details**:
- Configurable hidden layers
- Multiple activation functions (ReLU, Sigmoid, Tanh)
- Mini-batch gradient descent
- Weight initialization (He, Xavier)

**Key Files Created**:
```
svm.py
mlp.py
src/models/deep_learning/
├── layers.py
├── activations.py
└── losses.py
```

---

### Phase 5: Ensemble Methods (Week 5)

**Objective**: Implement meta-learning techniques

#### 5.1 Voting Classifier

**Strategies Implemented**:
- **Hard Voting**: Majority class vote
- **Soft Voting**: Average predicted probabilities
- **Weighted Voting**: Weighted probability averaging

#### 5.2 Stacking Classifier

**Two-Level Learning**:
1. **Level 0**: Train base estimators
2. **Level 1**: Train meta-classifier on base predictions

**Cross-Validation Approach**:
- K-fold CV to generate meta-features
- Prevents overfitting on training data

**Key Files Created**:
```
voting.py
stacking.py
```

---

### Phase 6: Evaluation & Visualization (Week 6)

**Objective**: Comprehensive model evaluation and visualization

#### 6.1 Metrics Implementation

All metrics implemented from scratch in `src/core/metrics.py`:

| Metric | Formula |
|--------|---------|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ |
| **Precision** | $\frac{TP}{TP + FP}$ |
| **Recall** | $\frac{TP}{TP + FN}$ |
| **F1-Score** | $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$ |
| **AUC-ROC** | Area under ROC curve (trapezoidal rule) |

#### 6.2 Visualizations Generated

- Confusion matrices for each model
- ROC curves comparison
- Accuracy bar charts
- Metrics heatmap
- Model comparison summary

**Key Files Created**:
```
src/core/metrics.py
src/visualization/plots.py
main.py
```

---

## 6. Algorithm Deep Dives

### 6.1 Logistic Regression

```python
class LogisticRegression:
    """
    Binary classification using sigmoid activation.
    Optimization via gradient descent with L2 regularization.
    """
    
    def sigmoid(self, z):
        # Numerically stable sigmoid
        return np.where(z >= 0,
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def compute_loss(self, y_true, y_pred):
        # Binary cross-entropy with L2 regularization
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        bce = -np.mean(y_true * np.log(y_pred) + 
                       (1 - y_true) * np.log(1 - y_pred))
        l2_penalty = (self.regularization / 2) * np.sum(self.weights ** 2)
        return bce + l2_penalty
```

### 6.2 Decision Tree (CART)

```python
def find_best_split(self, X, y):
    """Find optimal split using Gini impurity."""
    best_gain = -1
    best_feature, best_threshold = None, None
    
    for feature_idx in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_idx])
        for threshold in thresholds:
            gain = self.information_gain(y, X[:, feature_idx], threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
                
    return best_feature, best_threshold, best_gain
```

### 6.3 Neural Network Backpropagation

```python
def backward(self, y_true, y_pred):
    """Backpropagation through all layers."""
    # Output layer gradient
    dA = self.loss_fn.backward(y_true, y_pred)
    
    # Propagate backwards through layers
    for layer in reversed(self.layers):
        dA = layer.backward(dA)
        
def update_weights(self):
    """Gradient descent update."""
    for layer in self.layers:
        if hasattr(layer, 'dW'):
            layer.weights -= self.learning_rate * layer.dW
            layer.biases -= self.learning_rate * layer.db
```

---

## 7. Results & Analysis

### 7.1 Model Performance Summary

| Rank | Model | Accuracy | F1-Score | AUC-ROC |
|------|-------|----------|----------|---------|
| 1 | Random Forest | 73.48% | 72.35% | 0.792 |
| 2 | Hard Voting | 72.54% | 71.36% | 0.785 |
| 3 | Logistic Regression | 71.70% | 71.80% | 0.780 |
| 4 | Stacking | 70.78% | 69.77% | 0.784 |
| 5 | Soft Voting | 70.78% | 68.86% | 0.785 |
| 6 | Decision Tree | 70.28% | 68.94% | 0.703 |
| 7 | KNN | 67.55% | 67.10% | 0.732 |
| 8 | MLP | 66.14% | 66.99% | 0.719 |
| 9 | SVM | 58.34% | 59.08% | 0.596 |

### 7.2 Key Observations

1. **Random Forest performs best** - Bootstrap aggregating reduces variance effectively
2. **Ensemble methods are competitive** - Voting achieves second-best accuracy
3. **Logistic Regression is surprisingly strong** - Simple but effective for this dataset
4. **SVM struggles** - Gradient-based approach may need more iterations or tuning
5. **MLP needs more tuning** - Deep learning typically requires more hyperparameter optimization

### 7.3 Medical Interpretation

For CVD prediction, we care about:
- **High Recall**: Don't miss positive cases (potential patients)
- **Reasonable Precision**: Don't cause unnecessary alarm

Random Forest achieves a good balance with 69.44% recall and 75.52% precision.

---

## 8. Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Numerical instability in sigmoid** | Implemented numerically stable version |
| **Slow KNN on 70K samples** | Optimized distance computation with vectorization |
| **SVM convergence issues** | Added learning rate decay and tolerance |
| **Overfitting in Decision Tree** | Implemented pruning via max_depth |
| **Vanishing gradients in MLP** | Used ReLU activation in hidden layers |

---

## 9. References

### 9.1 Dataset
- **Cardiovascular Disease Dataset**: [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

### 9.2 Textbooks & Courses
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- Andrew Ng's Machine Learning Course (Stanford/Coursera)

### 9.3 Documentation
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

### 9.4 Algorithm References
- Logistic Regression: [scikit-learn Mathematical Formulation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- Decision Trees: Breiman, L. (1984). *Classification and Regression Trees*
- Random Forest: Breiman, L. (2001). *Random Forests*
- SVM: Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks*

---

## 10. Future Scope

### 10.1 Potential Improvements

1. **Model Optimization**
   - Hyperparameter tuning with grid/random search
   - Cross-validation for more robust evaluation

2. **Additional Algorithms**
   - Naive Bayes classifier
   - Gradient Boosting (XGBoost-style)
   - AdaBoost ensemble

3. **Feature Engineering**
   - More derived features
   - Feature selection methods
   - PCA for dimensionality reduction

4. **Deployment**
   - Flask/FastAPI web application
   - Docker containerization
   - Cloud deployment (AWS/GCP)

### 10.2 Research Extensions

- Explainability with SHAP/LIME integration
- Fairness analysis across demographic groups
- Uncertainty quantification with Bayesian methods

---

## Appendix: Code Examples

### A.1 Running the Pipeline

```bash
# Clone the repository
git clone https://github.com/ChidwipakKuppani/CardioML-FromScratch.git
cd CardioML-FromScratch

# Install dependencies
pip install numpy pandas matplotlib seaborn

# Run complete pipeline
python main.py
```

### A.2 Using Individual Models

```python
# Import and use a classifier
from random_forest import RandomForestClassifier
from src.preprocessing.scalers import StandardScaler
from src.utils.data_loader import load_cvd_data

# Load and preprocess data
X, y = load_cvd_data('cardio_train_cleaned.csv')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
rf = RandomForestClassifier(n_trees=100, max_depth=10)
rf.fit(X_scaled, y)

# Predict
predictions = rf.predict(X_test)
probabilities = rf.predict_proba(X_test)
```

---

*This documentation was created as part of the CardioML-FromScratch project for academic purposes at IIIT Sri City.*

**Last Updated**: November 2025  
**Document Version**: 1.0
