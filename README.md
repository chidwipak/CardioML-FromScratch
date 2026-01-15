# 🫀 CardioML-FromScratch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Only-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Models](https://img.shields.io/badge/Models-8-red.svg)

**A complete machine learning pipeline for cardiovascular disease prediction built entirely from scratch using only NumPy and Pandas.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-models) • [Results](#-results) • [Architecture](#-architecture)

</div>

---

> [!NOTE]
> **📋 About This Repository**  
> This project was developed in **November 2025** as part of my **Cyber Security academic coursework** at IIIT Sri City. The development was done on a remote college SSH system, which is why it's being pushed to GitHub now (January 2026) as I prepare my portfolio for internship applications.
> 
> All my academic and research projects were stored on the college's remote server infrastructure. Now, as I'm applying for internships, I'm systematically organizing and publishing my work to showcase my skills and experience.

---

## 📖 Overview

This project implements a comprehensive cardiovascular disease prediction system using **8 different machine learning algorithms**, all built from scratch without using any ML libraries like scikit-learn, TensorFlow, or PyTorch.

The system analyzes patient health data including blood pressure, cholesterol levels, BMI, and lifestyle factors to predict the presence of cardiovascular disease with high accuracy.

### 🎯 Key Highlights

- **100% From-Scratch Implementation** - Every algorithm coded using only NumPy
- **8 ML Classifiers** - From simple to advanced ensemble methods
- **70,000+ Patient Records** - Trained on real cardiovascular health data
- **Comprehensive Pipeline** - Data preprocessing, training, evaluation, and visualization
- **Modular Architecture** - Clean, reusable, and well-documented code

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Deep Learning** | Multi-Layer Perceptron with backpropagation |
| 🌲 **Tree-Based** | Decision Tree (CART) and Random Forest |
| 📊 **Classical ML** | Logistic Regression, KNN, SVM |
| 🎯 **Ensemble Methods** | Voting and Stacking classifiers |
| 📈 **Visualization** | Confusion matrices, ROC curves, feature importance |
| ⚡ **Optimized** | Efficient NumPy vectorization for 70K samples |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/chidwipak/CardioML-FromScratch.git
cd CardioML-FromScratch

# Install dependencies
pip install numpy pandas matplotlib seaborn

# Run the complete pipeline
python main.py
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | ≥1.21.0 | Numerical computations |
| Pandas | ≥1.3.0 | Data manipulation |
| Matplotlib | ≥3.4.0 | Visualization |
| Seaborn | ≥0.11.0 | Statistical plots |

---

## 💻 Usage

### Quick Start

```bash
# Run full training pipeline
python main.py

# The pipeline will:
# 1. Load and preprocess the dataset
# 2. Train all 8 classifiers
# 3. Evaluate and compare performance
# 4. Generate visualizations
```

### Individual Classifiers

```python
# Example: Using the MLP classifier
from mlp import MLPClassifier

# Initialize and train
mlp = MLPClassifier(
    hidden_layers=[128, 64, 32],
    learning_rate=0.001,
    epochs=100
)
mlp.fit(X_train, y_train)

# Predict
predictions = mlp.predict(X_test)
probabilities = mlp.predict_proba(X_test)
```

```python
# Example: Using Random Forest
from random_forest import RandomForestClassifier

rf = RandomForestClassifier(n_trees=100, max_depth=10)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
```

### Ensemble Methods

```python
# Voting Classifier
from voting import VotingClassifier
from logistic_regression import LogisticRegression
from decision_tree import DecisionTreeClassifier
from knn import KNNClassifier

voting = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('knn', KNNClassifier())
    ],
    voting='soft'
)
voting.fit(X_train, y_train)
```

---

## 🤖 Models

### Model Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TIER 1: CLASSICAL ML                      │
├─────────────────────────────────────────────────────────────┤
│  Logistic Regression    │  KNN            │  SVM             │
│  • Gradient Descent     │  • Euclidean    │  • RBF Kernel    │
│  • L2 Regularization    │  • Distance     │  • Hinge Loss    │
│  • Probabilistic        │  • Weighting    │  • Gradient      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    TIER 2: TREE-BASED                        │
├─────────────────────────────────────────────────────────────┤
│  Decision Tree (CART)           │  Random Forest             │
│  • Gini Impurity               │  • Bootstrap Aggregating    │
│  • Recursive Splitting         │  • Random Feature Selection │
│  • Pruning                     │  • Ensemble of Trees        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  TIER 3: DEEP LEARNING                       │
├─────────────────────────────────────────────────────────────┤
│  Multi-Layer Perceptron (MLP)                                │
│  • 3 Hidden Layers (128 → 64 → 32)                          │
│  • ReLU + Sigmoid Activations                                │
│  • Adam Optimizer with Learning Rate Scheduling              │
│  • Batch Normalization + Dropout                             │
│  • Binary Cross-Entropy Loss                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 TIER 4: ENSEMBLE METHODS                     │
├─────────────────────────────────────────────────────────────┤
│  Voting Classifier           │  Stacking Classifier          │
│  • Hard Voting (Majority)    │  • 2-Level Learning          │
│  • Soft Voting (Probability) │  • Meta-Classifier           │
│  • Weighted Voting           │  • Cross-Validation Blend     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Model Specifications

| Model | Implementation Details | Key Parameters |
|-------|----------------------|----------------|
| **MLP** | Fully connected neural network with backpropagation | `hidden_layers=[128,64,32]`, `lr=0.001` |
| **Logistic Regression** | Gradient descent with L2 regularization | `lr=0.01`, `reg=0.01`, `iters=1000` |
| **KNN** | Euclidean distance with weighted voting | `k=5`, `weights='distance'` |
| **Decision Tree** | CART with Gini impurity | `max_depth=10`, `min_samples=2` |
| **Random Forest** | Bootstrap aggregating with random features | `n_trees=100`, `max_features='sqrt'` |
| **SVM** | RBF kernel with gradient-based optimization | `C=1.0`, `gamma='scale'` |
| **Voting** | Soft voting ensemble | `voting='soft'` |
| **Stacking** | Meta-learning with logistic regression | `cv=5` |

---

## 📊 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **73.48%** | **75.52%** | **69.44%** | **72.35%** | **0.792** |
| Hard Voting | 72.54% | 74.51% | 68.47% | 71.36% | 0.785 |
| Logistic Regression | 71.70% | 71.51% | 72.09% | 71.80% | 0.780 |
| Stacking | 70.78% | 72.23% | 67.48% | 69.77% | 0.784 |
| Soft Voting | 70.78% | 73.64% | 64.66% | 68.86% | 0.785 |
| Decision Tree | 70.28% | 72.16% | 65.99% | 68.94% | 0.703 |
| KNN | 67.55% | 67.99% | 66.24% | 67.10% | 0.732 |
| MLP | 66.14% | 65.30% | 68.78% | 66.99% | 0.719 |
| SVM | 58.34% | 58.02% | 60.18% | 59.08% | 0.596 |

### Key Insights

- 🏆 **Random Forest achieves the best overall performance** with 73.48% accuracy
- 📈 Ensemble methods (Voting, Stacking) provide competitive results
- 🎯 All from-scratch implementations achieve reasonable accuracy
- ⚖️ Good balance between precision and recall for medical application

---

## 🏗️ Architecture

### Project Structure

```
CardioML-FromScratch/
├── 📄 main.py                    # Complete training pipeline
├── 📄 mlp.py                     # Multi-Layer Perceptron
├── 📄 logistic_regression.py     # Logistic Regression
├── 📄 knn.py                     # K-Nearest Neighbors
├── 📄 decision_tree.py           # Decision Tree (CART)
├── 📄 random_forest.py           # Random Forest
├── 📄 svm.py                     # Support Vector Machine
├── 📄 voting.py                  # Voting Ensemble
├── 📄 stacking.py                # Stacking Ensemble
├── 📊 cardio_train_cleaned.csv   # Dataset (70,000 samples)
├── 📄 PROJECT_DOCUMENTATION.md   # Detailed implementation guide
├── 📁 src/                       # Supporting modules
│   ├── 📁 models/
│   │   └── 📁 deep_learning/
│   │       ├── layers.py         # Dense, Dropout, BatchNorm
│   │       ├── activations.py    # ReLU, Sigmoid, Softmax
│   │       └── losses.py         # BCE, Categorical CE
│   ├── 📁 preprocessing/
│   │   ├── scalers.py            # StandardScaler, MinMaxScaler
│   │   ├── feature_engineering.py
│   │   └── outlier_detection.py  # IQR, Z-score methods
│   ├── 📁 utils/
│   │   ├── data_loader.py        # CSV loading utilities
│   │   └── train_test_split.py   # Stratified splitting
│   ├── 📁 core/
│   │   └── metrics.py            # Accuracy, Precision, Recall, F1, AUC
│   └── 📁 visualization/
│       └── plots.py              # Confusion matrix, ROC curves
└── 📁 results/                   # Output visualizations
```

### Data Flow

```
┌──────────────────┐
│  Raw CSV Data    │
│ (70,000 samples) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Preprocessing   │
│ • Missing values │
│ • Outlier removal│
│ • Feature eng.   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Train/Test      │
│  Split (80/20)   │
│  Stratified      │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ Train │ │ Test  │
│ Data  │ │ Data  │
└───┬───┘ └───┬───┘
    │         │
    ▼         │
┌──────────┐  │
│ 8 Models │  │
│ Training │  │
└────┬─────┘  │
     │        │
     ▼        ▼
┌─────────────────┐
│   Evaluation    │
│ • Metrics       │
│ • Visualization │
└─────────────────┘
```

---

## 📊 Dataset

### Cardiovascular Disease Dataset

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in days | Numerical |
| `gender` | Gender (1: Female, 2: Male) | Categorical |
| `height` | Height in cm | Numerical |
| `weight` | Weight in kg | Numerical |
| `ap_hi` | Systolic blood pressure | Numerical |
| `ap_lo` | Diastolic blood pressure | Numerical |
| `cholesterol` | Cholesterol level (1: Normal, 2: Above, 3: High) | Categorical |
| `gluc` | Glucose level (1: Normal, 2: Above, 3: High) | Categorical |
| `smoke` | Smoking status | Binary |
| `alco` | Alcohol intake | Binary |
| `active` | Physical activity | Binary |
| `bmi` | Body Mass Index (derived) | Numerical |
| **`cardio`** | **Target: Cardiovascular disease (0/1)** | **Binary** |

### Dataset Statistics

- **Total Samples**: 70,000
- **Features**: 12 input features + 1 target
- **Class Distribution**: ~50% positive, ~50% negative (balanced)
- **Source**: Kaggle Cardiovascular Disease Dataset

---

## 🔬 Technical Implementation

### No External ML Libraries

This project deliberately avoids using scikit-learn, TensorFlow, PyTorch, or any other ML framework. All algorithms are implemented from mathematical foundations using only:

- **NumPy** - Matrix operations, linear algebra
- **Pandas** - Data loading and manipulation
- **Matplotlib/Seaborn** - Visualization only

### Mathematical Foundations

#### Logistic Regression
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$

#### Neural Network Forward Pass
$$a^{[l]} = g(W^{[l]} \cdot a^{[l-1]} + b^{[l]})$$

#### Gini Impurity (Decision Tree)
$$Gini(D) = 1 - \sum_{i=1}^{c} p_i^2$$

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- IIIT Sri City for academic support and infrastructure
- Kaggle for the cardiovascular disease dataset
- NumPy community for the excellent numerical computing library

---

## 👥 Team

**IIIT Sri City - Cyber Security Academic Project (November 2025)**

| Name | Roll Number | Role |
|------|-------------|------|
| **Chidwipak Kuppani** | S20230010131 | Lead Developer |
| Mohan Ganesh | S20230010092 | Team Member |
| Sandeep | S20230010178 | Team Member |
| Praneeth | S20230010007 | Team Member |

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ and NumPy

</div>
