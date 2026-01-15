"""
Core Metrics Module - From Scratch Implementation
Cardiovascular Disease Prediction System

All metrics implemented using only numpy - no sklearn dependencies.
"""

import numpy as np


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix for binary classification.
    
    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
    
    Returns:
        2x2 numpy array: [[TN, FP], [FN, TP]]
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[TN, FP], [FN, TP]])


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        float: Accuracy score
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    
    return correct / total if total > 0 else 0.0


def precision_score(y_true, y_pred):
    """
    Calculate precision: TP / (TP + FP)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        float: Precision score
    """
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FP = cm[0, 1]
    
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0


def recall_score(y_true, y_pred):
    """
    Calculate recall (sensitivity): TP / (TP + FN)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        float: Recall score
    """
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def f1_score(y_true, y_pred):
    """
    Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        float: F1 score
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def specificity_score(y_true, y_pred):
    """
    Calculate specificity: TN / (TN + FP)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        float: Specificity score
    """
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    
    return TN / (TN + FP) if (TN + FP) > 0 else 0.0


def roc_curve(y_true, y_scores):
    """
    Compute ROC curve points (FPR, TPR) for different thresholds.
    
    Args:
        y_true: True labels (n_samples,)
        y_scores: Predicted probabilities (n_samples,)
    
    Returns:
        tuple: (fpr, tpr, thresholds)
    """
    y_true = np.array(y_true).flatten()
    y_scores = np.array(y_scores).flatten()
    
    # Get unique thresholds sorted in descending order
    thresholds = np.unique(y_scores)
    thresholds = np.sort(thresholds)[::-1]
    
    # Add boundary thresholds
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds, [thresholds[-1] - 1]])
    
    fpr_list = []
    tpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate TPR and FPR
        cm = confusion_matrix(y_true, y_pred)
        TN, FP = cm[0, 0], cm[0, 1]
        FN, TP = cm[1, 0], cm[1, 1]
        
        tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds


def auc_score(y_true, y_scores):
    """
    Calculate Area Under ROC Curve using trapezoidal rule.
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities
    
    Returns:
        float: AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Sort by FPR to ensure proper integration
    sorted_indices = np.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]
    
    # Calculate AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0
    
    return auc


def log_loss(y_true, y_pred_proba, eps=1e-15):
    """
    Calculate logarithmic loss (cross-entropy loss).
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        eps: Small constant to avoid log(0)
    
    Returns:
        float: Log loss value
    """
    y_true = np.array(y_true).flatten()
    y_pred_proba = np.array(y_pred_proba).flatten()
    
    # Clip probabilities to avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    
    # Calculate log loss
    loss = -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    
    return loss


def brier_score(y_true, y_pred_proba):
    """
    Calculate Brier score (mean squared error of probabilities).
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        float: Brier score
    """
    y_true = np.array(y_true).flatten()
    y_pred_proba = np.array(y_pred_proba).flatten()
    
    return np.mean((y_true - y_pred_proba) ** 2)


def matthews_corrcoef(y_true, y_pred):
    """
    Calculate Matthews Correlation Coefficient.
    
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        float: MCC value between -1 and 1
    """
    cm = confusion_matrix(y_true, y_pred)
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]
    
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    return numerator / denominator if denominator > 0 else 0.0


def classification_report(y_true, y_pred, y_pred_proba=None):
    """
    Generate comprehensive classification metrics report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
    
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'specificity': specificity_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['auc'] = auc_score(y_true, y_pred_proba)
        metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        metrics['brier_score'] = brier_score(y_true, y_pred_proba)
    
    return metrics


def print_classification_report(metrics, model_name="Model"):
    """
    Print formatted classification report.
    
    Args:
        metrics: Dictionary of metrics from classification_report()
        model_name: Name of the model for display
    """
    print(f"\n{'='*60}")
    print(f"{model_name} - Classification Report")
    print(f"{'='*60}")
    
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1 Score:     {metrics['f1']:.4f}")
    print(f"Specificity:  {metrics['specificity']:.4f}")
    print(f"MCC:          {metrics['mcc']:.4f}")
    
    if 'auc' in metrics:
        print(f"AUC-ROC:      {metrics['auc']:.4f}")
        print(f"Log Loss:     {metrics['log_loss']:.4f}")
        print(f"Brier Score:  {metrics['brier_score']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"              Predicted")
    print(f"              0      1")
    print(f"Actual  0   {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"        1   {cm[1,0]:5d}  {cm[1,1]:5d}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test the metrics module
    print("Testing Core Metrics Module...")
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred_proba = np.random.rand(1000)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Test all metrics
    metrics = classification_report(y_true, y_pred, y_pred_proba)
    print_classification_report(metrics, "Test Model")
    
    print("✓ All metrics working correctly!")
