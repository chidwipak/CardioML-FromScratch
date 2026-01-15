"""
Visualization Module - From Scratch Implementation
Cardiovascular Disease Prediction System

Uses matplotlib/seaborn ONLY - NO sklearn plotting utilities.
Implements: ROC curves, confusion matrices, feature importance, training curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix', 
                         save_path=None, figsize=(8, 6)):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (default: ['0', '1'])
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    from core.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Default class names
    if class_names is None:
        class_names = ['Negative', 'Positive']
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    return plt.gcf()


def plot_roc_curve(y_true, y_pred_proba, model_name='Model', 
                   save_path=None, figsize=(8, 6)):
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of model for legend
        save_path: Path to save figure (optional)
        figsize: Figure size
    
    Returns:
        AUC score
    """
    from core.metrics import roc_curve, auc_score
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = auc_score(y_true, y_pred_proba)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    
    return auc


def plot_multiple_roc_curves(y_true, predictions_dict, save_path=None, figsize=(10, 8)):
    """
    Plot multiple ROC curves on same plot for model comparison.
    
    Args:
        y_true: True labels
        predictions_dict: {model_name: predicted_probabilities}
        save_path: Path to save figure (optional)
        figsize: Figure size
    
    Returns:
        dict: {model_name: AUC score}
    """
    from core.metrics import roc_curve, auc_score
    
    plt.figure(figsize=figsize)
    
    auc_scores = {}
    colors = plt.cm.Set3(np.linspace(0, 1, len(predictions_dict)))
    
    # Plot each model's ROC curve
    for (model_name, y_pred_proba), color in zip(predictions_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = auc_score(y_true, y_pred_proba)
        auc_scores[model_name] = auc
        
        plt.plot(fpr, tpr, linewidth=2.5, color=color, 
                label=f'{model_name} (AUC = {auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-ROC curve to {save_path}")
    
    return auc_scores


def plot_model_comparison(results_df, metric='accuracy', save_path=None, figsize=(12, 6)):
    """
    Plot bar chart comparing models across a metric.
    
    Args:
        results_df: DataFrame with columns ['model', metric]
        metric: Metric to compare ('accuracy', 'precision', 'recall', 'f1', 'auc')
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Sort by metric value
    results_sorted = results_df.sort_values(metric, ascending=False)
    
    # Create bar plot
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results_sorted)))
    bars = plt.bar(range(len(results_sorted)), results_sorted[metric], color=colors, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, results_sorted[metric])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(range(len(results_sorted)), results_sorted['model'], rotation=45, ha='right')
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.title(f'Model Comparison - {metric.capitalize()}', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.1])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    return plt.gcf()


def plot_training_curves(loss_history, accuracy_history=None, title='Training Curves',
                        save_path=None, figsize=(12, 5)):
    """
    Plot training loss and accuracy curves.
    
    Args:
        loss_history: List of loss values per epoch
        accuracy_history: List of accuracy values per epoch (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    if accuracy_history is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        ax1.plot(loss_history, linewidth=2, color='red')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Accuracy plot
        ax2.plot(accuracy_history, linewidth=2, color='green')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training Accuracy', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
    else:
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, linewidth=2, color='red')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return plt.gcf()


def plot_feature_importance(feature_names, importance_values, top_n=20,
                           title='Feature Importance', save_path=None, figsize=(10, 8)):
    """
    Plot feature importance as horizontal bar chart.
    
    Args:
        feature_names: List of feature names
        importance_values: Array of importance values
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Sort by importance
    indices = np.argsort(importance_values)[::-1][:top_n]
    
    top_features = [feature_names[i] for i in indices]
    top_values = importance_values[indices]
    
    # Create horizontal bar plot
    plt.figure(figsize=figsize)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
    
    plt.barh(range(len(top_features)), top_values, color=colors, edgecolor='black')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest at top
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance to {save_path}")
    
    return plt.gcf()


def plot_metrics_heatmap(results_df, save_path=None, figsize=(12, 8)):
    """
    Plot heatmap of all metrics for all models.
    
    Args:
        results_df: DataFrame with models as rows, metrics as columns
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(results_df.set_index('model'), annot=True, fmt='.4f', 
                cmap='YlGnBu', cbar_kws={'label': 'Score'},
                linewidths=0.5, linecolor='gray')
    
    plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
    plt.ylabel('Model', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics heatmap to {save_path}")
    
    return plt.gcf()


def create_results_summary_figure(results_df, best_model_name, save_path=None):
    """
    Create comprehensive summary figure with multiple subplots.
    
    Args:
        results_df: DataFrame with model results
        best_model_name: Name of best performing model
        save_path: Path to save figure (optional)
    
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy comparison
    ax1 = fig.add_subplot(gs[0, :])
    results_sorted = results_df.sort_values('accuracy', ascending=False)
    colors = ['gold' if m == best_model_name else 'steelblue' for m in results_sorted['model']]
    ax1.bar(range(len(results_sorted)), results_sorted['accuracy'], color=colors, edgecolor='black')
    ax1.set_xticks(range(len(results_sorted)))
    ax1.set_xticklabels(results_sorted['model'], rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Add value labels
    for i, val in enumerate(results_sorted['accuracy']):
        ax1.text(i, val + 0.02, f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
    
    # 2. Precision comparison
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.barh(range(len(results_sorted)), results_sorted['precision'], color='coral', edgecolor='black')
    ax2.set_yticks(range(len(results_sorted)))
    ax2.set_yticklabels(results_sorted['model'], fontsize=9)
    ax2.set_xlabel('Precision')
    ax2.set_title('Precision', fontweight='bold')
    ax2.invert_yaxis()
    
    # 3. Recall comparison
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.barh(range(len(results_sorted)), results_sorted['recall'], color='lightgreen', edgecolor='black')
    ax3.set_yticks(range(len(results_sorted)))
    ax3.set_yticklabels(results_sorted['model'], fontsize=9)
    ax3.set_xlabel('Recall')
    ax3.set_title('Recall', fontweight='bold')
    ax3.invert_yaxis()
    
    # 4. F1-Score comparison
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.barh(range(len(results_sorted)), results_sorted['f1'], color='plum', edgecolor='black')
    ax4.set_yticks(range(len(results_sorted)))
    ax4.set_yticklabels(results_sorted['model'], fontsize=9)
    ax4.set_xlabel('F1-Score')
    ax4.set_title('F1-Score', fontweight='bold')
    ax4.invert_yaxis()
    
    # 5. AUC comparison
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.barh(range(len(results_sorted)), results_sorted['auc'], color='skyblue', edgecolor='black')
    ax5.set_yticks(range(len(results_sorted)))
    ax5.set_yticklabels(results_sorted['model'], fontsize=9)
    ax5.set_xlabel('AUC-ROC')
    ax5.set_title('AUC-ROC', fontweight='bold')
    ax5.invert_yaxis()
    
    # 6. Metrics radar chart for best model
    ax6 = fig.add_subplot(gs[2, 1:], projection='polar')
    best_model_row = results_df[results_df['model'] == best_model_name].iloc[0]
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    values = [best_model_row[m] for m in metrics]
    values += values[:1]  # Close the circle
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    ax6.plot(angles, values, 'o-', linewidth=2, color='red', label=best_model_name)
    ax6.fill(angles, values, alpha=0.25, color='red')
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels([m.capitalize() for m in metrics])
    ax6.set_ylim(0, 1)
    ax6.set_title(f'Best Model: {best_model_name}', fontweight='bold', pad=20)
    ax6.grid(True)
    
    plt.suptitle('Cardiovascular Disease Prediction - Model Performance Summary', 
                fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary figure to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test visualization functions
    print("Testing Visualization Module (Matplotlib/Seaborn)...")
    print("=" * 60)
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Generate test data
    np.random.seed(42)
    n_samples = 200
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = y_true.copy()
    y_pred[np.random.choice(n_samples, 20, replace=False)] = 1 - y_pred[np.random.choice(n_samples, 20, replace=False)]
    y_pred_proba = np.random.rand(n_samples) * 0.3 + y_true * 0.5
    
    print("\n1. Testing Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, title='Test Confusion Matrix')
    plt.show()
    print("   ✓ Confusion matrix plotted!")
    
    print("\n2. Testing ROC Curve...")
    auc = plot_roc_curve(y_true, y_pred_proba, model_name='Test Model')
    plt.show()
    print(f"   ✓ ROC curve plotted! AUC = {auc:.3f}")
    
    print("\n3. Testing Model Comparison...")
    import pandas as pd
    results_df = pd.DataFrame({
        'model': ['Model A', 'Model B', 'Model C'],
        'accuracy': [0.85, 0.92, 0.88],
        'precision': [0.83, 0.90, 0.86],
        'recall': [0.87, 0.94, 0.90],
        'f1': [0.85, 0.92, 0.88],
        'auc': [0.88, 0.95, 0.91]
    })
    
    plot_model_comparison(results_df, metric='accuracy')
    plt.show()
    print("   ✓ Model comparison plotted!")
    
    print("\n4. Testing Feature Importance...")
    feature_names = [f'Feature_{i}' for i in range(20)]
    importance_values = np.random.rand(20)
    plot_feature_importance(feature_names, importance_values, top_n=10)
    plt.show()
    print("   ✓ Feature importance plotted!")
    
    print("\n5. Testing Metrics Heatmap...")
    plot_metrics_heatmap(results_df)
    plt.show()
    print("   ✓ Metrics heatmap plotted!")
    
    print("\n" + "=" * 60)
    print("✓ All visualization functions working!")
    print("✓ Using matplotlib/seaborn ONLY - NO sklearn plotting!")
