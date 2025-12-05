"""Model evaluation module for AMR pipeline."""

import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

from .utils import get_results_dir, save_metrics

logger = logging.getLogger('amr_pipeline')


def evaluate_multiclass_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_encoder: Any,
    model_name: str = 'model'
) -> Dict[str, Any]:
    """Evaluate a multi-class classification model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder for target
        model_name: Name of the model
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    try:
        y_proba = model.predict_proba(X_test)
    except AttributeError:
        y_proba = None
    
    # Calculate metrics
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
    }
    
    # ROC-AUC (One-vs-Rest)
    if y_proba is not None:
        try:
            n_classes = len(label_encoder.classes_)
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            if n_classes == 2:
                metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_test_bin, y_proba, average='macro', multi_class='ovr'
                )
        except ValueError as e:
            logger.warning(f"ROC-AUC calculation failed: {e}")
            metrics['roc_auc_ovr'] = None
    else:
        metrics['roc_auc_ovr'] = None
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    
    # Classification report - use only labels present in test set
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    present_class_names = [label_encoder.classes_[i] for i in unique_labels if i < len(label_encoder.classes_)]
    metrics['classification_report'] = classification_report(
        y_test, y_pred, labels=unique_labels, target_names=present_class_names, 
        output_dict=True, zero_division=0
    )
    
    return metrics


def evaluate_binary_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = 'model'
) -> Dict[str, Any]:
    """Evaluate a binary classification model (MDR prediction).
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_proba = None
    
    # Calculate metrics
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
    }
    
    # ROC-AUC and PR-AUC
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
            metrics['pr_auc'] = auc(recall_curve, precision_curve)
        except ValueError as e:
            logger.warning(f"AUC calculation failed: {e}")
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
    else:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    
    return metrics


def evaluate_all_models(
    results: Dict[str, Any],
    task: str
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all trained models for a task.
    
    Args:
        results: Training results dictionary
        task: Task name
        
    Returns:
        Dictionary of model name to metrics
    """
    logger.info(f"Evaluating models for task: {task}")
    
    all_metrics = {}
    
    for model_name, model in results['trained_models'].items():
        if task == 'mdr':
            metrics = evaluate_binary_model(
                model,
                results['X_test_scaled'],
                results['y_test'],
                model_name
            )
        else:
            metrics = evaluate_multiclass_model(
                model,
                results['X_test_scaled'],
                results['y_test'],
                results['label_encoder'],
                model_name
            )
        
        all_metrics[model_name] = metrics
        logger.info(f"  {model_name}: F1={metrics.get('f1_macro', metrics.get('f1', 0)):.4f}")
    
    return all_metrics


def select_best_model(
    all_metrics: Dict[str, Dict[str, Any]],
    cv_results: Dict[str, Any],
    task: str
) -> Tuple[str, Dict[str, Any]]:
    """Select the best model based on evaluation metrics.
    
    Selection criteria:
    1. Highest macro F1 (or F1 for binary)
    2. Highest ROC-AUC
    3. Lowest overfitting gap (train vs validation)
    4. Stability across folds
    
    Args:
        all_metrics: Dictionary of model metrics
        cv_results: Cross-validation results
        task: Task name
        
    Returns:
        Tuple of (best model name, selection info)
    """
    scores = {}
    
    for model_name, metrics in all_metrics.items():
        # Primary metric
        if task == 'mdr':
            f1 = metrics.get('f1', 0) or 0
            roc_auc = metrics.get('roc_auc', 0) or 0
        else:
            f1 = metrics.get('f1_macro', 0) or 0
            roc_auc = metrics.get('roc_auc_ovr', 0) or 0
        
        # Calculate overfitting gap from CV results
        overfitting_gap = 0
        cv_stability = 1.0
        if cv_results.get(model_name) is not None:
            cv = cv_results[model_name]
            if task == 'mdr':
                train_score = cv.get('train_f1', cv.get('train_accuracy', [0]))
                val_score = cv.get('test_f1', cv.get('test_accuracy', [0]))
            else:
                train_score = cv.get('train_f1_macro', cv.get('train_accuracy', [0]))
                val_score = cv.get('test_f1_macro', cv.get('test_accuracy', [0]))
            
            if len(train_score) > 0 and len(val_score) > 0:
                overfitting_gap = np.mean(train_score) - np.mean(val_score)
                cv_stability = 1 / (np.std(val_score) + 0.01)  # Higher is better
        
        # Combined score (weighted)
        combined_score = (
            0.5 * f1 +
            0.3 * roc_auc +
            0.1 * max(0, 1 - overfitting_gap) +
            0.1 * min(1, cv_stability / 10)
        )
        
        scores[model_name] = {
            'f1': f1,
            'roc_auc': roc_auc,
            'overfitting_gap': overfitting_gap,
            'cv_stability': cv_stability,
            'combined_score': combined_score
        }
    
    # Select best model
    best_model = max(scores.items(), key=lambda x: x[1]['combined_score'])
    logger.info(f"Best model for {task}: {best_model[0]} (score: {best_model[1]['combined_score']:.4f})")
    
    return best_model[0], scores


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    filepath: str
) -> None:
    """Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        title: Plot title
        filepath: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix: {filepath}")


def plot_roc_curves(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    filepath: str
) -> None:
    """Plot ROC curves for all models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        task: Task name
        filepath: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                
                if task == 'mdr' or y_proba.shape[1] == 2:
                    # Binary classification
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
                else:
                    # Multi-class: plot macro-averaged ROC
                    n_classes = y_proba.shape[1]
                    y_test_bin = label_binarize(y_test, classes=range(n_classes))
                    
                    # Compute micro-average ROC
                    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{name} (micro-avg AUC = {roc_auc:.3f})')
        except Exception as e:
            logger.warning(f"Could not plot ROC for {name}: {e}")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {task}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    logger.info(f"Saved ROC curves: {filepath}")


def plot_calibration_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    filepath: str
) -> None:
    """Plot calibration curve for MDR probability prediction.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        filepath: Path to save the figure
    """
    if not hasattr(model, 'predict_proba'):
        logger.warning(f"Model {model_name} does not support probability prediction")
        return
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    plt.figure(figsize=(8, 8))
    
    # Plot calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_proba, n_bins=10
    )
    
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=model_name)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve - {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    logger.info(f"Saved calibration curve: {filepath}")


def create_comparison_table(
    all_metrics: Dict[str, Dict[str, Any]],
    task: str
) -> pd.DataFrame:
    """Create a comparison table of all models.
    
    Args:
        all_metrics: Dictionary of model metrics
        task: Task name
        
    Returns:
        DataFrame with comparison metrics
    """
    rows = []
    
    for model_name, metrics in all_metrics.items():
        if task == 'mdr':
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1': metrics.get('f1', 0),
                'ROC-AUC': metrics.get('roc_auc', 0) or 0,
                'PR-AUC': metrics.get('pr_auc', 0) or 0,
            }
        else:
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Balanced Accuracy': metrics.get('balanced_accuracy', 0),
                'F1 (Macro)': metrics.get('f1_macro', 0),
                'F1 (Weighted)': metrics.get('f1_weighted', 0),
                'Precision (Macro)': metrics.get('precision_macro', 0),
                'Recall (Macro)': metrics.get('recall_macro', 0),
                'ROC-AUC (OvR)': metrics.get('roc_auc_ovr', 0) or 0,
            }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values(by='F1' if task == 'mdr' else 'F1 (Macro)', ascending=False)
    return df


def save_evaluation_results(
    all_metrics: Dict[str, Dict[str, Any]],
    comparison_table: pd.DataFrame,
    task: str
) -> None:
    """Save evaluation results to files.
    
    Args:
        all_metrics: Dictionary of model metrics
        comparison_table: Comparison DataFrame
        task: Task name
    """
    results_dir = get_results_dir()
    metrics_dir = results_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics JSON
    save_metrics(all_metrics, f'{task}_metrics.json')
    
    # Save comparison table
    comparison_table.to_csv(metrics_dir / f'{task}_comparison.csv', index=False)
    logger.info(f"Saved comparison table: {metrics_dir / f'{task}_comparison.csv'}")
