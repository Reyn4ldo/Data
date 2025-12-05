"""Model interpretation module for AMR pipeline using SHAP."""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from .utils import get_results_dir

logger = logging.getLogger('amr_pipeline')


def get_shap_explainer(
    model: Any,
    X_train: pd.DataFrame,
    model_name: str
) -> shap.Explainer:
    """Get appropriate SHAP explainer for a model type.
    
    Args:
        model: Trained model
        X_train: Training features
        model_name: Name of the model
        
    Returns:
        SHAP explainer instance
    """
    # Sample data for background (for efficiency)
    if len(X_train) > 100:
        background = shap.sample(X_train, 100, random_state=42)
    else:
        background = X_train
    
    if model_name in ['random_forest', 'xgboost']:
        # Tree-based models
        return shap.TreeExplainer(model)
    elif model_name in ['logistic_regression']:
        # Linear models
        return shap.LinearExplainer(model, background)
    else:
        # Kernel SHAP for other models (slower)
        return shap.KernelExplainer(model.predict_proba, background)


def calculate_shap_values(
    model: Any,
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    model_name: str,
    max_samples: int = 100
) -> Optional[np.ndarray]:
    """Calculate SHAP values for model predictions.
    
    Args:
        model: Trained model
        X_test: Test features
        X_train: Training features (for background)
        model_name: Name of the model
        max_samples: Maximum samples to explain
        
    Returns:
        SHAP values array or None if failed
    """
    try:
        # Limit samples for efficiency
        if len(X_test) > max_samples:
            X_explain = X_test.sample(max_samples, random_state=42)
        else:
            X_explain = X_test
        
        explainer = get_shap_explainer(model, X_train, model_name)
        shap_values = explainer.shap_values(X_explain)
        
        logger.info(f"Calculated SHAP values for {model_name}")
        return shap_values, X_explain
    
    except Exception as e:
        logger.warning(f"SHAP calculation failed for {model_name}: {e}")
        return None, None


def plot_shap_summary(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    model_name: str,
    task: str,
    filepath: str,
    max_display: int = 20
) -> None:
    """Create SHAP summary plot showing feature importance.
    
    Args:
        shap_values: SHAP values array
        X_explain: Features used for explanation
        model_name: Name of the model
        task: Task name
        filepath: Path to save the figure
        max_display: Maximum features to display
    """
    plt.figure(figsize=(12, 10))
    
    # Handle multi-class case
    if isinstance(shap_values, list):
        # For multi-class, take mean absolute SHAP values across classes
        shap_values_combined = np.abs(np.array(shap_values)).mean(axis=0)
        shap.summary_plot(
            shap_values_combined, X_explain,
            max_display=max_display, show=False, plot_type='bar'
        )
    else:
        shap.summary_plot(
            shap_values, X_explain,
            max_display=max_display, show=False
        )
    
    plt.title(f'SHAP Feature Importance - {model_name} ({task})')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP summary plot: {filepath}")


def plot_shap_bar(
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    model_name: str,
    task: str,
    filepath: str,
    max_display: int = 20
) -> None:
    """Create SHAP bar plot of mean absolute importance.
    
    Args:
        shap_values: SHAP values array
        X_explain: Features used for explanation
        model_name: Name of the model
        task: Task name
        filepath: Path to save the figure
        max_display: Maximum features to display
    """
    try:
        plt.figure(figsize=(10, 8))
        
        # Handle multi-class case - shap_values can be a list of arrays
        if isinstance(shap_values, list):
            # List of arrays for each class
            shap_values_combined = np.abs(np.array(shap_values)).mean(axis=0)
            if shap_values_combined.ndim > 1:
                mean_abs_shap = np.abs(shap_values_combined).mean(axis=0)
            else:
                mean_abs_shap = np.abs(shap_values_combined)
        elif shap_values.ndim == 3:
            # 3D array: (n_classes, n_samples, n_features)
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
        elif shap_values.ndim == 2:
            # 2D array: (n_samples, n_features)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        else:
            mean_abs_shap = np.abs(shap_values)
        
        # Ensure it's 1-dimensional
        mean_abs_shap = np.array(mean_abs_shap).ravel()
        
        # Verify lengths match
        feature_names = list(X_explain.columns)
        if len(mean_abs_shap) != len(feature_names):
            logger.warning(f"SHAP values length ({len(mean_abs_shap)}) != features ({len(feature_names)}). Skipping plot.")
            plt.close()
            return
        
        # Create DataFrame for plotting
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=True).tail(max_display)
        
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'Feature Importance - {model_name} ({task})')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP bar plot: {filepath}")
    except Exception as e:
        logger.warning(f"Could not create SHAP bar plot: {e}")
        plt.close()


def get_feature_importance_from_model(
    model: Any,
    feature_names: List[str],
    model_name: str
) -> pd.DataFrame:
    """Extract feature importance directly from model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
        
    Returns:
        DataFrame with feature importance
    """
    importance = None
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        coef = model.coef_
        if coef.ndim > 1:
            importance = np.abs(coef).mean(axis=0)
        else:
            importance = np.abs(coef)
    
    if importance is not None:
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return df
    
    return None


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str,
    task: str,
    filepath: str,
    max_display: int = 20
) -> None:
    """Plot feature importance from model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
        task: Task name
        filepath: Path to save the figure
        max_display: Maximum features to display
    """
    importance_df = get_feature_importance_from_model(model, feature_names, model_name)
    
    if importance_df is None:
        logger.warning(f"Cannot extract feature importance for {model_name}")
        return
    
    # Take top features
    plot_df = importance_df.head(max_display).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(plot_df['feature'], plot_df['importance'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {model_name} ({task})')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature importance plot: {filepath}")


def interpret_models(
    results: Dict[str, Any],
    task: str,
    best_model_name: str,
    use_shap: bool = True
) -> Dict[str, Any]:
    """Run interpretation analysis on trained models.
    
    Args:
        results: Training results dictionary
        task: Task name
        best_model_name: Name of the best model to focus on
        use_shap: Whether to use SHAP values
        
    Returns:
        Dictionary with interpretation results
    """
    logger.info(f"Interpreting models for task: {task}")
    
    figures_dir = get_results_dir() / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    interpretation_results = {}
    
    for model_name, model in results['trained_models'].items():
        logger.info(f"  Interpreting {model_name}...")
        
        # Feature importance from model
        importance_df = get_feature_importance_from_model(
            model, results['feature_columns'], model_name
        )
        if importance_df is not None:
            interpretation_results[f'{model_name}_importance'] = importance_df
            
            # Plot feature importance
            plot_feature_importance(
                model, results['feature_columns'], model_name, task,
                str(figures_dir / f'{task}_{model_name}_importance.png')
            )
        
        # SHAP analysis for best model only (computationally expensive)
        if use_shap and model_name == best_model_name:
            if model_name in ['random_forest', 'xgboost']:
                shap_values, X_explain = calculate_shap_values(
                    model,
                    results['X_test_scaled'],
                    results['X_train_scaled'],
                    model_name,
                    max_samples=50
                )
                
                if shap_values is not None:
                    interpretation_results[f'{model_name}_shap'] = shap_values
                    
                    # Plot SHAP summary
                    plot_shap_bar(
                        shap_values, X_explain, model_name, task,
                        str(figures_dir / f'{task}_{model_name}_shap_bar.png')
                    )
    
    logger.info(f"Completed interpretation for task: {task}")
    return interpretation_results
