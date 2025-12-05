"""Model training module for AMR pipeline."""

import logging
import warnings
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedGroupKFold,
    cross_validate
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier

from .utils import save_model, get_models_dir
from .feature_engineering import get_feature_columns

logger = logging.getLogger('amr_pipeline')
warnings.filterwarnings('ignore')


def get_models(task: str = 'multiclass') -> Dict[str, Any]:
    """Get dictionary of models to train.
    
    Args:
        task: 'multiclass' or 'binary'
        
    Returns:
        Dictionary of model name to model instance
    """
    random_state = 42
    
    if task == 'binary':
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1,
                class_weight='balanced'
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1,
                verbose=-1, class_weight='balanced'
            ),
            'logistic_regression': LogisticRegression(
                random_state=random_state, max_iter=1000,
                class_weight='balanced', solver='lbfgs'
            ),
            'svm': SVC(
                random_state=random_state, probability=True,
                class_weight='balanced'
            ),
            'knn': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'naive_bayes': GaussianNB()
        }
    else:
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1,
                class_weight='balanced_subsample'
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1,
                verbose=-1, class_weight='balanced'
            ),
            'logistic_regression': LogisticRegression(
                random_state=random_state, max_iter=1000,
                class_weight='balanced', solver='lbfgs', multi_class='multinomial'
            ),
            'svm': SVC(
                random_state=random_state, probability=True,
                class_weight='balanced', decision_function_shape='ovr'
            ),
            'knn': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'naive_bayes': GaussianNB()
        }
    
    return models


def prepare_data_for_task(
    df: pd.DataFrame,
    task: str
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, LabelEncoder]:
    """Prepare features and target for a specific task.
    
    Args:
        df: Engineered DataFrame
        task: One of 'species', 'resistance', 'species_ast_only', 'mdr'
        
    Returns:
        Tuple of (X, y, groups, label_encoder)
    """
    # Get feature columns for task
    feature_cols = get_feature_columns(df, task)
    
    # Get target variable
    if task == 'species':
        target_col = 'bacterial_species'
    elif task == 'resistance':
        target_col = 'resistance_category'
    elif task == 'species_ast_only':
        target_col = 'bacterial_species'
    elif task == 'mdr':
        target_col = 'mdr_flag'
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Prepare X and y
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values in features
    X = X.fillna(0)
    
    # Get groups for stratified group split
    groups = df['isolate_code'].copy() if 'isolate_code' in df.columns else pd.Series(range(len(df)))
    
    # Encode target if categorical
    label_encoder = LabelEncoder()
    if y.dtype == 'object':
        y = pd.Series(label_encoder.fit_transform(y), index=y.index)
    else:
        label_encoder.fit(y.astype(str))
    
    logger.info(f"Task '{task}': X shape={X.shape}, y classes={len(np.unique(y))}")
    return X, y, groups, label_encoder


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split data into train and test sets with group-stratified sampling.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        groups: Group labels for stratification
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, groups_train, groups_test)
    """
    # Get unique groups and their most common class
    group_df = pd.DataFrame({'group': groups, 'y': y})
    group_class = group_df.groupby('group')['y'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
    
    # Split groups
    unique_groups = group_class.index.values
    group_labels = group_class.values
    
    # Check if all classes have at least 2 samples for stratification
    class_counts = pd.Series(group_labels).value_counts()
    can_stratify = (class_counts >= 2).all()
    
    try:
        if can_stratify:
            train_groups, test_groups = train_test_split(
                unique_groups,
                test_size=test_size,
                random_state=random_state,
                stratify=group_labels
            )
        else:
            # Fallback to non-stratified split
            logger.warning("Some classes have fewer than 2 samples, using non-stratified split")
            train_groups, test_groups = train_test_split(
                unique_groups,
                test_size=test_size,
                random_state=random_state
            )
    except ValueError as e:
        # Fallback to non-stratified split
        logger.warning(f"Stratified split failed: {e}. Using non-stratified split.")
        train_groups, test_groups = train_test_split(
            unique_groups,
            test_size=test_size,
            random_state=random_state
        )
    
    # Create train/test masks
    train_mask = groups.isin(train_groups)
    test_mask = groups.isin(test_groups)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    groups_train = groups[train_mask]
    groups_test = groups[test_mask]
    
    logger.info(f"Split data: train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test, groups_train, groups_test


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    scoring: List[str] = None
) -> Dict[str, np.ndarray]:
    """Perform group-stratified cross-validation.
    
    Args:
        model: Model instance to evaluate
        X: Feature DataFrame
        y: Target Series
        groups: Group labels
        n_splits: Number of CV folds
        scoring: List of scoring metrics
        
    Returns:
        Dictionary of CV results
    """
    if scoring is None:
        scoring = ['accuracy', 'f1_macro', 'balanced_accuracy']
    
    # Use StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    try:
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            groups=groups,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
    except ValueError as e:
        # Fallback to regular StratifiedKFold if groups cause issues
        logger.warning(f"StratifiedGroupKFold failed, using regular CV: {e}")
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
    
    return cv_results


def train_models_for_task(
    df: pd.DataFrame,
    task: str,
    n_cv_splits: int = 5
) -> Dict[str, Any]:
    """Train all models for a specific task.
    
    Args:
        df: Engineered DataFrame
        task: Task name
        n_cv_splits: Number of cross-validation folds
        
    Returns:
        Dictionary with training results
    """
    logger.info(f"Training models for task: {task}")
    
    # Prepare data
    X, y, groups, label_encoder = prepare_data_for_task(df, task)
    
    # Split data
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_data(
        X, y, groups
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for compatibility
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Get models
    model_type = 'binary' if task == 'mdr' else 'multiclass'
    models = get_models(model_type)
    
    # Define scoring metrics
    if task == 'mdr':
        scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    else:
        scoring = ['accuracy', 'f1_macro', 'balanced_accuracy']
    
    results = {
        'task': task,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'feature_columns': list(X.columns),
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'models': {},
        'cv_results': {},
        'trained_models': {}
    }
    
    for name, model in models.items():
        logger.info(f"  Training {name}...")
        
        # Cross-validation
        try:
            cv_results = cross_validate_model(
                model, X_train_scaled, y_train, groups_train,
                n_splits=n_cv_splits, scoring=scoring
            )
            results['cv_results'][name] = cv_results
        except Exception as e:
            logger.warning(f"  CV failed for {name}: {e}")
            results['cv_results'][name] = None
        
        # Train on full training set
        try:
            model.fit(X_train_scaled, y_train)
            results['trained_models'][name] = model
            results['models'][name] = model.__class__.__name__
        except Exception as e:
            logger.warning(f"  Training failed for {name}: {e}")
    
    logger.info(f"Completed training for task: {task}")
    return results


def save_task_models(results: Dict[str, Any], task: str) -> None:
    """Save trained models for a task.
    
    Args:
        results: Training results dictionary
        task: Task name
    """
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in results['trained_models'].items():
        filepath = models_dir / f"{task}_{name}.joblib"
        save_model(model, f"{task}_{name}.joblib")
        logger.info(f"Saved model: {filepath}")
    
    # Save scaler and encoder
    save_model(results['scaler'], f"{task}_scaler.joblib")
    save_model(results['label_encoder'], f"{task}_label_encoder.joblib")
