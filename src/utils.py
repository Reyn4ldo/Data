"""Utility functions for the AMR ML pipeline."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import joblib


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('amr_pipeline')


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path to project root directory
    """
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Get the data directory path.
    
    Returns:
        Path to data directory
    """
    return get_project_root() / 'data'


def get_models_dir() -> Path:
    """Get the models directory path.
    
    Returns:
        Path to models directory
    """
    return get_project_root() / 'models'


def get_results_dir() -> Path:
    """Get the results directory path.
    
    Returns:
        Path to results directory
    """
    return get_project_root() / 'results'


def save_model(model: Any, filename: str) -> str:
    """Save a trained model to disk.
    
    Args:
        model: Trained model object
        filename: Name for the saved file
        
    Returns:
        Full path to saved model file
    """
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    filepath = models_dir / filename
    joblib.dump(model, filepath)
    return str(filepath)


def load_model(filename: str) -> Any:
    """Load a trained model from disk.
    
    Args:
        filename: Name of the model file
        
    Returns:
        Loaded model object
    """
    filepath = get_models_dir() / filename
    return joblib.load(filepath)


def save_metrics(metrics: Dict[str, Any], filename: str) -> str:
    """Save evaluation metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filename: Name for the saved file
        
    Returns:
        Full path to saved metrics file
    """
    metrics_dir = get_results_dir() / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    filepath = metrics_dir / filename
    
    # Convert numpy types to Python native types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    metrics = convert_numpy(metrics)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    return str(filepath)


def load_metrics(filename: str) -> Dict[str, Any]:
    """Load evaluation metrics from JSON file.
    
    Args:
        filename: Name of the metrics file
        
    Returns:
        Dictionary of metrics
    """
    filepath = get_results_dir() / 'metrics' / filename
    with open(filepath, 'r') as f:
        return json.load(f)


# Antibiotic classes for MDR classification
ANTIBIOTIC_CLASSES = {
    'penicillins': ['ampicillin_int'],
    'beta_lactam_combinations': ['amoxicillin_clavulanic_acid_int'],
    'cephalosporins': [
        'ceftaroline_int', 'cefalexin_int', 'cefalotin_int',
        'cefpodoxime_int', 'cefotaxime_int', 'cefovecin_int',
        'ceftiofur_int', 'ceftazidime_avibactam_int'
    ],
    'carbapenems': ['imepenem_int'],
    'aminoglycosides': ['amikacin_int', 'gentamicin_int', 'neomycin_int'],
    'quinolones': [
        'nalidixic_acid_int', 'enrofloxacin_int',
        'marbofloxacin_int', 'pradofloxacin_int'
    ],
    'tetracyclines': ['doxycycline_int', 'tetracycline_int'],
    'nitrofurans': ['nitrofurantoin_int'],
    'phenicols': ['chloramphenicol_int'],
    'folate_pathway_inhibitors': ['trimethoprim_sulfamethazole_int']
}


def get_interpretation_columns(df: pd.DataFrame) -> List[str]:
    """Get all antibiotic interpretation columns (ending with _int).
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of interpretation column names
    """
    return [col for col in df.columns if col.endswith('_int')]


def get_mic_columns(df: pd.DataFrame) -> List[str]:
    """Get all MIC value columns (ending with _mic).
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of MIC column names
    """
    return [col for col in df.columns if col.endswith('_mic')]
