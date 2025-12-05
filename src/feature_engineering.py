"""Feature engineering module for AMR pipeline."""

import logging
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .utils import (
    get_interpretation_columns,
    ANTIBIOTIC_CLASSES
)

logger = logging.getLogger('amr_pipeline')


def encode_sir_values(df: pd.DataFrame) -> pd.DataFrame:
    """Encode S/I/R interpretation values to numeric (0/1/2).
    
    S (Susceptible) -> 0
    I (Intermediate) -> 1
    R (Resistant) -> 2
    
    Args:
        df: Input DataFrame with S/I/R values
        
    Returns:
        DataFrame with encoded interpretation values
    """
    df = df.copy()
    int_cols = get_interpretation_columns(df)
    
    sir_mapping = {'S': 0, 'I': 1, 'R': 2}
    
    for col in int_cols:
        if col in df.columns:
            encoded_col = col.replace('_int', '_encoded')
            df[encoded_col] = df[col].map(sir_mapping)
    
    logger.info(f"Encoded {len(int_cols)} S/I/R columns to numeric values")
    return df


def create_onehot_features(
    df: pd.DataFrame,
    columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, OneHotEncoder]]:
    """One-hot encode categorical features.
    
    Args:
        df: Input DataFrame
        columns: List of columns to one-hot encode
        
    Returns:
        Tuple of (DataFrame with encoded features, dict of encoders)
    """
    df = df.copy()
    encoders = {}
    
    for col in columns:
        if col in df.columns:
            # Fill missing values with 'unknown'
            df[col] = df[col].fillna('unknown').astype(str)
            
            # Get unique values and create dummy columns
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            
            # Store encoder info
            encoders[col] = list(dummies.columns)
            
            # Add dummy columns to dataframe
            df = pd.concat([df, dummies], axis=1)
            
            logger.info(f"One-hot encoded '{col}' into {len(dummies.columns)} columns")
    
    return df, encoders


def create_resistance_category(df: pd.DataFrame) -> pd.DataFrame:
    """Create resistance category based on MAR index.
    
    Categories:
    - Low: mar_index < 0.2
    - Moderate: 0.2 <= mar_index < 0.4
    - High: mar_index >= 0.4
    
    Args:
        df: Input DataFrame with mar_index column
        
    Returns:
        DataFrame with resistance_category column
    """
    df = df.copy()
    
    if 'mar_index' in df.columns:
        def categorize(mar):
            if pd.isna(mar):
                return 'unknown'
            elif mar < 0.2:
                return 'low'
            elif mar < 0.4:
                return 'moderate'
            else:
                return 'high'
        
        df['resistance_category'] = df['mar_index'].apply(categorize)
        logger.info(f"Created resistance categories: {df['resistance_category'].value_counts().to_dict()}")
    else:
        logger.warning("mar_index column not found, cannot create resistance_category")
    
    return df


def create_mdr_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Create Multi-Drug Resistant (MDR) binary flag.
    
    MDR Definition: 
    - Resistance to ≥3 antibiotic classes OR mar_index >= 0.2
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with mdr_flag column
    """
    df = df.copy()
    
    # Count resistant antibiotic classes
    encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
    
    # Check class-level resistance
    class_resistance = {}
    for class_name, antibiotics in ANTIBIOTIC_CLASSES.items():
        class_cols = []
        for ab in antibiotics:
            encoded_col = ab.replace('_int', '_encoded')
            if encoded_col in df.columns:
                class_cols.append(encoded_col)
        
        if class_cols:
            # Class is resistant if any antibiotic in class is R (value == 2)
            class_resistance[class_name] = (df[class_cols] == 2).any(axis=1).astype(int)
    
    # Count resistant classes per isolate
    if class_resistance:
        resistant_classes = pd.DataFrame(class_resistance).sum(axis=1)
        df['num_resistant_classes'] = resistant_classes
        
        # MDR flag: >=3 resistant classes OR mar_index >= 0.2
        mar_condition = df['mar_index'] >= 0.2 if 'mar_index' in df.columns else False
        class_condition = resistant_classes >= 3
        
        df['mdr_flag'] = ((mar_condition) | (class_condition)).astype(int)
        logger.info(f"Created MDR flag. MDR count: {df['mdr_flag'].sum()}, Non-MDR: {(1-df['mdr_flag']).sum()}")
    else:
        # Fallback to mar_index only
        if 'mar_index' in df.columns:
            df['mdr_flag'] = (df['mar_index'] >= 0.2).astype(int)
            logger.info(f"Created MDR flag (mar_index only). MDR count: {df['mdr_flag'].sum()}")
        else:
            logger.warning("Cannot create MDR flag: no encoded columns or mar_index found")
            df['mdr_flag'] = 0
    
    return df


def extract_class_resistance_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary features for resistance to each antibiotic class.
    
    Args:
        df: Input DataFrame with encoded interpretation columns
        
    Returns:
        DataFrame with class resistance pattern columns
    """
    df = df.copy()
    
    for class_name, antibiotics in ANTIBIOTIC_CLASSES.items():
        class_cols = []
        for ab in antibiotics:
            encoded_col = ab.replace('_int', '_encoded')
            if encoded_col in df.columns:
                class_cols.append(encoded_col)
        
        if class_cols:
            # Binary flag: 1 if resistant to any antibiotic in class, 0 otherwise
            df[f'class_{class_name}_resistant'] = (df[class_cols] == 2).any(axis=1).astype(int)
    
    class_cols_created = [col for col in df.columns if col.startswith('class_') and col.endswith('_resistant')]
    logger.info(f"Created {len(class_cols_created)} class resistance pattern features")
    return df


def get_feature_columns(
    df: pd.DataFrame,
    task: str = 'species'
) -> List[str]:
    """Get feature columns for a specific ML task.
    
    Args:
        df: Input DataFrame
        task: Task type ('species', 'resistance', 'species_ast_only', 'mdr')
        
    Returns:
        List of feature column names
    """
    # Encoded interpretation columns
    encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
    
    # One-hot encoded categorical columns
    onehot_cols = []
    for prefix in ['sample_source_', 'national_site_', 'local_site_', 'esbl_']:
        onehot_cols.extend([col for col in df.columns if col.startswith(prefix)])
    
    # Class resistance pattern columns
    class_cols = [col for col in df.columns if col.startswith('class_') and col.endswith('_resistant')]
    
    # Numeric features
    numeric_features = []
    if 'scored_resistance' in df.columns:
        numeric_features.append('scored_resistance')
    if 'num_resistant_classes' in df.columns:
        numeric_features.append('num_resistant_classes')
    
    if task == 'species':
        # All features for species prediction
        features = encoded_cols + onehot_cols + class_cols + numeric_features
        if 'mar_index' in df.columns:
            features.append('mar_index')
    
    elif task == 'resistance':
        # All features except mar_index (it's used to derive target)
        features = encoded_cols + onehot_cols + class_cols
        if 'scored_resistance' in df.columns:
            features.append('scored_resistance')
    
    elif task == 'species_ast_only':
        # Only AST interpretation columns
        features = encoded_cols
    
    elif task == 'mdr':
        # All features except mdr_flag and derived features
        features = encoded_cols + onehot_cols + class_cols
        if 'scored_resistance' in df.columns:
            features.append('scored_resistance')
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Filter to only include columns that exist
    features = [col for col in features if col in df.columns]
    logger.info(f"Task '{task}': {len(features)} feature columns")
    return features


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Step 1: Encode S/I/R values
    df = encode_sir_values(df)
    
    # Step 2: One-hot encode categorical features
    categorical_cols = ['sample_source', 'national_site', 'local_site', 'esbl']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    df, _ = create_onehot_features(df, categorical_cols)
    
    # Step 3: Create resistance category
    df = create_resistance_category(df)
    
    # Step 4: Extract class resistance patterns
    df = extract_class_resistance_patterns(df)
    
    # Step 5: Create MDR flag
    df = create_mdr_flag(df)
    
    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df
