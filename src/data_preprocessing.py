"""Data preprocessing module for AMR pipeline."""

import re
import logging
from typing import Tuple, Optional

import pandas as pd
import numpy as np

from .utils import get_data_dir, get_interpretation_columns, get_mic_columns

logger = logging.getLogger('amr_pipeline')


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load raw data from CSV file.
    
    Args:
        filepath: Path to raw data file. If None, uses default location.
        
    Returns:
        Raw DataFrame
    """
    if filepath is None:
        filepath = get_data_dir() / 'raw_data.csv'
    
    df = pd.read_csv(filepath)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    
    def clean_name(name: str) -> str:
        # Convert to lowercase
        name = name.lower().strip()
        # Replace slashes and spaces with underscores
        name = re.sub(r'[/\s]+', '_', name)
        # Remove special characters except underscores
        name = re.sub(r'[^a-z0-9_]', '', name)
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name
    
    df.columns = [clean_name(col) for col in df.columns]
    logger.info(f"Standardized {len(df.columns)} column names")
    return df


def clean_mic_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean MIC values by removing special symbols.
    
    Handles values like:
    - "<=2" -> 2
    - ">=32" -> 32
    - "*R" -> "R" (for interpretation columns)
    - "4*" -> 4
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned MIC values
    """
    df = df.copy()
    mic_cols = get_mic_columns(df)
    
    for col in mic_cols:
        if col in df.columns:
            # Convert to string, remove symbols, then try to convert to numeric
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace(r'^[<>=]+', '', regex=True)
            df[col] = df[col].str.replace(r'\*', '', regex=True)
            df[col] = df[col].replace('', np.nan).replace('nan', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(f"Cleaned {len(mic_cols)} MIC columns")
    return df


def clean_interpretation_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean interpretation values (S/I/R).
    
    Handles values like "*R" -> "R", "*S" -> "S"
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned interpretation values
    """
    df = df.copy()
    int_cols = get_interpretation_columns(df)
    
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.upper()
            df[col] = df[col].str.replace(r'\*', '', regex=True)
            df[col] = df[col].str.strip()
            # Keep only S, I, R, or missing
            df[col] = df[col].apply(
                lambda x: x if x in ['S', 'I', 'R'] else np.nan
            )
    
    logger.info(f"Cleaned {len(int_cols)} interpretation columns")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """Handle missing values in the dataset.
    
    For interpretation columns:
    - Drop columns with >threshold missing values
    - Impute remaining missing values with mode (most common interpretation)
    
    Args:
        df: Input DataFrame
        threshold: Maximum proportion of missing values allowed (default: 0.5)
        
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    int_cols = get_interpretation_columns(df)
    
    # Check missing proportions
    cols_to_drop = []
    for col in int_cols:
        if col in df.columns:
            missing_prop = df[col].isna().sum() / len(df)
            if missing_prop > threshold:
                cols_to_drop.append(col)
    
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Impute remaining interpretation columns with mode
    int_cols = get_interpretation_columns(df)
    for col in int_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
            else:
                df[col] = df[col].fillna('S')  # Default to Susceptible
    
    logger.info("Handled missing values in interpretation columns")
    return df


def clean_organism_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize organism/bacterial species names.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized organism names
    """
    df = df.copy()
    
    # Map raw organism names to standardized names
    organism_mapping = {
        'klebsiella pneumoniae ssp pneumoniae': 'klebsiella_pneumoniae',
        'escherichia coli': 'escherichia_coli',
        'enterobacter cloacae complex': 'enterobacter_cloacae_complex',
        'enterobacter aerogenes': 'enterobacter_aerogenes',
        'salmonella spp.': 'salmonella_group',
        'salmonella spp': 'salmonella_group',
        'pseudomonas aeruginosa': 'pseudomonas_aeruginosa',
        'vibrio spp.': 'vibrio_species',
        'vibrio spp': 'vibrio_species',
    }
    
    if 'organism' in df.columns:
        df['organism'] = df['organism'].str.lower().str.strip()
        df['bacterial_species'] = df['organism'].map(
            lambda x: organism_mapping.get(x, x.replace(' ', '_'))
        )
        logger.info(f"Standardized organism names. Unique species: {df['bacterial_species'].nunique()}")
    
    return df


def clean_sample_source(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize sample source names.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized sample source names
    """
    df = df.copy()
    
    source_mapping = {
        'drinking water': 'drinking_water',
        'river water': 'river_water',
        'lake water': 'lake_water',
        'effluent water': 'effluent_water',
        'fish tilapia': 'fish_tilapia',
        'fish gusaw': 'fish_gusaw',
        'fish banak': 'fish_banak',
        'fish kaolang': 'fish_kaolang',
    }
    
    if 'sample_source' in df.columns:
        df['sample_source'] = df['sample_source'].str.lower().str.strip()
        df['sample_source'] = df['sample_source'].map(
            lambda x: source_mapping.get(x, x.replace(' ', '_'))
        )
        logger.info(f"Standardized sample sources. Unique sources: {df['sample_source'].nunique()}")
    
    return df


def preprocess_data(
    df: pd.DataFrame,
    missing_threshold: float = 0.5
) -> pd.DataFrame:
    """Run full preprocessing pipeline on raw data.
    
    Args:
        df: Raw DataFrame
        missing_threshold: Maximum proportion of missing values allowed
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Starting data preprocessing pipeline...")
    
    # Step 1: Standardize column names
    df = standardize_column_names(df)
    
    # Step 2: Clean organism and sample source names
    df = clean_organism_names(df)
    df = clean_sample_source(df)
    
    # Step 3: Clean MIC and interpretation values
    df = clean_mic_values(df)
    df = clean_interpretation_values(df)
    
    # Step 4: Handle missing values
    df = handle_missing_values(df, threshold=missing_threshold)
    
    # Step 5: Standardize location columns
    for col in ['national_site', 'local_site']:
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip().str.replace(' ', '_')
    
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df


def save_processed_data(df: pd.DataFrame, filepath: Optional[str] = None) -> str:
    """Save processed data to CSV file.
    
    Args:
        df: Processed DataFrame
        filepath: Output path. If None, uses default location.
        
    Returns:
        Path to saved file
    """
    if filepath is None:
        filepath = get_data_dir() / 'processed_data.csv'
    
    df.to_csv(filepath, index=False)
    logger.info(f"Saved processed data to {filepath}")
    return str(filepath)
