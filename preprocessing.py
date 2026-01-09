"""
Data preprocessing utilities for supervised LDA co-pathology analysis.

This module handles loading, cleaning, and preparing neurodegenerative disease
data for the supervised LDA model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List


def load_wsev_data(csv_path: str):
    """
    Load the WSEV dataset with diagnosis and volumetric features.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing patient data

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with all patient information
    """
    df = pd.read_csv(csv_path)
    df = df[df['DX'] != 'HC'] # (260109) WSK remove healthy controls - applies to  WSEV dataset only 
    print(f"Loaded {len(df)} patients from {csv_path}")
    print(f"Columns: {df.shape[1]}")
    return df


def extract_cortical_features(df: pd.DataFrame):
    """
    Extract cortical regional features (ctx_lh_* and ctx_rh_*).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe with all features

    Returns
    -------
    X_df : pd.DataFrame
        Dataframe with only cortical features
    feature_names : List[str]
        List of cortical feature column names
    """
    # Find all columns starting with ctx_
    cortical_cols = [col for col in df.columns if col.startswith('ctx_')] 
    # cortical_cols = [col for col in df.columns if col.startswith('ctx_')]

    if len(cortical_cols) == 0:
        raise ValueError("No cortical features (ctx_*) found in dataframe")

    X_df = df[cortical_cols].copy()

    # Check for missing values
    if X_df.isnull().any().any():
        print("Warning: Missing values detected in cortical features")
        n_missing = X_df.isnull().sum().sum()
        print(f"Total missing values: {n_missing}")
        print("Filling missing values with column means...")
        X_df = X_df.fillna(X_df.mean())

    print(f"Extracted {len(cortical_cols)} cortical features")

    return X_df, cortical_cols


def encode_diagnoses(df: pd.DataFrame):
    """
    Encode diagnosis labels as integers.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'DX' column

    Returns
    -------
    y : np.ndarray
        Integer-encoded diagnoses (0 to n_classes-1)
    dx_labels : List[str]
        Ordered list of diagnosis labels
    label_map : dict
        Mapping from diagnosis string to integer
    """
    if 'DX' not in df.columns:
        raise ValueError("'DX' column not found in dataframe")

    # Get unique diagnoses and sort alphabetically for consistency
    unique_dx = sorted(df['DX'].unique())

    # Create mapping
    label_map = {dx: i for i, dx in enumerate(unique_dx)}

    # Encode
    y = df['DX'].map(label_map).values

    print(f"Diagnosis distribution:")
    for dx in unique_dx:
        count = (df['DX'] == dx).sum()
        print(f"  {dx}: {count} patients (class {label_map[dx]})")

    return y, unique_dx, label_map


def prepare_slda_inputs(
    df: pd.DataFrame,
    standardize: bool = False
):
    """
    Prepare all inputs needed for sLDA model.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe with diagnosis and features
    standardize : bool, default=False
        Whether to standardize features (z-score normalization)
        Note: WSEV data is already normalized, so this is typically False

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_patients, n_features)
    y : np.ndarray
        Diagnosis labels of shape (n_patients,)
    feature_names : List[str]
        Names of cortical features
    dx_labels : List[str]
        Ordered list of diagnosis names
    """
    # Extract features
    # X_df, feature_names = extract_cortical_features(df)# (260109) temporarily blocked for using all VA values
    X_df  = df.loc[:, 'Left_Cerebral_White_Matter':'ctx_rh_insula']
    feature_names = X_df.columns.tolist()
    X = X_df.values

    # Encode diagnoses
    y, dx_labels, _ = encode_diagnoses(df)

    # Optional standardization
    if standardize:
        print("Standardizing features...")
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    print(f"\nFinal data shape:")
    print(f"  X: {X.shape} (patients × cortical regions)")
    print(f"  y: {y.shape} (patients,)")
    print(f"  Features: {len(feature_names)}")
    print(f"  Diagnoses: {len(dx_labels)}")

    return X, y, feature_names, dx_labels


def train_test_split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split data into train/test sets with stratification by diagnosis.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Diagnosis labels
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Train and test splits
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"\nTrain/test split:")
    print(f"  Train: {X_train.shape[0]} patients")
    print(f"  Test:  {X_test.shape[0]} patients")

    return X_train, X_test, y_train, y_test


def get_diagnosis_name(y_encoded: int, dx_labels: List[str]):
    """
    Convert encoded diagnosis back to string label.

    Parameters
    ----------
    y_encoded : int
        Integer-encoded diagnosis
    dx_labels : List[str]
        Ordered list of diagnosis labels

    Returns
    -------
    str
        Diagnosis name
    """
    return dx_labels[y_encoded]


if __name__ == "__main__":
    # Example usage
    df = load_wsev_data('/home/coder/data/updated_WSEV/260108_wsev_final_df.csv')
    df = df[df['DX'] != 'HC']
    X, y, feature_names, dx_labels = prepare_slda_inputs(df)

    print(f"\nFirst 5 feature names: {feature_names[:5]}")
    print(f"Diagnosis labels: {dx_labels}")
    print(f"\nFirst patient features (first 5 regions):")
    print(X[0, :5])
    print(f"First patient diagnosis: {get_diagnosis_name(y[0], dx_labels)}")
