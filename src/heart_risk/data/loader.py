"""
Data loading, validation, and stratified splitting utilities.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.heart_risk.config import Config, load_config


EXPECTED_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def find_dataset_path(custom_path: Optional[str] = None) -> Path:
    """Resolve the dataset path across local, project, and notebook working directories."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    search_candidates = [
        Path("data/raw/heart.csv"),
        Path("heart.csv"),
        Path("../data/raw/heart.csv"),
        Path("../heart.csv"),
        Path("../../data/raw/heart.csv"),
        Path("../../heart.csv"),
    ]

    for candidate in search_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Heart disease dataset not found. Looked in: {[str(p) for p in search_candidates]}"
    )


def load_dataset(
    filepath: Optional[str] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load and optionally validate the heart disease dataset.

    Args:
        filepath: Path to the CSV file. If None, resolves default locations.
        validate: Whether to perform schema and range checks.

    Returns:
        Validated pandas DataFrame.
    """
    resolved_path = find_dataset_path(filepath)
    df = pd.read_csv(resolved_path)

    if validate:
        validate_dataset(df)

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate DataFrame columns, missing values, and data types."""
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Check for missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        total_nulls = int(null_counts.sum())
        raise ValueError(
            f"Dataset contains {total_nulls} missing values across columns: {null_counts[null_counts > 0].to_dict()}"
        )

    # Validate target column values
    unique_targets = set(df["target"].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target column must only contain 0 or 1, found: {unique_targets}"
        )


def get_data_splits(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.20,
    random_seed: int = 42,
    config: Optional[Config] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into stratified train and test sets.

    Args:
        df: Input DataFrame. If None, loads using default loader.
        test_size: Proportion of test set (e.g. 0.20).
        random_seed: Random seed for reproducibility.
        config: Optional Config instance.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if config is not None:
        test_size = config.training.test_size
        random_seed = config.project.random_seed

    if df is None:
        df = load_dataset()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
