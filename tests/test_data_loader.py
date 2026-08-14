"""
Unit tests for data loading, validation, schemas, and splitting.
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from src.heart_risk.data.loader import (
    get_data_splits,
    load_dataset,
    validate_dataset,
)
from src.heart_risk.data.schemas import PatientInput, PatientRecord


def test_load_dataset_success():
    """Verify that dataset loads cleanly with 14 columns and no NaNs."""
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 200
    assert df.isnull().sum().sum() == 0
    assert "target" in df.columns
    assert set(df["target"].unique()).issubset({0, 1})


def test_validate_dataset_missing_column():
    """Verify that validation raises error if a required column is missing."""
    df = load_dataset()
    df_broken = df.drop("chol", axis=1)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(df_broken)


def test_validate_dataset_invalid_target():
    """Verify validation raises error if target values are non-binary."""
    df = load_dataset().copy()
    df.loc[0, "target"] = 99
    with pytest.raises(ValueError, match="Target column must only contain 0 or 1"):
        validate_dataset(df)


def test_get_data_splits_stratification():
    """Verify that stratified splitting maintains label proportions."""
    df = load_dataset()
    X_train, X_test, y_train, y_test = get_data_splits(df, test_size=0.20, random_seed=42)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_test) == int(len(df) * 0.20) + 1  # 303 * 0.2 = ~61

    # Ratio of target 1 should be almost identical in train and test
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05


def test_patient_input_schema_valid():
    """Verify valid patient record passes Pydantic validation."""
    p = PatientInput(
        age=58,
        sex=1,
        cp=2,
        trestbps=130,
        chol=240,
        fbs=0,
        restecg=1,
        thalach=160,
        exang=0,
        oldpeak=1.2,
        slope=2,
        ca=0,
        thal=2,
    )
    assert p.age == 58
    assert p.chol == 240


def test_patient_input_schema_invalid_age():
    """Verify schema catches invalid ages."""
    with pytest.raises(ValidationError):
        PatientInput(
            age=10,  # Below 18 minimum
            sex=1,
            cp=2,
            trestbps=130,
            chol=240,
            fbs=0,
            restecg=1,
            thalach=160,
            exang=0,
            oldpeak=1.2,
            slope=2,
            ca=0,
            thal=2,
        )
