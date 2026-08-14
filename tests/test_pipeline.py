"""
Unit tests for feature engineering, preprocessing pipelines, and leak-free transformations.
"""

import numpy as np
import pandas as pd
import pytest

from src.heart_risk.data.loader import get_data_splits, load_dataset
from src.heart_risk.features.engineering import ClinicalFeatureEngineer
from src.heart_risk.features.pipeline import (
    create_preprocessing_pipeline,
    get_feature_names_after_preprocessing,
)


def test_clinical_feature_engineer():
    """Verify feature engineer adds domain variables accurately."""
    df = pd.DataFrame(
        {
            "age": [50, 60],
            "trestbps": [130, 150],
            "chol": [200, 240],
            "thalach": [170, 120],
            "oldpeak": [1.0, 2.0],
            "exang": [0, 1],
        }
    )

    fe = ClinicalFeatureEngineer()
    df_trans = fe.fit_transform(df)

    assert "pulse_pressure" in df_trans.columns
    assert "hr_reserve_ratio" in df_trans.columns
    assert "chol_age_ratio" in df_trans.columns
    assert "exercise_risk_index" in df_trans.columns

    # Test numerical calculations
    assert df_trans["pulse_pressure"].iloc[0] == 50.0  # 130 - 80
    assert df_trans["chol_age_ratio"].iloc[0] == 4.0  # 200 / 50
    assert df_trans["exercise_risk_index"].iloc[1] == 4.0  # 2.0 * (1 + 1)


def test_preprocessing_pipeline_no_nan_leak():
    """Verify that preprocessing pipeline scales and one-hot encodes without creating NaNs."""
    df = load_dataset()
    X_train, X_test, _, _ = get_data_splits(df)

    pipeline = create_preprocessing_pipeline()

    # Fit ONLY on train
    X_train_trans = pipeline.fit_transform(X_train)
    # Transform test
    X_test_trans = pipeline.transform(X_test)

    assert isinstance(X_train_trans, np.ndarray)
    assert isinstance(X_test_trans, np.ndarray)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()
    assert X_train_trans.shape[1] == X_test_trans.shape[1]
    assert X_train_trans.shape[1] > X_train.shape[1]  # Expanded via one-hot


def test_get_feature_names_after_preprocessing():
    """Verify feature names extraction from the pipeline."""
    df = load_dataset()
    X_train, _, _, _ = get_data_splits(df)

    pipeline = create_preprocessing_pipeline()
    pipeline.fit(X_train)

    feature_names = get_feature_names_after_preprocessing(pipeline, X_train)
    assert len(feature_names) > 0
    assert isinstance(feature_names[0], str)
