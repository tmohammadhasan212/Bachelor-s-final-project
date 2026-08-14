"""
Clinical Feature Engineering module for cardiology risk models.
Provides domain-specific derived features using Scikit-Learn Transformer API.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Generates clinically relevant derived features for cardiovascular risk assessment.

    Derived Features:
    - pulse_pressure: trestbps - 80 (estimate of systolic load above resting standard)
    - hr_reserve_ratio: thalach / (220 - age) (fraction of age-predicted max HR achieved)
    - chol_age_ratio: chol / age (cholesterol accumulation rate relative to age)
    - exercise_risk_index: oldpeak * (exang + 1) (compound ST depression and angina marker)
    """

    def __init__(
        self,
        add_pulse_pressure: bool = True,
        add_hr_reserve_ratio: bool = True,
        add_chol_age_ratio: bool = True,
        add_exercise_risk_index: bool = True,
    ):
        self.add_pulse_pressure = add_pulse_pressure
        self.add_hr_reserve_ratio = add_hr_reserve_ratio
        self.add_chol_age_ratio = add_chol_age_ratio
        self.add_exercise_risk_index = add_exercise_risk_index
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit method (stateless transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe by adding derived clinical features."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        if self.add_pulse_pressure and "trestbps" in X_out.columns:
            # Systolic pulse pressure approximation
            X_out["pulse_pressure"] = X_out["trestbps"] - 80.0

        if (
            self.add_hr_reserve_ratio
            and "thalach" in X_out.columns
            and "age" in X_out.columns
        ):
            # Predicted Max HR = 220 - age; ratio achieved
            max_hr_pred = np.maximum(220.0 - X_out["age"], 50.0)
            X_out["hr_reserve_ratio"] = np.clip(
                X_out["thalach"] / max_hr_pred, 0.1, 2.0
            )

        if (
            self.add_chol_age_ratio
            and "chol" in X_out.columns
            and "age" in X_out.columns
        ):
            # Ratio of cholesterol to patient age
            age_safe = np.maximum(X_out["age"], 18.0)
            X_out["chol_age_ratio"] = X_out["chol"] / age_safe

        if (
            self.add_exercise_risk_index
            and "oldpeak" in X_out.columns
            and "exang" in X_out.columns
        ):
            # Compound stress-induced ischemia score
            X_out["exercise_risk_index"] = X_out["oldpeak"] * (
                X_out["exang"] + 1.0
            )

        self.feature_names_out_ = list(X_out.columns)
        return X_out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return generated feature names."""
        return np.array(self.feature_names_out_)
