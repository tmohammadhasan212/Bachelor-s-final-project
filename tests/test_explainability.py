"""
Unit tests for Explainable AI (XAI) and Counterfactual Simulator.
"""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.heart_risk.data.loader import load_dataset
from src.heart_risk.explainability.counterfactual import CounterfactualSimulator
from src.heart_risk.explainability.shap_engine import ClinicalExplainer
from src.heart_risk.features.pipeline import create_preprocessing_pipeline


@pytest.fixture
def trained_pipeline():
    """Fixture providing a fitted model pipeline and background data."""
    df = load_dataset()
    X = df.drop("target", axis=1)
    y = df["target"]

    pipe = Pipeline(
        [
            ("preprocessor", create_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )
    pipe.fit(X, y)
    return pipe, X


def test_clinical_explainer(trained_pipeline):
    """Verify local and global explanations."""
    pipe, bg_df = trained_pipeline
    explainer = ClinicalExplainer(pipe, bg_df)

    patient_sample = bg_df.iloc[[0]]
    explanation = explainer.explain_instance(patient_sample, top_k=5)

    assert "base_risk" in explanation
    assert "predicted_risk" in explanation
    assert "top_contributions" in explanation
    assert len(explanation["top_contributions"]) == 5

    first_factor = explanation["top_contributions"][0]
    assert first_factor.direction in ("increases_risk", "decreases_risk")

    global_imp = explainer.get_global_feature_importance()
    assert isinstance(global_imp, dict)
    assert len(global_imp) > 0


def test_counterfactual_simulator(trained_pipeline):
    """Verify simulation of modifiable risk factors."""
    pipe, bg_df = trained_pipeline
    sim = CounterfactualSimulator(pipe)

    # Create a high-risk test patient
    high_risk_patient = pd.DataFrame(
        [
            {
                "age": 65,
                "sex": 1,
                "cp": 0,
                "trestbps": 165,
                "chol": 290,
                "fbs": 1,
                "restecg": 0,
                "thalach": 110,
                "exang": 1,
                "oldpeak": 2.5,
                "slope": 1,
                "ca": 2,
                "thal": 2,
            }
        ]
    )

    res = sim.simulate_interventions(high_risk_patient)

    assert "baseline_risk_pct" in res
    assert "counterfactual_risk_pct" in res
    assert "absolute_risk_reduction_pct" in res
    assert isinstance(res["baseline_risk_pct"], float)
    assert isinstance(res["counterfactual_risk_pct"], float)
    assert 0.0 <= res["counterfactual_risk_pct"] <= 100.0
    assert isinstance(res["recommendations"], list)
