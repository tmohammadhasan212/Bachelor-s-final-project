"""
Unit tests for FastAPI routes and inference endpoints.
"""

import pytest
from src.heart_risk.api.routes import (
    determine_risk_category,
    generate_clinical_recommendation,
    health_check,
    predict_patient_risk,
    explain_patient_risk,
    counterfactual_simulation,
)
from src.heart_risk.data.schemas import PatientInput


@pytest.fixture
def sample_patient():
    return PatientInput(
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


def test_determine_risk_category():
    assert determine_risk_category(0.95) == "Critical Risk"
    assert determine_risk_category(0.80) == "High Risk"
    assert determine_risk_category(0.60) == "Moderate Risk"
    assert determine_risk_category(0.20) == "Low Risk"


def test_generate_clinical_recommendation():
    rec = generate_clinical_recommendation(0.85, "High Risk")
    assert "Cardiology Referral" in rec or "Immediate" in rec

    rec_low = generate_clinical_recommendation(0.10, "Low Risk")
    assert "Low Immediate Risk" in rec_low


def test_health_check_endpoint():
    res = health_check()
    assert res["status"] == "healthy"
    assert res["service"] == "heart-risk-cds"


def test_predict_patient_risk_endpoint(sample_patient):
    res = predict_patient_risk(sample_patient)
    assert 0.0 <= res.risk_probability <= 1.0
    assert 0.0 <= res.risk_percentage <= 100.0
    assert res.risk_category in ("Critical Risk", "High Risk", "Moderate Risk", "Low Risk")
    assert res.risk_class in (0, 1)
    assert res.top_risk_factors is not None


def test_explain_patient_risk_endpoint(sample_patient):
    res = explain_patient_risk(sample_patient)
    assert "base_risk" in res
    assert "predicted_risk" in res
    assert "top_contributions" in res


def test_counterfactual_endpoint(sample_patient):
    res = counterfactual_simulation(sample_patient)
    assert "baseline_risk_pct" in res
    assert "counterfactual_risk_pct" in res
    assert "recommendations" in res
