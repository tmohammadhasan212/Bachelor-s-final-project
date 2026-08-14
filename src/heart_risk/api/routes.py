from typing import Any, Dict, List, Optional
import joblib
import pandas as pd

from src.heart_risk.config import load_config
from src.heart_risk.data.loader import find_dataset_path, load_dataset
from src.heart_risk.data.schemas import (
    BatchPredictionResponse,
    PatientInput,
    PredictionResponse,
    FeatureContribution,
)
from src.heart_risk.explainability.counterfactual import CounterfactualSimulator
from src.heart_risk.explainability.shap_engine import ClinicalExplainer

try:
    from fastapi import APIRouter, HTTPException, status

    FASTAPI_AVAILABLE = True
    router = APIRouter()
except ImportError:
    FASTAPI_AVAILABLE = False

    class DummyRouter:
        def get(self, *args, **kwargs):
            def decorator(f):
                return f

            return decorator

        def post(self, *args, **kwargs):
            def decorator(f):
                return f

            return decorator

    router = DummyRouter()

    class DummyStatus:
        HTTP_200_OK = 200
        HTTP_400_BAD_REQUEST = 400

    status = DummyStatus()

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

cfg = load_config()

# Global cached instances
_MODEL = None
_BACKGROUND_DF = None
_EXPLAINER = None
_SIMULATOR = None


def get_model():
    """Load serialized model pipeline with caching."""
    global _MODEL
    if _MODEL is None:
        model_path = find_dataset_path()  # ensure working dir
        saved_path = cfg.paths.models_dir + "/best_model.joblib"
        try:
            _MODEL = joblib.load(saved_path)
        except Exception:
            # If not yet trained, train on demand
            from src.heart_risk.models.train import train_and_export_models

            trainer = train_and_export_models()
            _MODEL = trainer.best_pipeline_
    return _MODEL


def get_background_data():
    """Load reference background dataset for explanations."""
    global _BACKGROUND_DF
    if _BACKGROUND_DF is None:
        _BACKGROUND_DF = load_dataset()
    return _BACKGROUND_DF


def get_explainer():
    """Get singleton explainer instance."""
    global _EXPLAINER
    if _EXPLAINER is None:
        model = get_model()
        bg = get_background_data().drop("target", axis=1)
        _EXPLAINER = ClinicalExplainer(model, bg)
    return _EXPLAINER


def get_simulator():
    """Get singleton counterfactual simulator."""
    global _SIMULATOR
    if _SIMULATOR is None:
        model = get_model()
        _SIMULATOR = CounterfactualSimulator(model)
    return _SIMULATOR


def determine_risk_category(prob: float) -> str:
    """Classify probability into clinical triage tiers."""
    if prob >= cfg.clinical_thresholds.critical_risk:
        return "Critical Risk"
    elif prob >= cfg.clinical_thresholds.high_risk:
        return "High Risk"
    elif prob >= cfg.clinical_thresholds.moderate_risk:
        return "Moderate Risk"
    return "Low Risk"


def generate_clinical_recommendation(prob: float, cat: str) -> str:
    """Generate concise medical triage advice."""
    if cat in ("Critical Risk", "High Risk"):
        return (
            "Immediate Cardiology Referral Recommended. Schedule urgent 12-lead ECG, "
            "cardiac enzyme panel (Troponin), echocardiogram, and evaluate for invasive coronary angiography."
        )
    elif cat == "Moderate Risk":
        return (
            "Intermediate Cardiovascular Risk. Recommend exercise stress testing, continuous ambulatory ECG "
            "monitoring, and aggressive guideline-directed risk factor modification (lipid & BP control)."
        )
    return (
        "Low Immediate Risk. Maintain regular primary care follow-up, healthy lifestyle counseling, "
        "and annual cardiovascular wellness checkups."
    )


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """Health check endpoint confirming model readiness."""
    model = get_model()
    return {
        "status": "healthy",
        "service": "heart-risk-cds",
        "version": cfg.project.version,
        "model_loaded": model is not None,
    }


@router.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
def predict_patient_risk(patient: PatientInput) -> PredictionResponse:
    """Predict calibrated cardiovascular event risk for a single patient."""
    model = get_model()
    patient_df = pd.DataFrame([patient.model_dump()])

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(patient_df)[0, 1])
    else:
        prob = float(model.predict(patient_df)[0])

    cat = determine_risk_category(prob)
    rec = generate_clinical_recommendation(prob, cat)
    risk_class = 1 if prob >= 0.50 else 0

    # Top risk factors attribution
    explainer = get_explainer()
    explanation = explainer.explain_instance(patient_df, top_k=4)

    return PredictionResponse(
        risk_probability=round(prob, 4),
        risk_percentage=round(prob * 100.0, 1),
        risk_category=cat,
        risk_class=risk_class,
        clinical_recommendation=rec,
        model_version=cfg.project.version,
        top_risk_factors=explanation.get("top_contributions"),
    )


@router.post("/api/v1/explain", status_code=status.HTTP_200_OK)
def explain_patient_risk(patient: PatientInput) -> Dict[str, Any]:
    """Compute complete feature contribution breakdown for a patient."""
    explainer = get_explainer()
    patient_df = pd.DataFrame([patient.model_dump()])
    return explainer.explain_instance(patient_df, top_k=10)


@router.post("/api/v1/counterfactual", status_code=status.HTTP_200_OK)
def counterfactual_simulation(patient: PatientInput) -> Dict[str, Any]:
    """Simulate modifiable lifestyle and clinical parameter interventions."""
    simulator = get_simulator()
    patient_df = pd.DataFrame([patient.model_dump()])
    return simulator.simulate_interventions(patient_df)


@router.post(
    "/api/v1/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
)
def batch_predict(patients: List[PatientInput]) -> BatchPredictionResponse:
    """Perform batch risk scoring across multiple patient records."""
    if not patients:
        raise HTTPException(
            status_code=400, detail="Empty patient list provided."
        )

    predictions: List[PredictionResponse] = []
    high_count = 0
    low_count = 0

    for p in patients:
        res = predict_patient_risk(p)
        if res.risk_class == 1:
            high_count += 1
        else:
            low_count += 1
        predictions.append(res)

    return BatchPredictionResponse(
        total_patients=len(patients),
        high_risk_count=high_count,
        low_risk_count=low_count,
        predictions=predictions,
    )
