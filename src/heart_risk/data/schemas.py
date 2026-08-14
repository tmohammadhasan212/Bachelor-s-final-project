"""
Pydantic data validation schemas for clinical patient records and predictions.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    """Clinical input features for a single patient assessment."""

    age: int = Field(
        ...,
        ge=18,
        le=120,
        description="Patient age in years (18-120)",
        examples=[58],
    )
    sex: int = Field(
        ...,
        ge=0,
        le=1,
        description="Biological sex (1 = male, 0 = female)",
        examples=[1],
    )
    cp: int = Field(
        ...,
        ge=0,
        le=3,
        description="Chest pain type: 0=Typical Angina, 1=Atypical Angina, 2=Non-anginal Pain, 3=Asymptomatic",
        examples=[2],
    )
    trestbps: int = Field(
        ...,
        ge=60,
        le=260,
        description="Resting blood pressure in mm Hg on admission to the hospital",
        examples=[130],
    )
    chol: int = Field(
        ...,
        ge=80,
        le=600,
        description="Serum cholesterol in mg/dl",
        examples=[240],
    )
    fbs: int = Field(
        ...,
        ge=0,
        le=1,
        description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
        examples=[0],
    )
    restecg: int = Field(
        ...,
        ge=0,
        le=2,
        description="Resting electrocardiographic results: 0=Normal, 1=ST-T wave abnormality, 2=Left ventricular hypertrophy",
        examples=[1],
    )
    thalach: int = Field(
        ...,
        ge=50,
        le=250,
        description="Maximum heart rate achieved during exercise stress test",
        examples=[160],
    )
    exang: int = Field(
        ...,
        ge=0,
        le=1,
        description="Exercise induced angina (1 = yes, 0 = no)",
        examples=[0],
    )
    oldpeak: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="ST depression induced by exercise relative to rest",
        examples=[1.2],
    )
    slope: int = Field(
        ...,
        ge=0,
        le=2,
        description="Slope of the peak exercise ST segment: 0=Upsloping, 1=Flat, 2=Downsloping",
        examples=[2],
    )
    ca: int = Field(
        ...,
        ge=0,
        le=4,
        description="Number of major vessels (0-4) colored by fluoroscopy",
        examples=[0],
    )
    thal: int = Field(
        ...,
        ge=0,
        le=3,
        description="Thalassemia status: 0=Null/Unknown, 1=Fixed defect, 2=Normal, 3=Reversible defect",
        examples=[2],
    )


class PatientRecord(PatientInput):
    """Complete patient record including target outcome."""

    target: Optional[int] = Field(
        default=None,
        ge=0,
        le=1,
        description="Ground truth cardiac risk (1 = high risk / heart attack presence, 0 = lower risk)",
    )


class FeatureContribution(BaseModel):
    """Individual feature contribution to risk prediction."""

    feature: str
    display_name: str
    value: float
    contribution: float
    direction: str  # "increases_risk" or "decreases_risk"


class PredictionResponse(BaseModel):
    """Clinical risk assessment response payload."""

    risk_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated probability of high cardiovascular risk",
    )
    risk_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Risk probability expressed as percentage",
    )
    risk_category: str = Field(
        ...,
        description="Clinical triage tier: Low, Moderate, High, or Critical Risk",
    )
    risk_class: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary decision threshold outcome (0 or 1)",
    )
    clinical_recommendation: str
    model_version: str
    top_risk_factors: Optional[List[FeatureContribution]] = None


class BatchPredictionResponse(BaseModel):
    """Response payload for multi-patient batch triage."""

    total_patients: int
    high_risk_count: int
    low_risk_count: int
    predictions: List[PredictionResponse]
