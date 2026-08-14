"""
Explainable AI (XAI), SHAP attributions, and Counterfactual Simulator.
"""

from src.heart_risk.explainability.shap_engine import ClinicalExplainer
from src.heart_risk.explainability.counterfactual import (
    CounterfactualSimulator,
    LifestyleRecommendation,
)

__all__ = [
    "ClinicalExplainer",
    "CounterfactualSimulator",
    "LifestyleRecommendation",
]
