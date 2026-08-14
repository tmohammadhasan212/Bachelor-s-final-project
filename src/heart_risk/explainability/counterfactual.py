"""
Counterfactual "What-If" Simulator and Actionable Lifestyle Recommendations.
Analyzes modifiable risk factors to calculate risk reduction trajectories for patient counseling.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.pipeline import Pipeline


class LifestyleRecommendation(BaseModel):
    """Actionable clinical recommendation with estimated risk reduction."""

    factor: str
    current_value: Any
    target_value: Any
    action_text: str
    risk_reduction_pct: float


class CounterfactualSimulator:
    """Simulates clinical interventions on modifiable cardiovascular risk factors."""

    MODIFIABLE_FACTORS = {
        "trestbps": {
            "name": "Resting Blood Pressure",
            "ideal": 118,
            "unit": "mmHg",
            "clinical_advice": "Hypertension control via DASH diet, sodium restriction, and ACE-inhibitors/antihypertensives.",
        },
        "chol": {
            "name": "Serum Cholesterol",
            "ideal": 180,
            "unit": "mg/dL",
            "clinical_advice": "Lipid-lowering therapy (statin therapy) combined with low-saturated fat Mediterranean diet.",
        },
        "thalach": {
            "name": "Max Exercise Heart Rate / Cardiorespiratory Fitness",
            "ideal_factor": 0.85,  # 85% of age predicted max
            "unit": "bpm",
            "clinical_advice": "Supervised aerobic exercise training (150 min/week moderate-intensity).",
        },
        "exang": {
            "name": "Exercise Induced Angina",
            "ideal": 0,
            "unit": "boolean",
            "clinical_advice": "Medical anti-anginal therapy (beta-blockers, nitrates) and revascularization evaluation.",
        },
        "oldpeak": {
            "name": "Exercise-Induced ST Depression",
            "ideal": 0.0,
            "unit": "mm",
            "clinical_advice": "Optimization of coronary perfusion and anti-ischemic pharmacological therapy.",
        },
    }

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def simulate_interventions(
        self,
        patient_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute counterfactual patient trajectories and actionable risk reduction."""
        current_prob = float(self.pipeline.predict_proba(patient_df)[0, 1])
        age = int(patient_df["age"].iloc[0])

        recommendations: List[LifestyleRecommendation] = []
        optimized_df = patient_df.copy()

        # 1. Blood pressure optimization
        current_bp = float(patient_df["trestbps"].iloc[0])
        if current_bp > 125:
            target_bp = 118.0
            test_df = patient_df.copy()
            test_df["trestbps"] = target_bp
            prob_new = float(self.pipeline.predict_proba(test_df)[0, 1])
            reduction = max((current_prob - prob_new) * 100.0, 0.0)
            if current_bp > 125:
                recommendations.append(
                    LifestyleRecommendation(
                        factor="Resting Blood Pressure",
                        current_value=f"{int(current_bp)} mmHg",
                        target_value=f"{int(target_bp)} mmHg",
                        action_text=f"Lower blood pressure by {int(current_bp - target_bp)} mmHg via DASH diet and medical management.",
                        risk_reduction_pct=round(reduction, 1),
                    )
                )
                optimized_df["trestbps"] = target_bp

        # 2. Cholesterol optimization
        current_chol = float(patient_df["chol"].iloc[0])
        if current_chol > 200:
            target_chol = 180.0
            test_df = patient_df.copy()
            test_df["chol"] = target_chol
            prob_new = float(self.pipeline.predict_proba(test_df)[0, 1])
            reduction = max((current_prob - prob_new) * 100.0, 0.0)

            recommendations.append(
                LifestyleRecommendation(
                    factor="Serum Cholesterol",
                    current_value=f"{int(current_chol)} mg/dL",
                    target_value=f"{int(target_chol)} mg/dL",
                    action_text=f"Reduce total cholesterol by {int(current_chol - target_chol)} mg/dL with statin/dietary therapy.",
                    risk_reduction_pct=round(reduction, 1),
                )
            )
            optimized_df["chol"] = target_chol

        # 3. Cardiorespiratory fitness (thalach)
        current_thalach = float(patient_df["thalach"].iloc[0])
        age_pred_max = max(220 - age, 100)
        target_thalach = min(int(age_pred_max * 0.85), 180)
        if current_thalach < target_thalach:
            test_df = patient_df.copy()
            test_df["thalach"] = target_thalach
            prob_new = float(self.pipeline.predict_proba(test_df)[0, 1])
            reduction = max((current_prob - prob_new) * 100.0, 0.0)

            recommendations.append(
                LifestyleRecommendation(
                    factor="Max Exercise Heart Rate",
                    current_value=f"{int(current_thalach)} bpm",
                    target_value=f"{int(target_thalach)} bpm",
                    action_text=f"Improve aerobic peak capacity from {int(current_thalach)} to {target_thalach} bpm through structured cardio.",
                    risk_reduction_pct=round(reduction, 1),
                )
            )
            optimized_df["thalach"] = target_thalach

        # 4. Ischemia resolution (oldpeak & exang)
        current_oldpeak = float(patient_df["oldpeak"].iloc[0])
        current_exang = int(patient_df["exang"].iloc[0])
        if current_oldpeak > 0.5 or current_exang == 1:
            test_df = patient_df.copy()
            test_df["oldpeak"] = 0.0
            test_df["exang"] = 0
            prob_new = float(self.pipeline.predict_proba(test_df)[0, 1])
            reduction = max((current_prob - prob_new) * 100.0, 0.0)

            recommendations.append(
                LifestyleRecommendation(
                    factor="Exercise Ischemia & Angina",
                    current_value=f"Oldpeak: {current_oldpeak}, Angina: {'Yes' if current_exang==1 else 'No'}",
                    target_value="Oldpeak: 0.0, Angina: No",
                    action_text="Resolve exercise ischemia through coronary intervention and anti-anginal pharmacotherapy.",
                    risk_reduction_pct=round(reduction, 1),
                )
            )
            optimized_df["oldpeak"] = 0.0
            optimized_df["exang"] = 0

        # Full combined optimization
        full_optimized_prob = float(
            self.pipeline.predict_proba(optimized_df)[0, 1]
        )
        total_arr = (current_prob - full_optimized_prob) * 100.0
        total_rrr = (
            (total_arr / (current_prob * 100.0)) * 100.0
            if current_prob > 0
            else 0.0
        )

        return {
            "baseline_risk_pct": round(current_prob * 100.0, 1),
            "counterfactual_risk_pct": round(full_optimized_prob * 100.0, 1),
            "absolute_risk_reduction_pct": round(max(total_arr, 0.0), 1),
            "relative_risk_reduction_pct": round(max(total_rrr, 0.0), 1),
            "recommendations": recommendations,
        }
