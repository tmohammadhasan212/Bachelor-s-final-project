"""
Clinical Utility and Decision Curve Analysis (DCA) for Cardiovascular Risk Models.
Quantifies Net Clinical Benefit across decision thresholds (Vickers & Elkin, 2006).
"""

from typing import Dict, List, Tuple
import numpy as np


def calculate_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> float:
    """Calculate Net Benefit for a specific clinical probability threshold.

    Formula:
        Net Benefit = (TP / N) - (FP / N) * (threshold / (1 - threshold))
    """
    if threshold <= 0.0 or threshold >= 1.0:
        return 0.0

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    n_total = len(y_true)
    if n_total == 0:
        return 0.0

    y_pred_thresh = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred_thresh == 1) & (y_true == 1))
    fp = np.sum((y_pred_thresh == 1) & (y_true == 0))

    weight = threshold / (1.0 - threshold)
    net_benefit = (tp / n_total) - (fp / n_total) * weight

    return float(net_benefit)


def compute_decision_curve_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.05, 0.90, 50),
) -> Dict[str, List[float]]:
    """Compute Decision Curve Analysis comparing Model vs Treat-All vs Treat-None.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted risk probabilities in [0, 1].
        thresholds: Array of decision threshold probabilities.

    Returns:
        Dictionary containing threshold values and net benefits for model, treat-all, and treat-none.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    n_total = len(y_true)
    prevalence = float(np.mean(y_true))

    model_net_benefits = []
    treat_all_net_benefits = []
    treat_none_net_benefits = [0.0] * len(thresholds)

    for thresh in thresholds:
        # Model Net Benefit
        nb_model = calculate_net_benefit(y_true, y_prob, thresh)
        model_net_benefits.append(nb_model)

        # Treat All Net Benefit = prevalence - (1 - prevalence) * (thresh / (1 - thresh))
        weight = thresh / (1.0 - thresh)
        nb_treat_all = prevalence - (1.0 - prevalence) * weight
        treat_all_net_benefits.append(float(nb_treat_all))

    return {
        "thresholds": [float(t) for t in thresholds],
        "model_net_benefit": model_net_benefits,
        "treat_all_net_benefit": treat_all_net_benefits,
        "treat_none_net_benefit": treat_none_net_benefits,
        "prevalence": prevalence,
    }
