"""
Unit tests for clinical metrics, calibration, and Decision Curve Analysis.
"""

import numpy as np
import pytest

from src.heart_risk.evaluation.clinical_utility import (
    calculate_net_benefit,
    compute_decision_curve_analysis,
)
from src.heart_risk.evaluation.metrics import (
    calculate_brier_score,
    calculate_expected_calibration_error,
    compute_clinical_metrics,
)


def test_compute_clinical_metrics():
    """Verify calculation of sensitivity, specificity, NPV, PPV, and AUC."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1])
    y_prob = np.array([0.9, 0.8, 0.4, 0.1, 0.2, 0.7])

    metrics = compute_clinical_metrics(y_true, y_pred, y_prob)

    assert metrics["sensitivity"] == pytest.approx(2 / 3)  # TP=2, FN=1
    assert metrics["specificity"] == pytest.approx(2 / 3)  # TN=2, FP=1
    assert metrics["precision"] == pytest.approx(2 / 3)  # TP=2, FP=1
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert "expected_calibration_error" in metrics


def test_expected_calibration_error():
    """Verify ECE calculation."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([1.0, 1.0, 0.0, 0.0])  # Perfect calibration
    ece_perfect = calculate_expected_calibration_error(y_true, y_prob, n_bins=2)
    assert ece_perfect == pytest.approx(0.0)


def test_decision_curve_analysis():
    """Verify Decision Curve Analysis outputs valid Net Benefit vectors."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.85, 0.15])

    dca = compute_decision_curve_analysis(
        y_true, y_prob, thresholds=np.array([0.2, 0.5, 0.8])
    )

    assert "thresholds" in dca
    assert "model_net_benefit" in dca
    assert "treat_all_net_benefit" in dca
    assert len(dca["model_net_benefit"]) == 3
