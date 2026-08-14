"""
Clinical and Statistical Performance Evaluation Metrics for Cardiology Classification.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def calculate_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted risk probabilities in [0, 1].
        n_bins: Number of probability bins.

    Returns:
        ECE float value between 0 and 1.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            bin_mask = bin_mask | (y_prob == bin_edges[i + 1])

        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute standard and clinical diagnostic metrics.

    Calculates:
    - Sensitivity / True Positive Rate / Recall
    - Specificity / True Negative Rate
    - Positive Predictive Value (PPV) / Precision
    - Negative Predictive Value (NPV)
    - Accuracy, Balanced Accuracy, F1-Score
    - ROC-AUC, PR-AUC, Brier Score, ECE (if y_prob is provided)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(ppv),
        "npv": float(npv),
        "f1_score": float(f1),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        # In case 2D array of class probabilities is passed
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]

        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        ece = calculate_expected_calibration_error(y_true, y_prob)

        metrics.update(
            {
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
                "brier_score": float(brier),
                "expected_calibration_error": float(ece),
            }
        )

    return metrics


def evaluate_classification_performance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate a trained scikit-learn model on test data."""
    y_pred = model.predict(X_test)
    y_prob = None

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_vals = model.decision_function(X_test)
        y_prob = 1.0 / (1.0 + np.exp(-df_vals))

    return compute_clinical_metrics(y_test, y_pred, y_prob)


def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Brier score."""
    return float(brier_score_loss(y_true, y_prob))
