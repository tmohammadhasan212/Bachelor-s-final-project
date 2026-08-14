"""
Unit tests for model registry, TabularMLP, cross-validation, and training harness.
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.heart_risk.config import Config
from src.heart_risk.data.loader import get_data_splits, load_dataset
from src.heart_risk.models.neural_net import TabularMLPClassifier
from src.heart_risk.models.registry import (
    get_hyperparameter_grids,
    get_model_instances,
)
from src.heart_risk.models.train import ModelTrainer


def test_model_registry_instantiation():
    """Verify all defined models instantiate correctly."""
    models = get_model_instances(random_seed=42)
    expected = [
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "adaboost",
        "svc",
        "neural_network_mlp",
        "stacking_ensemble",
        "voting_ensemble",
    ]
    for name in expected:
        assert name in models
        assert hasattr(models[name], "fit")


def test_hyperparameter_grids():
    """Verify hyperparameter search grids are valid dictionaries."""
    grids = get_hyperparameter_grids()
    assert "logistic_regression" in grids
    assert "random_forest" in grids
    assert len(grids["random_forest"]) >= 3


def test_tabular_mlp_classifier():
    """Verify TabularMLP fits and outputs calibrated probabilities."""
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, size=50)

    mlp = TabularMLPClassifier(hidden_layer_sizes=(16, 8), max_iter=50)
    mlp.fit(X, y)

    preds = mlp.predict(X)
    probs = mlp.predict_proba(X)

    assert len(preds) == 50
    assert probs.shape == (50, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_trainer_evaluate_cv_fast():
    """Verify cross-validation harness runs and outputs proper statistics."""
    df = load_dataset().sample(60, random_state=42)  # Fast subset
    X = df.drop("target", axis=1)
    y = df["target"]

    cfg = Config()
    trainer = ModelTrainer(cfg)
    models = get_model_instances(42)

    res = trainer.evaluate_model_cv(
        model_name="logistic_regression",
        model_instance=models["logistic_regression"],
        X=X,
        y=y,
        n_splits=2,
        n_repeats=1,
    )

    assert "accuracy_mean" in res
    assert "roc_auc_mean" in res
    assert "sensitivity_mean" in res
    assert "brier_score_mean" in res
    assert 0.0 <= res["accuracy_mean"] <= 1.0
