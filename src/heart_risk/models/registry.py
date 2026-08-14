"""
Model Registry and Factory for Clinical Machine Learning Classifiers.
Provides standardized construction of base models, ensembles, and hyperparameter grids.
"""

from typing import Any, Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def get_model_instances(random_seed: int = 42) -> Dict[str, Any]:
    """Create initialized instances of all supported classification algorithms."""
    lr = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_seed,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_seed,
    )

    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=4,
        class_weight="balanced",
        random_state=random_seed,
    )

    gb = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=3,
        subsample=0.85,
        random_state=random_seed,
    )

    ada = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=random_seed,
    )

    svc = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=random_seed,
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=0.01,
        learning_rate="adaptive",
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=random_seed,
    )

    # Base estimators for stacking and voting
    base_estimators = [
        ("rf", rf),
        ("gb", gb),
        ("lr", lr),
        ("svc", svc),
        ("et", et),
    ]

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=0.5, random_state=random_seed),
        cv=5,
        n_jobs=-1,
    )

    voting = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("gb", gb),
            ("lr", lr),
            ("svc", svc),
        ],
        voting="soft",
        n_jobs=-1,
    )

    return {
        "logistic_regression": lr,
        "random_forest": rf,
        "extra_trees": et,
        "gradient_boosting": gb,
        "adaboost": ada,
        "svc": svc,
        "neural_network_mlp": mlp,
        "stacking_ensemble": stacking,
        "voting_ensemble": voting,
    }


def get_hyperparameter_grids() -> Dict[str, Dict[str, List[Any]]]:
    """Return hyperparameter search spaces for systematic tuning."""
    return {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
        },
        "random_forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_split": [2, 4, 8],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "extra_trees": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [4, 6, 8],
            "model__min_samples_split": [2, 4],
        },
        "gradient_boosting": {
            "model__n_estimators": [80, 120, 200],
            "model__learning_rate": [0.03, 0.08, 0.15],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 0.9, 1.0],
        },
        "adaboost": {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.05, 0.1, 0.5, 1.0],
        },
        "svc": {
            "model__C": [0.1, 0.5, 1.0, 5.0, 10.0],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto", 0.01, 0.1],
        },
        "neural_network_mlp": {
            "model__hidden_layer_sizes": [(32,), (64, 32), (64, 32, 16)],
            "model__alpha": [0.0001, 0.001, 0.01, 0.1],
            "model__learning_rate_init": [0.001, 0.005, 0.01],
        },
    }
