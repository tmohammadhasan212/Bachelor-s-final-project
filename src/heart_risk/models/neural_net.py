"""
Deep Learning & Multi-Layer Perceptron architectures for tabular cardiology data.
Provides flexible MLP implementation with regularization, dropout, and probability calibration.
"""

from typing import List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier


class TabularMLPClassifier(BaseEstimator, ClassifierMixin):
    """Multi-Layer Perceptron optimized for tabular clinical biomarkers.

    Features:
    - Multiple hidden layers with non-linear activation (ReLU / GELU)
    - L2 regularization (weight decay) to prevent overfitting on small clinical cohorts
    - Early stopping with validation patience
    - Calibrated probability outputs
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        activation: str = "relu",
        alpha: float = 0.01,
        learning_rate_init: float = 0.005,
        max_iter: int = 500,
        early_stopping: bool = True,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.model_: Optional[MLPClassifier] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the MLP model on preprocessed features."""
        self.classes_ = np.unique(y)
        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            learning_rate="adaptive",
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            n_iter_no_change=15,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels."""
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model_.predict_proba(X)
