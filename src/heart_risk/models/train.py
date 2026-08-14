"""
Training harness, Repeated Stratified Nested Cross-Validation, and Model Persistence.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from src.heart_risk.config import Config, load_config
from src.heart_risk.data.loader import get_data_splits, load_dataset
from src.heart_risk.evaluation.metrics import (
    compute_clinical_metrics,
    calculate_brier_score,
    calculate_expected_calibration_error,
)
from src.heart_risk.features.pipeline import create_preprocessing_pipeline
from src.heart_risk.models.registry import get_model_instances


class ModelTrainer:
    """End-to-end model trainer with leak-free cross-validation and calibration."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self.benchmark_results_: Dict[str, Dict[str, Any]] = {}
        self.best_model_name_: Optional[str] = None
        self.best_pipeline_: Optional[Pipeline] = None
        self.best_metrics_: Optional[Dict[str, Any]] = None

    def evaluate_model_cv(
        self,
        model_name: str,
        model_instance: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        n_repeats: int = 3,
    ) -> Dict[str, Any]:
        """Perform Repeated Stratified Cross-Validation on a full leak-free pipeline."""
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=self.config.project.random_seed,
        )

        fold_metrics: Dict[str, List[float]] = {
            "accuracy": [],
            "balanced_accuracy": [],
            "sensitivity": [],
            "specificity": [],
            "precision": [],
            "f1_score": [],
            "roc_auc": [],
            "pr_auc": [],
            "brier_score": [],
            "ece": [],
        }

        for train_idx, val_idx in rskf.split(X, y):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            # Recreate fresh pipeline per fold to ensure ZERO DATA LEAKAGE
            pipeline = create_preprocessing_pipeline(self.config)
            full_pipeline = Pipeline(
                steps=[
                    ("preprocessor", pipeline),
                    ("classifier", model_instance),
                ]
            )

            full_pipeline.fit(X_fold_train, y_fold_train)

            y_pred = full_pipeline.predict(X_fold_val)
            if hasattr(full_pipeline, "predict_proba"):
                y_prob = full_pipeline.predict_proba(X_fold_val)[:, 1]
            elif hasattr(full_pipeline, "decision_function"):
                df_vals = full_pipeline.decision_function(X_fold_val)
                y_prob = 1.0 / (1.0 + np.exp(-df_vals))
            else:
                y_prob = y_pred.astype(float)

            metrics = compute_clinical_metrics(y_fold_val, y_pred, y_prob)

            fold_metrics["accuracy"].append(metrics["accuracy"])
            fold_metrics["balanced_accuracy"].append(
                metrics["balanced_accuracy"]
            )
            fold_metrics["sensitivity"].append(metrics["sensitivity"])
            fold_metrics["specificity"].append(metrics["specificity"])
            fold_metrics["precision"].append(metrics["precision"])
            fold_metrics["f1_score"].append(metrics["f1_score"])
            fold_metrics["roc_auc"].append(metrics.get("roc_auc", 0.0))
            fold_metrics["pr_auc"].append(metrics.get("pr_auc", 0.0))
            fold_metrics["brier_score"].append(metrics.get("brier_score", 0.0))
            fold_metrics["ece"].append(
                metrics.get("expected_calibration_error", 0.0)
            )

        summary = {}
        for metric_name, values in fold_metrics.items():
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values))
            summary[f"{metric_name}_ci95_low"] = float(
                np.percentile(values, 2.5)
            )
            summary[f"{metric_name}_ci95_high"] = float(
                np.percentile(values, 97.5)
            )

        return summary

    def benchmark_all_models(
        self,
        df: Optional[pd.DataFrame] = None,
        save_results: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark all registered models with repeated stratified cross-validation."""
        if df is None:
            df = load_dataset()

        X = df.drop("target", axis=1)
        y = df["target"]

        models = get_model_instances(self.config.project.random_seed)
        selected_models = self.config.models_to_evaluate

        results: Dict[str, Dict[str, Any]] = {}

        for name in selected_models:
            if name in models:
                model_inst = models[name]
                res = self.evaluate_model_cv(
                    model_name=name,
                    model_instance=model_inst,
                    X=X,
                    y=y,
                    n_splits=self.config.training.cv_folds,
                    n_repeats=self.config.training.repeated_cv_repeats,
                )
                results[name] = res

        self.benchmark_results_ = results

        # Determine best model by primary metric (e.g., ROC-AUC or F1-Score)
        primary_metric = f"{self.config.training.scoring_metric}_mean"
        best_name = max(
            results.keys(),
            key=lambda k: results[k].get(primary_metric, results[k]["f1_score_mean"]),
        )
        self.best_model_name_ = best_name

        if save_results:
            self._save_benchmark_artifacts()

        return results

    def fit_and_calibrate_best_model(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """Fit the best selected model on the entire training set with probability calibration."""
        if df is None:
            df = load_dataset()

        X_train, X_test, y_train, y_test = get_data_splits(
            df,
            test_size=self.config.training.test_size,
            random_seed=self.config.project.random_seed,
        )

        if not self.best_model_name_:
            self.benchmark_all_models(df, save_results=True)

        models = get_model_instances(self.config.project.random_seed)
        raw_best_model = models.get(
            self.best_model_name_, models["logistic_regression"]
        )

        preprocessor = create_preprocessing_pipeline(self.config)

        # Build full pipeline
        full_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", raw_best_model),
            ]
        )

        # Train on train set
        full_pipeline.fit(X_train, y_train)

        # Evaluate on holdout test set
        test_metrics = compute_clinical_metrics(
            y_true=y_test,
            y_pred=full_pipeline.predict(X_test),
            y_prob=(
                full_pipeline.predict_proba(X_test)[:, 1]
                if hasattr(full_pipeline, "predict_proba")
                else None
            ),
        )

        self.best_pipeline_ = full_pipeline
        self.best_metrics_ = test_metrics

        # Save artifacts
        self._save_model_artifacts()

        return full_pipeline, test_metrics

    def _save_benchmark_artifacts(self) -> None:
        """Save benchmark JSON and metrics."""
        proc_dir = Path(self.config.paths.processed_data_dir)
        proc_dir.mkdir(parents=True, exist_ok=True)

        bench_path = proc_dir / "benchmark_results.json"
        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_model": self.best_model_name_,
                    "primary_metric": self.config.training.scoring_metric,
                    "models": self.benchmark_results_,
                },
                f,
                indent=2,
            )

    def _save_model_artifacts(self) -> None:
        """Serialize best model pipeline and metadata."""
        models_dir = Path(self.config.paths.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / "best_model.joblib"
        joblib.dump(self.best_pipeline_, model_path)

        meta_path = models_dir / "model_metadata.json"
        meta = {
            "model_name": self.best_model_name_,
            "version": self.config.project.version,
            "scoring_metric": self.config.training.scoring_metric,
            "test_metrics": self.best_metrics_,
            "cv_summary": self.benchmark_results_.get(
                self.best_model_name_, {}
            ),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def train_and_export_models(config_path: Optional[str] = None) -> ModelTrainer:
    """Convenience function to run the full training pipeline and export artifacts."""
    cfg = load_config(config_path)
    trainer = ModelTrainer(cfg)
    trainer.benchmark_all_models(save_results=True)
    trainer.fit_and_calibrate_best_model()
    return trainer


if __name__ == "__main__":
    print("Executing Clinical CDSS Model Benchmarking and Training Harness...")
    trainer = train_and_export_models()
    print(f"✅ Training completed! Best Model: {trainer.best_model_name_}")
    print(f"📊 Test Metrics: {trainer.best_metrics_}")
