"""
Configuration module for the Heart Risk CDSS.
Loads and validates settings from YAML configuration files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ProjectConfig:
    name: str = "heart-risk-cds"
    version: str = "1.0.0"
    random_seed: int = 42


@dataclass
class PathsConfig:
    raw_data: str = "data/raw/heart.csv"
    processed_data_dir: str = "data/processed"
    models_dir: str = "models/saved"
    figures_dir: str = "reports/figures"


@dataclass
class FeatureEngineeringConfig:
    enabled: bool = True
    add_pulse_pressure: bool = True
    add_hr_reserve_ratio: bool = True
    add_chol_age_ratio: bool = True
    add_exercise_risk_index: bool = True


@dataclass
class DataConfig:
    target_column: str = "target"
    numerical_features: List[str] = field(
        default_factory=lambda: ["age", "trestbps", "chol", "thalach", "oldpeak"]
    )
    categorical_features: List[str] = field(
        default_factory=lambda: [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "ca",
            "thal",
        ]
    )
    feature_engineering: FeatureEngineeringConfig = field(
        default_factory=FeatureEngineeringConfig
    )


@dataclass
class ClinicalThresholdsConfig:
    low_risk: float = 0.25
    moderate_risk: float = 0.50
    high_risk: float = 0.75
    critical_risk: float = 0.90


@dataclass
class TrainingConfig:
    test_size: float = 0.20
    cv_folds: int = 5
    repeated_cv_repeats: int = 3
    scoring_metric: str = "roc_auc"
    calibration_method: str = "sigmoid"


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    clinical_thresholds: ClinicalThresholdsConfig = field(
        default_factory=ClinicalThresholdsConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models_to_evaluate: List[str] = field(
        default_factory=lambda: [
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "adaboost",
            "svc",
            "stacking_ensemble",
            "voting_ensemble",
            "neural_network_mlp",
        ]
    )


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from a YAML file or return defaults.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config instance populated with settings.
    """
    if config_path is None:
        default_paths = [
            Path("configs/config.yaml"),
            Path("../configs/config.yaml"),
            Path("../../configs/config.yaml"),
        ]
        for p in default_paths:
            if p.exists():
                config_path = str(p)
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        fe_data = raw.get("data", {}).get("feature_engineering", {})
        fe_config = FeatureEngineeringConfig(
            enabled=fe_data.get("enabled", True),
            add_pulse_pressure=fe_data.get("add_pulse_pressure", True),
            add_hr_reserve_ratio=fe_data.get("add_hr_reserve_ratio", True),
            add_chol_age_ratio=fe_data.get("add_chol_age_ratio", True),
            add_exercise_risk_index=fe_data.get("add_exercise_risk_index", True),
        )

        data_section = raw.get("data", {})
        data_config = DataConfig(
            target_column=data_section.get("target_column", "target"),
            numerical_features=data_section.get(
                "numerical_features",
                ["age", "trestbps", "chol", "thalach", "oldpeak"],
            ),
            categorical_features=data_section.get(
                "categorical_features",
                ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
            ),
            feature_engineering=fe_config,
        )

        project_section = raw.get("project", {})
        project_config = ProjectConfig(
            name=project_section.get("name", "heart-risk-cds"),
            version=project_section.get("version", "1.0.0"),
            random_seed=project_section.get("random_seed", 42),
        )

        paths_section = raw.get("paths", {})
        paths_config = PathsConfig(
            raw_data=paths_section.get("raw_data", "data/raw/heart.csv"),
            processed_data_dir=paths_section.get(
                "processed_data_dir", "data/processed"
            ),
            models_dir=paths_section.get("models_dir", "models/saved"),
            figures_dir=paths_section.get("figures_dir", "reports/figures"),
        )

        thresh_section = raw.get("clinical_thresholds", {})
        thresh_config = ClinicalThresholdsConfig(
            low_risk=thresh_section.get("low_risk", 0.25),
            moderate_risk=thresh_section.get("moderate_risk", 0.50),
            high_risk=thresh_section.get("high_risk", 0.75),
            critical_risk=thresh_section.get("critical_risk", 0.90),
        )

        train_section = raw.get("training", {})
        train_config = TrainingConfig(
            test_size=train_section.get("test_size", 0.20),
            cv_folds=train_section.get("cv_folds", 5),
            repeated_cv_repeats=train_section.get("repeated_cv_repeats", 3),
            scoring_metric=train_section.get("scoring_metric", "roc_auc"),
            calibration_method=train_section.get("calibration_method", "sigmoid"),
        )

        return Config(
            project=project_config,
            paths=paths_config,
            data=data_config,
            clinical_thresholds=thresh_config,
            training=train_config,
            models_to_evaluate=raw.get(
                "models_to_evaluate",
                [
                    "logistic_regression",
                    "random_forest",
                    "extra_trees",
                    "gradient_boosting",
                    "adaboost",
                    "svc",
                    "stacking_ensemble",
                    "voting_ensemble",
                    "neural_network_mlp",
                ],
            ),
        )

    return Config()
