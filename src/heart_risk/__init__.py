"""
Heart Risk Clinical Decision Support System (CDSS)
===================================================
A production-grade, explainable machine learning package for early cardiovascular
risk stratification, clinical utility analysis, and interactive patient triage.
"""

__version__ = "1.0.0"
__author__ = "Mohammad Hasan"

from src.heart_risk.config import Config, load_config
from src.heart_risk.data.loader import load_dataset
from src.heart_risk.features.pipeline import create_preprocessing_pipeline
from src.heart_risk.models.train import ModelTrainer

__all__ = [
    "Config",
    "load_config",
    "load_dataset",
    "create_preprocessing_pipeline",
    "ModelTrainer",
]
