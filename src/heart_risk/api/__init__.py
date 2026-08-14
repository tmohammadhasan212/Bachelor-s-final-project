"""
FastAPI production REST API service for clinical risk prediction.
"""

from src.heart_risk.api.app import create_app

__all__ = ["create_app"]
