"""
FastAPI application entrypoint for the Heart Risk Clinical Decision Support System.
"""

from typing import Any, Optional
from src.heart_risk.config import load_config

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from src.heart_risk.api.routes import router

    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    CORSMiddleware = None
    router = None
    FASTAPI_AVAILABLE = False


def create_app() -> Optional[Any]:
    """Factory creating and configuring the FastAPI app."""
    if not FASTAPI_AVAILABLE:
        return None

    cfg = load_config()

    app = FastAPI(
        title="Heart Risk Clinical Decision Support API",
        description=(
            "Production-grade, Explainable Machine Learning microservice "
            "for early cardiovascular risk stratification, patient triage, and counterfactual simulation."
        ),
        version=cfg.project.version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


app = create_app()


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        import uvicorn

        uvicorn.run(
            "src.heart_risk.api.app:app", host="0.0.0.0", port=8000, reload=True
        )
    else:
        print("FastAPI is not installed. Run: pip install fastapi uvicorn")
