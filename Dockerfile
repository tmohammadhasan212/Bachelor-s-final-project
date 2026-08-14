# Multi-stage Dockerfile for Heart Risk Clinical Decision Support System
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt && pip install -e .

# Copy project source code and data
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/
COPY docs/ ./docs/
COPY tests/ ./tests/

# Pre-train and serialize default model artifacts
RUN python -m src.heart_risk.models.train

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command launches the Streamlit Clinical Dashboard
CMD ["streamlit", "run", "src/heart_risk/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
