from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
SAMPLE_DATASET = ROOT_DIR / "data" / "sample_datasets" / "housing_sample.csv"

APP_NAME = "PricePredict AI"
RANDOM_STATE = 42
MIN_TRAINING_ROWS = 30
HIGH_CARDINALITY_THRESHOLD = 50


def ensure_runtime_directories() -> None:
    """Create directories used for session artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
