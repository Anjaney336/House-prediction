from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.validation.prediction_validator import validate_and_prepare
from src.validation.schema_contract import ModelSchemaContract


def validate_prediction_schema(df: pd.DataFrame, features: list[str]) -> None:
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Prediction data is missing: {', '.join(missing)}")


def predict_batch(pipeline, rows: pd.DataFrame, features: list[str], contract: ModelSchemaContract | None = None) -> np.ndarray:
    if contract is not None:
        validation = validate_and_prepare(rows, contract)
        if validation.invalid_rows:
            raise ValueError(f"{validation.invalid_rows} row(s) contain values incompatible with the trained schema.")
        return np.asarray(pipeline.predict(validation.prepared))
    validate_prediction_schema(rows, features)
    return np.asarray(pipeline.predict(rows.loc[:, features]))


def prediction_interval(
    prediction: float,
    residual_std: float | None = None,
    confidence: float = 0.95,
    calibrated_radius: float | None = None,
) -> tuple[float, float]:
    """Return a calibrated interval when available, with legacy residual fallback."""
    if calibrated_radius is not None:
        if calibrated_radius < 0:
            raise ValueError("Calibrated interval radius cannot be negative.")
        return prediction - calibrated_radius, prediction + calibrated_radius
    if residual_std is None or residual_std < 0:
        raise ValueError("A non-negative residual scale or calibrated radius is required.")
    z = 1.96 if confidence >= 0.95 else 1.645
    return prediction - z * residual_std, prediction + z * residual_std


def similar_rows(pipeline, training_rows: pd.DataFrame, query: pd.DataFrame, n: int = 5) -> pd.Index:
    """Find nearest rows in the fitted preprocessed feature space."""
    preprocess = pipeline.named_steps["preprocess"]
    train_matrix = preprocess.transform(training_rows)
    query_matrix = preprocess.transform(query)
    neighbors = NearestNeighbors(n_neighbors=min(n, len(training_rows))).fit(train_matrix)
    indices = neighbors.kneighbors(query_matrix, return_distance=False)[0]
    return training_rows.iloc[indices].index
