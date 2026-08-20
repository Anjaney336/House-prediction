from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.platform.persistence import list_records


def due_for_retraining(tenant_id: str, cadence_days: int = 90, now: datetime | None = None) -> list[dict[str, Any]]:
    """Return active models whose configured retraining cadence has elapsed."""
    current = now or datetime.now(timezone.utc)
    due = []
    for record in list_records("models", tenant_id):
        if not record["is_active"]:
            continue
        trained = datetime.fromisoformat(record["created_at"])
        if current - trained >= timedelta(days=cadence_days):
            card = json.loads(record["model_card"])
            due.append({"model_id": record["id"], "dataset_id": record["dataset_id"], "target": card["target"], "trained_at": record["created_at"]})
    return due


def drift_status(deployment_rmse: float, recent_rmse: float, threshold: float = 0.20) -> dict[str, Any]:
    if deployment_rmse <= 0:
        raise ValueError("Deployment RMSE must be positive.")
    degradation = (recent_rmse - deployment_rmse) / deployment_rmse
    return {
        "deployment_rmse": deployment_rmse,
        "recent_rmse": recent_rmse,
        "degradation_percent": degradation * 100,
        "threshold_percent": threshold * 100,
        "alert": degradation > threshold,
    }


def _distribution_drift(reference: pd.Series, recent: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(reference):
        clean_reference = pd.to_numeric(reference, errors="coerce").dropna()
        clean_recent = pd.to_numeric(recent, errors="coerce").dropna()
        if clean_reference.empty or clean_recent.empty:
            return 1.0
        edges = np.unique(clean_reference.quantile(np.linspace(0, 1, 11)).to_numpy())
        if len(edges) < 3:
            return float(abs(clean_reference.mean() - clean_recent.mean()) / max(abs(clean_reference.mean()), 1e-9))
        reference_hist = np.histogram(clean_reference, bins=edges)[0] / len(clean_reference)
        recent_hist = np.histogram(clean_recent, bins=edges)[0] / len(clean_recent)
        return float(0.5 * np.abs(reference_hist - recent_hist).sum())
    reference_share = reference.astype("string").fillna("Missing").value_counts(normalize=True)
    recent_share = recent.astype("string").fillna("Missing").value_counts(normalize=True)
    categories = reference_share.index.union(recent_share.index)
    return float(0.5 * np.abs(reference_share.reindex(categories, fill_value=0) - recent_share.reindex(categories, fill_value=0)).sum())


def drift_report(
    reference: pd.DataFrame, recent: pd.DataFrame, features: list[str],
    *, target: str | None = None, reference_predictions=None, recent_predictions=None,
    recent_actual=None, threshold: float = 0.20,
) -> dict[str, Any]:
    available = [column for column in features if column in reference and column in recent]
    feature_drift = {column: _distribution_drift(reference[column], recent[column]) for column in available}
    target_drift = _distribution_drift(reference[target], recent[target]) if target and target in reference and target in recent else None
    prediction_drift = None
    if reference_predictions is not None and recent_predictions is not None:
        prediction_drift = _distribution_drift(pd.Series(reference_predictions), pd.Series(recent_predictions))
    error = None
    if recent_actual is not None and recent_predictions is not None:
        residual = np.asarray(recent_actual, dtype=float) - np.asarray(recent_predictions, dtype=float)
        error = float(np.sqrt(np.mean(residual**2)))
    maximum = max([*feature_drift.values(), target_drift or 0, prediction_drift or 0], default=0)
    return {
        "feature_drift": feature_drift, "target_drift": target_drift,
        "prediction_drift": prediction_drift, "recent_rmse": error,
        "threshold": threshold, "alert": maximum > threshold,
        "retraining_recommended": maximum > threshold,
    }
