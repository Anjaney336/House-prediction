from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score

from src.domain.property_ontology import map_column
from src.models.trainer import TrainingResult


def production_validation_evidence(
    result: TrainingResult, frame: pd.DataFrame, target: str, features: list[str], folds: int = 3,
) -> dict[str, Any]:
    """Audit the selected pipeline under geographic and temporal structures when available."""
    complete = frame.loc[frame[target].notna(), [*features, target]].copy()
    X, y = complete[features], pd.to_numeric(complete[target], errors="raise")
    evidence: dict[str, Any] = {"geographic_rmse": None, "temporal_rmse": None, "error_by_segment": {}}
    location_columns = [column for column in features if (map_column(column) and map_column(column).role in {"city", "locality", "sector"})]
    if location_columns:
        group_column = location_columns[0]
        groups = X[group_column].fillna("__missing_location__")
        splits = min(folds, int(groups.nunique()))
        if splits >= 2:
            scores = cross_val_score(result.active_pipeline, X, y, groups=groups, cv=GroupKFold(splits), scoring="neg_root_mean_squared_error", n_jobs=1)
            evidence["geographic_rmse"] = float(-scores.mean())
            evidence["geographic_group"] = group_column
    date_columns = [column for column in features if (map_column(column) and map_column(column).role in {"listing_date", "transaction_date"})]
    if date_columns:
        date_column = date_columns[0]
        parsed = pd.to_datetime(X[date_column], errors="coerce")
        valid = parsed.notna()
        if valid.sum() >= 30 and parsed[valid].nunique() >= folds + 1:
            order = parsed[valid].sort_values(kind="stable").index
            scores = cross_val_score(result.active_pipeline, X.loc[order], y.loc[order], cv=TimeSeriesSplit(folds), scoring="neg_root_mean_squared_error", n_jobs=1)
            evidence["temporal_rmse"] = float(-scores.mean())
            evidence["temporal_field"] = date_column
    property_columns = [column for column in features if (map_column(column) and map_column(column).role == "property_type")]
    if property_columns:
        property_column = property_columns[0]
        audit = pd.DataFrame({
            "segment": result.X_test[property_column].astype("string").fillna("Missing").to_numpy(),
            "actual": result.y_test.to_numpy(),
            "prediction": np.asarray(result.test_predictions[result.active_model_name]),
        })
        for segment, group in audit.groupby("segment"):
            if len(group) >= 3:
                evidence["error_by_segment"][str(segment)] = {"rows": len(group), "mae": float(mean_absolute_error(group["actual"], group["prediction"]))}
    return evidence


def shadow_decision(current_card: dict[str, Any], candidate_card: dict[str, Any], minimum_improvement: float = 0.0) -> dict[str, Any]:
    current = float(current_card["metrics"]["CV RMSE"])
    candidate = float(candidate_card["metrics"]["CV RMSE"])
    improvement = (current - candidate) / max(current, 1e-9)
    coverage = float(candidate_card.get("conformal_test_coverage") or 0)
    approved = improvement >= minimum_improvement and coverage >= 0.85
    return {"approved": approved, "cv_rmse_improvement_percent": improvement * 100, "candidate_test_coverage": coverage, "minimum_improvement_percent": minimum_improvement * 100}
