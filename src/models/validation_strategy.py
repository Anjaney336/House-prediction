from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from src.utils.config import RANDOM_STATE


@dataclass(frozen=True)
class ValidationPlan:
    splitter: object
    name: str
    reason: str
    groups: pd.Series | None = None
    order: pd.Index | None = None
    group_column: str | None = None


def recommend_validation(
    df: pd.DataFrame,
    features: list[str],
    folds: int,
    strategy: str = "auto",
    group_column: str | None = None,
) -> ValidationPlan:
    """Choose shuffled, chronological, property-grouped, or geographic validation."""
    strategy = strategy.lower().replace("-", "_").replace(" ", "_")
    if strategy in {"random", "shuffled", "kfold"}:
        return ValidationPlan(KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE), "Shuffled K-fold", "Randomized folds were explicitly selected.")

    lower_names = {column: str(column).lower() for column in df.columns}
    date_columns = [
        column for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[column])
        or any(token in lower_names[column] for token in ("date", "time", "year_sold", "sale_year"))
    ]
    if strategy in {"auto", "time", "chronological", "time_series"}:
        for column in date_columns:
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().mean() >= 0.8 and parsed.nunique() >= folds + 1:
                return ValidationPlan(
                    TimeSeriesSplit(n_splits=folds), "Time-series split",
                    f"Chronological validation selected because '{column}' is a reliable time field.",
                    order=parsed.sort_values(kind="stable").index,
                )
        if strategy != "auto":
            raise ValueError("Time-aware validation was requested, but no reliable date column was found.")

    id_columns = [column for column in df.columns if any(token in lower_names[column] for token in ("property_id", "listing_id", "parcel_id", "house_id"))]
    if strategy in {"auto", "group", "property_grouped"}:
        for column in ([group_column] if group_column else id_columns):
            if column not in df:
                continue
            groups = df[column]
            unique, repeated = groups.nunique(dropna=True), len(groups) - groups.nunique(dropna=True)
            if unique >= folds and repeated >= max(3, int(0.05 * len(groups))):
                return ValidationPlan(
                    GroupKFold(n_splits=folds), "Grouped K-fold",
                    f"Rows are grouped by repeated property identifier '{column}' to reduce identity leakage.",
                    groups=groups.fillna("__missing_group__"), group_column=column,
                )
        if strategy != "auto":
            raise ValueError("Grouped validation was requested, but no repeated identifier with enough groups was found.")

    if strategy in {"geographic", "spatial", "location_grouped"}:
        candidates = [group_column] if group_column else [column for column in df.columns if lower_names[column] in {"city", "locality", "location", "region", "neighborhood", "neighbourhood"}]
        for column in candidates:
            if column in df and df[column].nunique(dropna=True) >= folds:
                return ValidationPlan(
                    GroupKFold(n_splits=folds), "Geographic GroupKFold",
                    f"Locations in '{column}' are held together to measure geographic generalization.",
                    groups=df[column].fillna("__missing_location__"), group_column=column,
                )
        raise ValueError("Geographic validation was requested, but no location field has enough groups.")

    return ValidationPlan(
        KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE), "Shuffled K-fold",
        "No reliable repeated property ID or chronological field was detected; shuffled K-fold is used.",
    )
