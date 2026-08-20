from __future__ import annotations

import pandas as pd

from src.utils.schema import infer_column_roles


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact, UI-friendly column profile."""
    roles = infer_column_roles(df)
    role_map = {
        **{c: "numeric" for c in roles.numeric},
        **{c: "categorical" for c in roles.categorical},
        **{c: "boolean" for c in roles.boolean},
        **{c: "datetime" for c in roles.datetime},
        **{c: "high-cardinality text" for c in roles.high_cardinality},
    }
    rows = []
    for column in df.columns:
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        rows.append(
            {
                "column": column,
                "role": role_map.get(column, "unknown"),
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "missing_%": round(float(series.isna().mean() * 100), 2),
                "unique": int(series.nunique(dropna=True)),
                "mean": round(float(numeric.mean()), 3) if numeric.notna().any() else None,
                "median": round(float(numeric.median()), 3) if numeric.notna().any() else None,
                "min": round(float(numeric.min()), 3) if numeric.notna().any() else None,
                "max": round(float(numeric.max()), 3) if numeric.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def data_quality_summary(df: pd.DataFrame) -> dict[str, int | float]:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 2),
    }
