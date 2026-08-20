from __future__ import annotations

import numpy as np
import pandas as pd


def target_is_right_skewed(y: pd.Series, threshold: float = 1.0) -> bool:
    """Recommend log1p only for non-negative, strongly right-skewed targets."""
    values = pd.to_numeric(y, errors="coerce").dropna()
    return bool(len(values) and values.min() >= 0 and values.skew() > threshold)


def add_safe_ratio(df: pd.DataFrame, numerator: str, denominator: str, name: str) -> pd.DataFrame:
    """Add a ratio while replacing division infinities with missing values."""
    result = df.copy()
    result[name] = result[numerator] / result[denominator].replace(0, np.nan)
    return result
