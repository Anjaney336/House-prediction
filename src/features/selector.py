from __future__ import annotations

import numpy as np
import pandas as pd


def numeric_target_correlations(df: pd.DataFrame, target: str) -> pd.Series:
    """Rank numeric features by absolute target correlation."""
    numeric = df.select_dtypes(include=np.number)
    if target not in numeric:
        return pd.Series(dtype=float)
    return numeric.corr()[target].drop(labels=[target]).abs().sort_values(ascending=False)
