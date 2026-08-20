from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.domain.property_ontology import normalize_name


@dataclass(frozen=True)
class LeakageWarning:
    column: str
    severity: str
    reason: str


def detect_leakage(df: pd.DataFrame, target: str, features: list[str]) -> list[LeakageWarning]:
    """Flag post-outcome fields, identifiers, and suspicious target proxies."""
    warnings = []
    target_name = normalize_name(target)
    for column in features:
        name = normalize_name(column)
        if any(token in name for token in ("transaction_id", "listing_id", "property_id", "url", "sold_date", "closing_date")):
            warnings.append(LeakageWarning(column, "medium", "Identifier or post-event field may leak entity/outcome information."))
        if any(token in name for token in ("sold_value", "sale_amount", "target_price", "future_price", "future_sale", "post_sale", "transaction_amount", "target_copy", "near_perfect")):
            warnings.append(LeakageWarning(column, "high", "Column name suggests a direct proxy for the selected outcome."))
        if "price" in target_name and "price_per" in name:
            warnings.append(LeakageWarning(column, "high", "A price-derived feature can contain the selected price target."))
        if pd.api.types.is_numeric_dtype(df[column]) and pd.api.types.is_numeric_dtype(df[target]):
            pair = df[[column, target]].dropna()
            if len(pair) > 20 and pair[column].nunique() > 2:
                correlation = abs(float(pair.corr().iloc[0, 1]))
                if np.isfinite(correlation) and correlation > 0.995:
                    warnings.append(LeakageWarning(column, "high", f"Near-perfect target correlation ({correlation:.3f}) suggests leakage."))
    deduped = {(warning.column, warning.reason): warning for warning in warnings}
    return list(deduped.values())
