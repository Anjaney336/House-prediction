from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.domain.property_ontology import map_column, normalize_name


@dataclass(frozen=True)
class TargetCandidate:
    column: str
    target_type: str
    score: float
    reason: str
    currency_hypothesis: str | None = None
    target_unit: str = "currency amount"


TARGET_WEIGHTS = {
    "sale_price": 1.0,
    "rent": 0.95,
    "price_per_sqft": 0.85,
}


def infer_currency(column: str) -> str | None:
    name = normalize_name(column)
    markers = {"inr": "INR", "usd": "USD", "gbp": "GBP", "eur": "EUR", "aed": "AED", "cad": "CAD", "aud": "AUD"}
    return next((currency for marker, currency in markers.items() if marker in name.split("_")), None)


def detect_targets(df: pd.DataFrame) -> list[TargetCandidate]:
    """Rank numeric prediction targets and explain the recommendation."""
    candidates = []
    for column in df.select_dtypes("number").columns:
        entry = map_column(str(column))
        semantic = TARGET_WEIGHTS.get(entry.role, 0.0) if entry else 0.0
        name = normalize_name(str(column))
        fallback = 0.55 if any(token in name for token in ("price", "value", "rent", "cost", "amount", "target")) else 0.0
        score = max(semantic, fallback)
        if score:
            target_type = entry.role if entry else "numeric_target"
            completeness = float(df[column].notna().mean())
            variance = 0.1 if df[column].nunique(dropna=True) > 10 else 0.0
            final = min(1.0, score * 0.8 + completeness * 0.1 + variance)
            unit = "currency per square foot" if target_type == "price_per_sqft" else "currency amount"
            candidates.append(TargetCandidate(str(column), target_type, round(final, 3), "Numeric column with a strong valuation semantic match.", infer_currency(str(column)), unit))
    return sorted(candidates, key=lambda item: item.score, reverse=True)
