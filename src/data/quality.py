from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.domain.domain_detector import DomainAnalysis
from src.features.leakage import LeakageWarning


@dataclass(frozen=True)
class QualityAssessment:
    overall: int
    completeness: int
    consistency: int
    target_quality: int
    feature_richness: int
    semantic_confidence: int
    leakage_risk: str
    valuation_suitability: int
    suitability_label: str

    def to_dict(self) -> dict:
        return asdict(self)


def assess_quality(
    df: pd.DataFrame,
    target: str | None,
    features: list[str],
    domain: DomainAnalysis,
    leakage: list[LeakageWarning] | None = None,
) -> QualityAssessment:
    leakage = leakage or []
    requested = list(dict.fromkeys(features + ([target] if target and target in df else [])))
    available = [column for column in requested if column in df.columns]
    relevant = df.loc[:, available] if available else df
    completeness = int(round(100 * (1 - float(relevant.isna().mean().mean()))))
    duplicate_penalty = min(30, int(round(100 * df.duplicated().mean())))
    consistency = max(0, 100 - duplicate_penalty)
    if target and target in df and pd.api.types.is_numeric_dtype(df[target]):
        valid = pd.to_numeric(df[target], errors="coerce")
        target_quality = int(round(100 * valid.notna().mean()))
        if valid.nunique(dropna=True) < 10:
            target_quality = min(target_quality, 45)
    else:
        target_quality = 0
    feature_richness = min(100, 25 + len(features) * 7 + len(domain.available_signals) * 4)
    semantic = int(round(domain.confidence * 100)) if domain.is_real_estate else 0
    high = sum(item.severity == "high" for item in leakage)
    medium = sum(item.severity == "medium" for item in leakage)
    leakage_penalty = min(45, high * 18 + medium * 6)
    leakage_risk = "High" if high else ("Medium" if medium else "Low")
    overall = int(round(np.clip(0.28 * completeness + 0.15 * consistency + 0.24 * target_quality + 0.18 * feature_richness + 0.15 * max(semantic, 50 if not domain.is_real_estate else semantic) - leakage_penalty, 0, 100)))
    granularity_bonus = 12 if domain.granularity not in {"Unknown", "Census Block / Geographic Area"} else 0
    suitability = int(round(np.clip(0.3 * semantic + 0.25 * target_quality + 0.2 * completeness + 0.15 * feature_richness + granularity_bonus - leakage_penalty, 0, 100))) if domain.is_real_estate else 0
    label = "Strong" if suitability >= 80 else "Moderate" if suitability >= 60 else "Limited" if suitability >= 35 else "Unsuitable"
    return QualityAssessment(overall, completeness, consistency, target_quality, feature_richness, semantic, leakage_risk, suitability, label)
