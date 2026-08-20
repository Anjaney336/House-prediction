from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.domain.property_ontology import map_column, normalize_name


MARKETS = {
    "NOIDA": ("noida", "greater noida"),
    "GURUGRAM": ("gurugram", "gurgaon"),
    "DELHI": ("delhi", "new delhi"),
    "MUMBAI": ("mumbai", "bombay"),
    "BENGALURU": ("bengaluru", "bangalore"),
    "DUBAI": ("dubai",),
    "AUSTIN": ("austin",),
    "DENVER": ("denver",),
    "PORTLAND": ("portland",),
}
MARKET_REGIONS = {"NOIDA": "DELHI-NCR", "GURUGRAM": "DELHI-NCR", "DELHI": "DELHI-NCR"}


def region_for_market(market: str | None) -> str | None:
    if not market:
        return None
    normalized = market.upper()
    return MARKET_REGIONS.get(normalized, normalized)


@dataclass(frozen=True)
class MarketAnalysis:
    candidate: str | None
    confidence: str
    score: float
    evidence: tuple[str, ...]
    requires_confirmation: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def detect_market(df: pd.DataFrame, source_name: str | None = None) -> MarketAnalysis:
    evidence: dict[str, list[str]] = {market: [] for market in MARKETS}
    score: dict[str, float] = {market: 0.0 for market in MARKETS}
    semantic_columns = []
    for column in df.columns:
        entry = map_column(str(column))
        if entry and entry.role in {"city", "locality", "sector"}:
            semantic_columns.append((column, entry.role))
    for column, role in semantic_columns:
        values = " ".join(df[column].dropna().astype(str).value_counts().head(100).index).casefold()
        for market, aliases in MARKETS.items():
            if any(alias in values for alias in aliases):
                weight = 0.85 if role == "city" else 0.65
                score[market] += weight
                evidence[market].append(f"Observed {role} values in '{column}' reference {market.title()}.")
    if source_name:
        source = normalize_name(Path(source_name).stem)
        for market, aliases in MARKETS.items():
            if any(normalize_name(alias) in source for alias in aliases):
                score[market] += 0.35
                evidence[market].append(f"Source filename suggests {market.title()}; filename evidence is not authoritative.")
    if semantic_columns and any(role == "sector" for _, role in semantic_columns):
        for market in MARKETS:
            if score[market]:
                score[market] += 0.1
                evidence[market].append("A sector field provides additional market-structure evidence.")
    candidate = max(score, key=score.get)
    best = min(1.0, score[candidate])
    if best <= 0:
        return MarketAnalysis(None, "Unconfirmed", 0.0, tuple("Sector/locality structure exists but no city can be established from the data." for _ in [0]) if semantic_columns else ("No reliable market evidence was found.",))
    confidence = "High" if best >= 0.8 else "Medium" if best >= 0.45 else "Low"
    return MarketAnalysis(candidate, confidence, round(best, 3), tuple(evidence[candidate]))
