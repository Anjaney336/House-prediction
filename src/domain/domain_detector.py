from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.demo.california import is_california_aggregate
from src.domain.property_ontology import VALUATION_SIGNAL_ROLES, normalize_name, schema_mapping


RETAIL_TERMS = {"product", "quantity", "discount", "revenue", "customer", "sku", "inventory", "sales"}
AUTO_TERMS = {"vehicle", "car", "mileage", "engine", "make", "model", "horsepower"}
FINANCE_TERMS = {"account", "loan", "credit", "balance", "interest", "borrower", "default"}


@dataclass(frozen=True)
class DomainAnalysis:
    domain: str
    scores: dict[str, float]
    is_real_estate: bool
    confidence: float
    asset_type: str
    granularity: str
    prediction_label: str
    rationale: tuple[str, ...]
    available_signals: tuple[str, ...]
    unavailable_signals: tuple[str, ...]
    comparables_supported: bool
    property_types: tuple[str, ...]


def _term_hits(columns: list[str], terms: set[str]) -> int:
    normalized = [normalize_name(column) for column in columns]
    return sum(any(term in name.split("_") or term in name for term in terms) for name in normalized)


def analyze_domain(df: pd.DataFrame) -> DomainAnalysis:
    """Classify a dataset using names, types, cardinality, and feature co-occurrence."""
    columns = list(map(str, df.columns))
    mapping = schema_mapping(columns)
    roles = {item["normalized_role"] for item in mapping.values()}
    signals = sorted(roles & VALUATION_SIGNAL_ROLES)
    financial = bool(roles & {"sale_price", "rent", "price_per_sqft"})
    geo_pair = {"latitude", "longitude"}.issubset(roles)
    structure = bool(roles & {"area", "built_up_area", "plot_area", "bedrooms", "bathrooms", "property_type"})
    real_raw = len(signals) * 1.3 + financial * 3 + geo_pair * 2 + structure * 2
    retail_raw = _term_hits(columns, RETAIL_TERMS) * 2.0
    auto_raw = _term_hits(columns, AUTO_TERMS) * 2.0
    finance_raw = _term_hits(columns, FINANCE_TERMS) * 2.0
    other_raw = 1.5
    total = real_raw + retail_raw + auto_raw + finance_raw + other_raw
    scores = {
        "REAL_ESTATE": real_raw / total,
        "RETAIL": retail_raw / total,
        "AUTOMOTIVE": auto_raw / total,
        "FINANCE": finance_raw / total,
        "OTHER": other_raw / total,
    }
    is_real_estate = scores["REAL_ESTATE"] >= 0.48 and (financial or len(signals) >= 3)

    normalized = set(normalize_name(c) for c in columns)
    california_block = is_california_aggregate(normalized)
    land = ("plot_area" in roles or {"zoning", "frontage"}.issubset(roles)) and not bool(roles & {"bedrooms", "bathrooms"})
    commercial = _term_hits(columns, {"commercial", "office", "shop", "warehouse", "industrial", "lease"}) > 0
    rental = "rent" in roles or _term_hits(columns, {"rent"}) > 0
    property_identity = _term_hits(columns, {"property_id", "listing_id", "transaction_id", "address"}) > 0

    if california_block:
        asset_type, granularity = "Housing", "Census Block / Geographic Area"
        prediction_label = "Estimated Area/Block Housing Value"
    elif land:
        asset_type, granularity = "Land", "Land Parcel"
        prediction_label = "Estimated Land Value"
    elif commercial:
        asset_type, granularity = "Commercial", "Commercial Property"
        prediction_label = "Estimated Commercial Property Value"
    elif rental:
        asset_type, granularity = "Rental", "Rental Unit"
        prediction_label = "Estimated Rental Value"
    elif is_real_estate and (property_identity or structure):
        asset_type, granularity = "Residential", "Property Listing" if property_identity else "Individual Property"
        prediction_label = "Estimated Property Value"
    else:
        asset_type, granularity = "Generic", "Unknown"
        prediction_label = "Predicted Target Value"

    rationale = []
    if financial:
        rationale.append("A valuation or rental target signal is present.")
    if geo_pair:
        rationale.append("Latitude and longitude provide geographic context.")
    if structure:
        rationale.append("Property characteristics were detected.")
    if california_block:
        rationale.append("Aggregate demographic and room-count fields indicate area-level observations.")
    if not is_real_estate:
        strongest = max((k for k in scores if k != "REAL_ESTATE"), key=scores.get)
        rationale.append(f"The strongest non-property signal is {strongest.lower().replace('_', ' ')}.")

    expected = {"area", "bedrooms", "bathrooms", "property_type", "locality", "parking", "building_age", "floor", "latitude", "longitude", "amenities"}
    available = sorted(expected & roles)
    unavailable = sorted(expected - roles)
    comparables = bool(is_real_estate and granularity in {"Individual Property", "Property Listing", "Property Transaction", "Land Parcel", "Commercial Property", "Rental Unit"} and len(df) >= 50 and len(signals) >= 3)
    type_columns = [column for column, item in mapping.items() if item["normalized_role"] == "property_type"]
    property_types = tuple(map(str, df[type_columns[0]].dropna().value_counts().head(25).index)) if type_columns else ()
    return DomainAnalysis(
        domain="REAL_ESTATE" if is_real_estate else "GENERIC_REGRESSION",
        scores={key: round(value, 3) for key, value in scores.items()},
        is_real_estate=is_real_estate,
        confidence=round(scores["REAL_ESTATE"], 3),
        asset_type=asset_type,
        granularity=granularity,
        prediction_label=prediction_label,
        rationale=tuple(rationale),
        available_signals=tuple(available),
        unavailable_signals=tuple(unavailable),
        comparables_supported=comparables,
        property_types=property_types,
    )
