from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.domain.domain_detector import DomainAnalysis
from src.domain.property_ontology import schema_mapping


@dataclass(frozen=True)
class FeatureContract:
    name: str
    dtype: str
    normalized_role: str
    group: str
    label: str
    required: bool
    imputation: str
    missing_rate: float
    minimum: float | None
    maximum: float | None
    median: float | None
    vocabulary: tuple[str, ...]
    unit: str | None = None


@dataclass(frozen=True)
class ModelSchemaContract:
    version: str
    features: tuple[FeatureContract, ...]
    feature_order: tuple[str, ...]
    target: str
    target_dtype: str
    dataset_domain: str
    asset_type: str
    prediction_granularity: str
    prediction_label: str
    currency: str
    row_count: int
    market: str = "UNCONFIRMED"
    region: str | None = None
    property_types: tuple[str, ...] = ()
    transaction_type: str = "Sale"
    target_unit: str = "currency amount"
    training_dataset_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelSchemaContract":
        values = dict(payload)
        values["features"] = tuple(
            FeatureContract(**{**feature, "vocabulary": tuple(feature.get("vocabulary", ()))})
            for feature in values["features"]
        )
        values["feature_order"] = tuple(values["feature_order"])
        values["property_types"] = tuple(values.get("property_types", ()))
        return cls(**values)


def build_schema_contract(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    domain: DomainAnalysis,
    currency: str = "USD",
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    market: str = "UNCONFIRMED",
    region: str | None = None,
    transaction_type: str = "Sale",
    target_unit: str = "currency amount",
    dataset_fingerprint: str | None = None,
) -> ModelSchemaContract:
    mapping = schema_mapping(features)
    specs = []
    for name in features:
        series = df[name]
        normalized_role = mapping[name]["normalized_role"]
        boolean = pd.api.types.is_bool_dtype(series)
        numeric = pd.api.types.is_numeric_dtype(series) and not boolean
        datetime = pd.api.types.is_datetime64_any_dtype(series) or normalized_role in {"listing_date", "transaction_date"}
        clean = pd.to_numeric(series, errors="coerce").dropna() if numeric else pd.Series(dtype=float)
        vocabulary = tuple(map(str, series.dropna().astype(str).value_counts().head(200).index)) if not numeric and not datetime else ()
        specs.append(
            FeatureContract(
                name=name,
                dtype="numeric" if numeric else ("boolean" if boolean else ("datetime" if datetime else "categorical")),
                normalized_role=normalized_role,
                group=mapping[name]["group"],
                label=mapping[name]["label"],
                required=bool(
                    series.isna().mean() <= 0.02
                    and normalized_role in {"property_type", "city", "locality", "sector", "area", "built_up_area", "carpet_area", "plot_area", "bedrooms", "bathrooms", "floor"}
                ),
                imputation=numeric_strategy if numeric else categorical_strategy,
                missing_rate=round(float(series.isna().mean()), 4),
                minimum=float(clean.min()) if len(clean) else None,
                maximum=float(clean.max()) if len(clean) else None,
                median=float(clean.median()) if len(clean) else None,
                vocabulary=vocabulary,
                unit=next((unit for marker, unit in (("sqft", "sqft"), ("sq_m", "sqm"), ("sqm", "sqm"), ("km", "km"), ("year", "years")) if marker in name.lower()), None),
            )
        )
    return ModelSchemaContract(
        version="2.0",
        features=tuple(specs),
        feature_order=tuple(features),
        target=target,
        target_dtype=str(df[target].dtype),
        dataset_domain=domain.domain,
        asset_type=domain.asset_type,
        prediction_granularity=domain.granularity,
        prediction_label=domain.prediction_label,
        currency=currency,
        row_count=len(df),
        market=market,
        region=region,
        property_types=domain.property_types,
        transaction_type=transaction_type,
        target_unit=target_unit,
        training_dataset_fingerprint=dataset_fingerprint,
    )
