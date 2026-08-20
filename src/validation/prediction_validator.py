from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.validation.schema_contract import ModelSchemaContract


@dataclass(frozen=True)
class PredictionValidation:
    valid_rows: int
    invalid_rows: int
    missing_columns: tuple[str, ...]
    unexpected_columns: tuple[str, ...]
    warnings: tuple[str, ...]
    prepared: pd.DataFrame


def validate_and_prepare(rows: pd.DataFrame, contract: ModelSchemaContract) -> PredictionValidation:
    """Validate names/order/types and retain missing values for fitted imputers."""
    expected = list(contract.feature_order)
    missing = tuple(column for column in expected if column not in rows.columns)
    unexpected = tuple(column for column in rows.columns if column not in expected)
    prepared = rows.copy()
    for column in missing:
        prepared[column] = np.nan
    warnings = []
    invalid_mask = pd.Series(False, index=prepared.index)
    for spec in contract.features:
        if spec.dtype == "numeric":
            original_non_null = prepared[spec.name].notna()
            converted = pd.to_numeric(prepared[spec.name], errors="coerce")
            bad = original_non_null & converted.isna()
            invalid_mask |= bad
            if bad.any():
                warnings.append(f"{spec.label}: {int(bad.sum())} value(s) could not be parsed as numeric.")
            prepared[spec.name] = converted
        elif spec.dtype == "datetime":
            original_non_null = prepared[spec.name].notna()
            converted = pd.to_datetime(prepared[spec.name], errors="coerce")
            bad = original_non_null & converted.isna()
            invalid_mask |= bad
            if bad.any():
                warnings.append(f"{spec.label}: {int(bad.sum())} value(s) could not be parsed as a date.")
            prepared[spec.name] = converted
        else:
            prepared[spec.name] = prepared[spec.name].where(prepared[spec.name].isna(), prepared[spec.name].astype(str))
            unseen = set(prepared[spec.name].dropna().astype(str)) - set(spec.vocabulary)
            if unseen:
                warnings.append(f"{spec.label}: {len(unseen)} unseen category value(s) will use unknown-category handling.")
    if missing:
        warnings.append(f"Missing columns will be imputed by the fitted pipeline: {', '.join(missing)}.")
    if unexpected:
        warnings.append(f"Unexpected columns will be ignored: {', '.join(unexpected)}.")
    return PredictionValidation(
        valid_rows=int((~invalid_mask).sum()),
        invalid_rows=int(invalid_mask.sum()),
        missing_columns=missing,
        unexpected_columns=unexpected,
        warnings=tuple(warnings),
        prepared=prepared.loc[~invalid_mask, expected],
    )
