from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.domain.property_ontology import map_column
from src.utils.config import HIGH_CARDINALITY_THRESHOLD, MIN_TRAINING_ROWS


TARGET_PATTERN = re.compile(r"price|value|cost|amount|sale|target", re.IGNORECASE)
ID_PATTERN = re.compile(r"(^id$|_id$|^uuid$|identifier|index)", re.IGNORECASE)


@dataclass(frozen=True)
class ColumnRoles:
    numeric: list[str]
    categorical: list[str]
    boolean: list[str]
    datetime: list[str]
    high_cardinality: list[str]


@dataclass(frozen=True)
class SchemaValidationResult:
    valid: bool
    requested_columns: tuple[str, ...]
    available_columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]
    invalid_types: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


def validate_columns(
    df: pd.DataFrame,
    requested: list[str] | tuple[str, ...],
    *,
    expected_numeric: list[str] | tuple[str, ...] = (),
) -> SchemaValidationResult:
    """Validate a requested projection without indexing missing dataframe columns."""
    requested_unique = tuple(dict.fromkeys(map(str, requested)))
    present = set(map(str, df.columns))
    available = tuple(column for column in requested_unique if column in present)
    missing = tuple(column for column in requested_unique if column not in present)
    extra = tuple(column for column in map(str, df.columns) if column not in requested_unique)
    invalid_types = tuple(
        column for column in expected_numeric
        if column in present and not pd.api.types.is_numeric_dtype(df[column])
    )
    warnings = tuple(filter(None, (
        f"Unavailable columns were excluded: {', '.join(missing)}." if missing else "",
        f"Additional dataset columns are available but not selected: {len(extra)}." if extra else "",
    )))
    errors = tuple(filter(None, (
        "No requested columns exist in the active dataset." if requested_unique and not available else "",
        f"Numeric columns have incompatible types: {', '.join(invalid_types)}." if invalid_types else "",
    )))
    return SchemaValidationResult(not errors and not missing and not invalid_types, requested_unique, available, missing, extra, invalid_types, warnings, errors)


def infer_column_roles(df: pd.DataFrame) -> ColumnRoles:
    """Infer modeling roles without mutating the source data."""
    numeric: list[str] = []
    categorical: list[str] = []
    boolean: list[str] = []
    datetime: list[str] = []
    high_cardinality: list[str] = []

    for column in df.columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            boolean.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
        elif pd.api.types.is_datetime64_any_dtype(series):
            datetime.append(column)
        else:
            non_null = series.dropna()
            parsed = pd.to_datetime(non_null, errors="coerce", format="mixed") if len(non_null) else pd.Series(dtype="datetime64[ns]")
            parse_ratio = float(parsed.notna().mean()) if len(non_null) else 0.0
            if parse_ratio >= 0.9 and series.nunique(dropna=True) > 5:
                datetime.append(column)
            elif series.nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD:
                high_cardinality.append(column)
            else:
                categorical.append(column)
    return ColumnRoles(numeric, categorical, boolean, datetime, high_cardinality)


def suggest_target(df: pd.DataFrame) -> str:
    """Choose a likely numeric regression target."""
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    matches = [column for column in numeric if TARGET_PATTERN.search(str(column))]
    return matches[0] if matches else (numeric[-1] if numeric else str(df.columns[-1]))


def suggest_features(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Return useful defaults and columns excluded as likely identifiers/noise."""
    selected: list[str] = []
    excluded: list[str] = []
    for column in df.columns:
        if column == target:
            continue
        unique = df[column].nunique(dropna=True)
        near_constant = unique <= 1
        ontology = map_column(str(column))
        semantic_identifier = bool(ontology and ontology.role == "property_id")
        # High uniqueness is normal for measurements, coordinates, and dates.
        # Treat it as identifier evidence only for unrecognized string fields;
        # otherwise valid continuous predictors such as area were discarded.
        unrecognized_string_key = (
            not pd.api.types.is_numeric_dtype(df[column])
            and not pd.api.types.is_datetime64_any_dtype(df[column])
            and ontology is None
            and unique >= max(100, int(len(df) * 0.95))
        )
        id_like = bool(ID_PATTERN.search(str(column))) or semantic_identifier or unrecognized_string_key
        if near_constant or id_like:
            excluded.append(column)
        else:
            selected.append(column)
    return selected, excluded


def validate_training_frame(df: pd.DataFrame, target: str, features: list[str]) -> list[str]:
    """Return user-facing validation errors for a requested training schema."""
    errors: list[str] = []
    if len(df) < MIN_TRAINING_ROWS:
        errors.append(f"At least {MIN_TRAINING_ROWS} rows are required; found {len(df)}.")
    if target not in df.columns:
        errors.append("The selected target column is missing.")
    elif not pd.api.types.is_numeric_dtype(df[target]):
        errors.append("Regression requires a numeric target column.")
    elif df[target].notna().sum() < MIN_TRAINING_ROWS:
        errors.append("The target has too few non-missing values.")
    if not features:
        errors.append("Select at least one feature column.")
    projection = validate_columns(df, features)
    missing = list(projection.missing_columns)
    if missing:
        errors.append(f"Missing feature columns: {', '.join(missing)}")
    if target in features:
        errors.append("The target cannot also be used as a feature.")
    return errors
