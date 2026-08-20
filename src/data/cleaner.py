from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.utils.config import HIGH_CARDINALITY_THRESHOLD


@dataclass
class CleaningConfig:
    numeric_imputation: str = "median"
    categorical_imputation: str = "most_frequent"
    scaling: str = "standard"
    outlier_strategy: str = "none"
    numeric_by_column: dict[str, str] = field(default_factory=dict)
    categorical_by_column: dict[str, str] = field(default_factory=dict)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Fit training-only IQR bounds and clip numeric values during transform."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.q1_ = np.nanpercentile(values, 25, axis=0)
        self.q3_ = np.nanpercentile(values, 75, axis=0)
        spread = self.q3_ - self.q1_
        self.lower_ = self.q1_ - self.factor * spread
        self.upper_ = self.q3_ + self.factor * spread
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


class DateTimeOrdinalTransformer(BaseEstimator, TransformerMixin):
    """Convert datetime columns to elapsed days without learning from validation data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        output = np.empty((len(frame), len(frame.columns)), dtype=float)
        for index, column in enumerate(frame.columns):
            parsed = pd.to_datetime(frame[column], errors="coerce")
            output[:, index] = parsed.astype("int64", copy=False).to_numpy(dtype=float) / 86_400_000_000_000
            output[parsed.isna().to_numpy(), index] = np.nan
        return output


def _numeric_pipeline(strategy: str, config: CleaningConfig) -> Pipeline:
    strategy = strategy if strategy in {"mean", "median", "most_frequent", "constant"} else "median"
    steps = [("imputer", SimpleImputer(strategy=strategy, fill_value=0))]
    if config.outlier_strategy == "cap_iqr":
        steps.append(("outliers", IQRClipper()))
    if config.scaling == "standard":
        steps.append(("scaler", StandardScaler()))
    elif config.scaling == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    return Pipeline(steps)


def _categorical_pipeline(strategy: str, high_cardinality: bool = False) -> Pipeline:
    strategy = strategy if strategy in {"most_frequent", "constant"} else "most_frequent"
    encoder = (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if high_cardinality
        else OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy=strategy, fill_value="Missing")),
            ("encoder", encoder),
        ]
    )


def build_preprocessor(
    X: pd.DataFrame,
    config: CleaningConfig | None = None,
) -> ColumnTransformer:
    """Build a leakage-safe preprocessing graph from selected training features."""
    config = config or CleaningConfig()
    transformers = []
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    datetime_columns = X.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric and column not in datetime_columns]

    for column in numeric:
        strategy = config.numeric_by_column.get(column, config.numeric_imputation)
        transformers.append((f"num_{column}", _numeric_pipeline(strategy, config), [column]))
    for column in categorical:
        strategy = config.categorical_by_column.get(column, config.categorical_imputation)
        high_cardinality = X[column].nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD
        transformers.append((f"cat_{column}", _categorical_pipeline(strategy, high_cardinality), [column]))
    for column in datetime_columns:
        date_pipeline = Pipeline([
            ("ordinal_days", DateTimeOrdinalTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append((f"date_{column}", date_pipeline, [column]))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def cleaning_preview(df: pd.DataFrame, config: CleaningConfig, rows: int = 20) -> pd.DataFrame:
    """Create a readable imputation/outlier preview without fitting model state globally."""
    preview = df.head(rows).copy()
    for column in preview.select_dtypes(include=np.number):
        strategy = config.numeric_by_column.get(column, config.numeric_imputation)
        fill = preview[column].mean() if strategy == "mean" else preview[column].median()
        if strategy == "constant":
            fill = 0
        preview[column] = preview[column].fillna(fill)
        if config.outlier_strategy == "cap_iqr":
            q1, q3 = df[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            preview[column] = preview[column].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    for column in preview.select_dtypes(exclude=np.number):
        strategy = config.categorical_by_column.get(column, config.categorical_imputation)
        mode = df[column].mode(dropna=True)
        fill = "Missing" if strategy == "constant" or mode.empty else mode.iloc[0]
        preview[column] = preview[column].fillna(fill)
    return preview
