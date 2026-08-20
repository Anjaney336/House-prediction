from __future__ import annotations

from dataclasses import asdict, dataclass
import re

import pandas as pd

from src.domain.property_ontology import map_column


@dataclass(frozen=True)
class DataIssue:
    column: str
    issue_type: str
    severity: str
    message: str
    recommendation: str

    def to_dict(self) -> dict:
        return asdict(self)


FORMATTED_NUMBER = re.compile(r"(?:₹|\$|€|£)?\s*[\d,]+(?:\.\d+)?\s*(?:sq\.?\s*ft|sqft|lakh|lac|crore|cr|bhk|bathrooms?)", re.IGNORECASE)


def detect_data_issues(df: pd.DataFrame) -> list[DataIssue]:
    """Detect invalid values separately from valid-but-extreme observations."""
    issues: list[DataIssue] = []
    for column in df.columns:
        series = df[column]
        entry = map_column(str(column))
        role = entry.role if entry else "other"
        if not pd.api.types.is_numeric_dtype(series):
            sample = series.dropna().astype(str).head(500)
            matches = sample.str.contains(FORMATTED_NUMBER).sum()
            if matches:
                examples = ", ".join(sample[sample.str.contains(FORMATTED_NUMBER)].head(3).tolist())
                issues.append(DataIssue(column, "Formatted numeric text", "medium", f"Detected {matches} formatted numeric value(s), such as {examples}.", "Preview a unit-aware numeric parsing rule before applying it."))
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if role in {"bedrooms", "bathrooms", "parking", "floor", "total_floors", "area", "built_up_area", "carpet_area", "plot_area", "sale_price", "rent"} and (numeric < 0).any():
            issues.append(DataIssue(column, "Invalid negative values", "high", f"{int((numeric < 0).sum())} negative value(s) are not semantically valid.", "Review or mark invalid entries as missing; do not treat them as ordinary outliers."))
        if role == "latitude" and ((numeric < -90) | (numeric > 90)).any():
            issues.append(DataIssue(column, "Invalid coordinates", "high", "Latitude values must be between -90 and 90.", "Correct or mark invalid coordinates as missing."))
        if role == "longitude" and ((numeric < -180) | (numeric > 180)).any():
            issues.append(DataIssue(column, "Invalid coordinates", "high", "Longitude values must be between -180 and 180.", "Correct or mark invalid coordinates as missing."))
        clean = numeric.dropna()
        if len(clean) >= 20:
            q1, q3 = clean.quantile([0.25, 0.75])
            iqr = q3 - q1
            extreme = ((clean < q1 - 3 * iqr) | (clean > q3 + 3 * iqr)).sum()
            if extreme:
                issues.append(DataIssue(column, "Extreme observations", "low", f"{int(extreme)} value(s) fall beyond 3×IQR.", "Inspect these records; they may be valid premium/extreme properties and should not be deleted blindly."))
    duplicates = int(df.duplicated().sum())
    if duplicates:
        issues.append(DataIssue("All columns", "Exact duplicates", "medium", f"{duplicates} exact duplicate row(s) detected.", "Investigate repeated listings before splitting to reduce validation contamination."))
    return issues
