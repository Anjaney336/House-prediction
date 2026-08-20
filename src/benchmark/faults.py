from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.data.validator import detect_data_issues
from src.domain.property_ontology import map_column
from src.features.leakage import detect_leakage


@dataclass(frozen=True)
class FaultFinding:
    status: str
    problem: str
    severity: str
    recommended_action: str
    column: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def audit_faults(frame: pd.DataFrame, target: str | None = None, features: list[str] | None = None) -> list[FaultFinding]:
    """Return user-actionable findings without mutating the source frame."""
    findings: list[FaultFinding] = []
    if frame.empty:
        return [FaultFinding("BLOCKED", "The dataset contains no rows.", "BLOCKER", "Upload a non-empty CSV.")]
    if target is None or target not in frame:
        findings.append(FaultFinding("BLOCKED", "No prediction target is available.", "BLOCKER", "Select or provide a historical price, value, or rent target.", target))
    else:
        target_values = frame[target]
        missing = float(target_values.isna().mean())
        if missing == 1:
            findings.append(FaultFinding("BLOCKED", "The target column contains no usable values.", "BLOCKER", "Populate the target from verified historical outcomes.", target))
        elif missing >= 0.8:
            findings.append(FaultFinding("UNRELIABLE", f"The target is {missing:.0%} missing.", "HIGH", "Obtain substantially more labeled outcomes before training.", target))
        numeric = pd.to_numeric(target_values, errors="coerce")
        malformed = target_values.notna() & numeric.isna()
        if malformed.any():
            findings.append(FaultFinding("BLOCKED", f"{int(malformed.sum())} target values are not numeric.", "BLOCKER", "Correct malformed target values or choose a numeric target.", target))
        if numeric.notna().sum() and numeric.dropna().nunique() <= 1:
            findings.append(FaultFinding("BLOCKED", "The target is constant and cannot support regression.", "BLOCKER", "Use a target with meaningful historical variation.", target))
        if (numeric.dropna() < 0).any():
            findings.append(FaultFinding("UNRELIABLE", "Negative valuation outcomes were detected.", "HIGH", "Verify currency, signs, and invalid records before training.", target))
    if len(frame) < 20:
        findings.append(FaultFinding("BLOCKED", f"Only {len(frame)} rows are available.", "BLOCKER", "Provide at least 20 labeled rows; hundreds are strongly preferred."))
    elif len(frame) < 100:
        findings.append(FaultFinding("LIMITED", f"Only {len(frame)} rows are available.", "MEDIUM", "Use simpler models and collect more representative observations."))
    all_null = frame.columns[frame.isna().all()].tolist()
    for column in all_null:
        findings.append(FaultFinding("LIMITED", "The column contains no values.", "MEDIUM", "Remove it from model features or restore source data.", column))
    numeric_roles = {"area", "built_up_area", "carpet_area", "plot_area", "bedrooms", "bathrooms", "floor", "total_floors", "building_age", "parking", "latitude", "longitude", "sale_price", "rent", "road_width", "frontage"}
    for column in frame.columns:
        entry = map_column(column)
        if not entry or entry.role not in numeric_roles or pd.api.types.is_numeric_dtype(frame[column]):
            continue
        raw = frame[column].copy()
        if pd.api.types.is_object_dtype(raw) or pd.api.types.is_string_dtype(raw):
            raw = raw.mask(raw.astype(str).str.strip().eq(""))
        parsed = pd.to_numeric(raw, errors="coerce")
        malformed = raw.notna() & parsed.isna()
        if malformed.any():
            findings.append(FaultFinding("UNRELIABLE", f"{int(malformed.sum())} value(s) cannot be interpreted as numeric measurements.", "HIGH", "Correct malformed values and standardize units before modeling.", column))
    for issue in detect_data_issues(frame):
        severity = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}.get(issue.severity, issue.severity.upper())
        findings.append(FaultFinding("UNRELIABLE" if severity == "HIGH" else "LIMITED", issue.message, severity, issue.recommendation, issue.column))
    if target in frame:
        chosen = features or [column for column in frame.columns if column != target]
        for warning in detect_leakage(frame, target, chosen):
            severity = "HIGH" if warning.severity == "high" else "MEDIUM"
            findings.append(FaultFinding("UNRELIABLE", warning.reason, severity, "Exclude this field unless it is provably available before the outcome.", warning.column))
    high_cardinality = [column for column in frame.select_dtypes(exclude="number") if frame[column].nunique(dropna=True) > max(100, len(frame) * 0.5)]
    for column in high_cardinality:
        findings.append(FaultFinding("LIMITED", "High-cardinality text may behave like an identifier.", "LOW", "Exclude it or use a validated encoding strategy.", column))
    return findings or [FaultFinding("READY", "No blocking structural faults were detected.", "LOW", "Continue with model validation.")]


def findings_frame(findings: list[FaultFinding]) -> pd.DataFrame:
    return pd.DataFrame([finding.to_dict() for finding in findings])


def fault_scenarios(frame: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    """Create deterministic regression scenarios for recovery and crash-resilience tests."""
    base = frame.copy(deep=True)
    feature = next((column for column in base.columns if column != target and "area" in column.lower()), next(column for column in base if column != target))
    category = next((column for column in base.select_dtypes(exclude="number") if column not in {target, "property_id"}), None)
    scenarios: dict[str, pd.DataFrame] = {
        "missing_target_column": base.drop(columns=[target]),
        "extra_column": base.assign(unexpected_payload="extra"),
        "wrong_feature_type": base.assign(**{feature: "not-a-number"}),
        "empty_strings": base.assign(**{feature: ""}),
        "missing_target_values": base.assign(**{target: np.where(np.arange(len(base)) % 2, base[target], np.nan)}),
        "multiple_targets": base.assign(target_copy=base[target]),
        "constant_target": base.assign(**{target: float(pd.to_numeric(base[target], errors="coerce").median())}),
        "all_null_column": base.assign(all_null_feature=np.nan),
        "all_null_target": base.assign(**{target: np.nan}),
        "target_strings": base.assign(**{target: "unknown"}),
        "negative_target": base.assign(**{target: -pd.to_numeric(base[target], errors="coerce").abs()}),
        "mixed_currency": base.assign(currency=np.where(np.arange(len(base)) % 2, "USD", "INR")),
        "inconsistent_units": base.assign(**{feature: np.where(np.arange(len(base)) % 2, base[feature].astype(str) + " sqft", base[feature].astype(str) + " sqm")}),
        "tiny_dataset": base.head(10),
    }
    if category:
        scenarios["high_cardinality_category"] = base.assign(**{category: [f"unique-{index}" for index in range(len(base))]})
    return scenarios
