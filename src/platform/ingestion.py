from __future__ import annotations

import re
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import DataLoadError, dataset_hash, load_csv
from src.data.profiler import data_quality_summary, profile_dataframe
from src.data.validator import detect_data_issues
from src.domain.domain_detector import analyze_domain
from src.domain.market_intelligence import detect_market
from src.domain.property_ontology import map_column, normalize_name
from src.domain.target_detector import detect_targets
from src.features.leakage import detect_leakage
from src.validation.schema_contract import build_schema_contract


AREA_UNITS = {
    "sqft": ("sqft", "sq_ft", "square_feet", "square_foot"),
    "sqm": ("sqm", "sq_m", "square_meter", "square_metre"),
    "marla": ("marla",),
    "bigha": ("bigha",),
    "acre": ("acre",),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def load_tabular(content: bytes, filename: str, max_size_mb: int = 100) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return load_csv(content, max_size_mb=max_size_mb)
    if suffix in {".xlsx", ".xls"}:
        if len(content) > max_size_mb * 1024 * 1024:
            raise DataLoadError(f"Spreadsheet exceeds the {max_size_mb} MB upload limit.")
        try:
            frame = pd.read_excel(BytesIO(content))
        except ImportError as exc:
            raise DataLoadError("Excel upload support requires the optional openpyxl dependency.") from exc
        except Exception as exc:
            raise DataLoadError(f"Could not read this spreadsheet: {exc}") from exc
        if frame.empty or len(frame.columns) < 2:
            raise DataLoadError("The spreadsheet must contain rows and at least two columns.")
        return frame
    raise DataLoadError("Supported upload formats are CSV and XLSX.")


def infer_unit(column: str) -> str | None:
    normalized = normalize_name(column)
    for unit, aliases in AREA_UNITS.items():
        if any(alias in normalized for alias in aliases):
            return unit
    return None


def _normalization_suggestions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    suggestions = []
    for column in frame.select_dtypes(exclude="number").columns:
        entry = map_column(str(column))
        if not entry or entry.role not in {"city", "locality", "sector", "pincode"}:
            continue
        values = frame[column].dropna().astype(str)
        groups: dict[str, set[str]] = {}
        for value in values.unique()[:1000]:
            canonical = re.sub(r"[^a-z0-9]+", "", value.casefold().replace("sector", "sec"))
            groups.setdefault(canonical, set()).add(value)
        variants = [sorted(group) for group in groups.values() if len(group) > 1]
        if variants:
            suggestions.append({"column": str(column), "variant_groups": variants[:20], "action": "Review and approve a canonical mapping; no values were merged automatically."})
    return suggestions


def profile_contract(frame: pd.DataFrame, source_name: str | None = None) -> dict[str, Any]:
    domain = analyze_domain(frame)
    market = detect_market(frame, source_name)
    targets = detect_targets(frame)
    primary_target = targets[0].column if targets else None
    candidate_features = [str(column) for column in frame.columns if str(column) != primary_target]
    leakage = detect_leakage(frame, primary_target, candidate_features) if primary_target else []
    basic_contract = None
    if primary_target:
        basic_contract = build_schema_contract(frame, candidate_features, primary_target, domain).to_dict()
    columns = []
    profile = profile_dataframe(frame).set_index("column").to_dict(orient="index")
    for column in frame.columns:
        mapped = map_column(str(column))
        columns.append({
            "name": str(column),
            "semantic_role": mapped.role if mapped else "other",
            "group": mapped.group if mapped else "Other",
            "unit": infer_unit(str(column)),
            **profile[str(column)],
        })
    return _json_safe({
        "version": "2.0",
        "dataset_id": dataset_hash(frame),
        "domain": {
            "classification": domain.domain,
            "confidence": domain.confidence,
            "scores": domain.scores,
            "asset_type": domain.asset_type,
            "granularity": domain.granularity,
            "prediction_label": domain.prediction_label,
            "rationale": list(domain.rationale),
        },
        "market_hypothesis": market.to_dict(),
        "target_candidates": [candidate.__dict__ for candidate in targets],
        "operator_confirmation_required": len(targets) != 1,
        "columns": columns,
        "quality": data_quality_summary(frame),
        "issues": [issue.to_dict() for issue in detect_data_issues(frame)],
        "leakage_candidates": [warning.__dict__ for warning in leakage],
        "normalization_suggestions": _normalization_suggestions(frame),
        "model_schema_preview": basic_contract,
    })
