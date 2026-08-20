from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.utils.formatting import format_value
from src.validation.schema_contract import ModelSchemaContract


def build_valuation_report(
    row: pd.DataFrame,
    prediction: float,
    interval: tuple[float, float],
    confidence: tuple[int, str, dict[str, str]],
    contract: ModelSchemaContract,
    model_name: str,
    model_id: str,
    metrics: dict,
    quality_score: int,
    warnings: list[str],
    interval_method: str = "Model-based residual range",
) -> str:
    score, label, factors = confidence
    inputs = "\n".join(f"- **{column}:** {value}" for column, value in row.iloc[0].items())
    factor_text = "\n".join(f"- {name}: {value}" for name, value in factors.items())
    warning_text = "\n".join(f"- {warning}" for warning in warnings) or "- No additional warnings."
    return f"""# PricePredict AI Valuation Report

- Generated: {datetime.now(timezone.utc).isoformat()}
- Model ID: {model_id}
- Model: {model_name}
- Asset type: {contract.asset_type}
- Prediction granularity: {contract.prediction_granularity}

## {contract.prediction_label}

**{format_value(prediction, contract.currency)}**

{interval_method}: **{format_value(interval[0], contract.currency)} – {format_value(interval[1], contract.currency)}**

Model Confidence Score: **{score}/100 — {label}**

{factor_text}

## Property / observation inputs

{inputs}

## Model evidence

- Holdout R²: {metrics.get('Test R²', 0):.3f}
- Holdout RMSE: {metrics.get('Test RMSE', 0):,.2f}
- Dataset quality: {quality_score}/100

## Important warnings

{warning_text}

This estimate is data-driven and depends on historical dataset quality. It is not a guaranteed market price, legal valuation, or substitute for a licensed appraisal where one is required.
"""
