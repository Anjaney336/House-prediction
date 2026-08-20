from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorruptionResult:
    frame: pd.DataFrame
    scenario: str
    parameters: dict
    affected_rows: int


def inject_missingness(
    frame: pd.DataFrame,
    percentage: float,
    seed: int = 42,
    mechanism: str = "MCAR",
    protected: tuple[str, ...] = ("sale_price", "monthly_rent", "property_id"),
) -> CorruptionResult:
    if not 0 <= percentage <= 0.8:
        raise ValueError("Missingness percentage must be between 0 and 0.8.")
    rng, output = np.random.default_rng(seed), frame.copy(deep=True)
    candidates = [column for column in output.columns if column not in protected]
    if not candidates or percentage == 0:
        return CorruptionResult(output, f"missingness_{mechanism.lower()}", {"percentage": percentage, "seed": seed}, 0)
    affected: set[int] = set()
    mechanism = mechanism.upper()
    for column in candidates:
        probability = np.full(len(output), percentage)
        if mechanism == "MAR":
            numeric = output.select_dtypes(include="number").drop(columns=[column], errors="ignore")
            if not numeric.empty:
                driver = numeric.iloc[:, 0].rank(pct=True).fillna(0.5).to_numpy()
                probability = np.clip(percentage * (0.35 + 1.3 * driver), 0, 0.95)
        elif mechanism == "FEATURE_DEPENDENT":
            series = output[column]
            if pd.api.types.is_numeric_dtype(series):
                driver = series.rank(pct=True).fillna(0.5).to_numpy()
            else:
                frequencies = series.map(series.value_counts(normalize=True)).fillna(0).to_numpy()
                driver = 1 - frequencies
            probability = np.clip(percentage * (0.3 + 1.5 * driver), 0, 0.95)
        elif mechanism != "MCAR":
            raise ValueError("Mechanism must be MCAR, MAR, or FEATURE_DEPENDENT.")
        mask = rng.random(len(output)) < probability
        output.loc[mask, column] = np.nan
        affected.update(np.flatnonzero(mask).tolist())
    return CorruptionResult(output, f"missingness_{mechanism.lower()}", {"percentage": percentage, "seed": seed}, len(affected))


def inject_outliers(frame: pd.DataFrame, percentage: float = 0.02, seed: int = 42, impossible: bool = False) -> CorruptionResult:
    rng, output = np.random.default_rng(seed), frame.copy(deep=True)
    count = max(1, int(len(output) * percentage)) if percentage else 0
    indexes = rng.choice(output.index, count, replace=False) if count else []
    numeric = output.select_dtypes(include="number").columns.tolist()
    targets = [column for column in numeric if any(token in column.lower() for token in ("area", "price", "rent", "bedroom", "floor", "latitude", "longitude"))]
    for position, index in enumerate(indexes):
        if not targets:
            break
        column = targets[position % len(targets)]
        if impossible:
            if "latitude" in column:
                output.at[index, column] = 145
            elif "longitude" in column:
                output.at[index, column] = 260
            else:
                output.at[index, column] = -abs(float(pd.to_numeric(output[column], errors="coerce").median() or 1))
        else:
            output.at[index, column] = float(pd.to_numeric(output[column], errors="coerce").quantile(0.99)) * 8
    scenario = "impossible_values" if impossible else "valid_extremes"
    return CorruptionResult(output, scenario, {"percentage": percentage, "seed": seed}, count)


def inject_duplicates(frame: pd.DataFrame, percentage: float = 0.05, seed: int = 42, near: bool = False) -> CorruptionResult:
    rng, output = np.random.default_rng(seed), frame.copy(deep=True)
    count = max(1, int(len(output) * percentage)) if percentage else 0
    additions = output.sample(count, random_state=seed).copy()
    if near:
        numeric = [column for column in additions.select_dtypes("number") if "price" not in column and "rent" not in column]
        if numeric:
            additions[numeric[0]] = additions[numeric[0]] * rng.normal(1, 0.003, count)
        if "property_id" in additions:
            additions["property_id"] = additions["property_id"].astype(str) + "-RELIST"
    output = pd.concat([output, additions], ignore_index=True)
    return CorruptionResult(output, "near_duplicate_listings" if near else "exact_duplicates", {"percentage": percentage, "seed": seed}, count)


def inject_leakage(frame: pd.DataFrame, target: str) -> CorruptionResult:
    output = frame.copy(deep=True)
    area_candidates = [column for column in output.columns if "area" in column.lower() and pd.api.types.is_numeric_dtype(output[column])]
    denominator = output[area_candidates[0]].replace(0, np.nan) if area_candidates else 1.0
    output["price_per_sqft"] = output[target] / denominator
    output["future_sale_price"] = output[target] * 1.03
    output["post_sale_status"] = "Closed"
    output["target_copy"] = output[target]
    output["near_perfect_proxy"] = output[target] * np.random.default_rng(42).normal(1, 0.001, len(output))
    return CorruptionResult(output, "target_leakage", {"target": target}, len(output))


def apply_market_shift(frame: pd.DataFrame, target: str, shift: float = 0.25, city: str | None = None) -> CorruptionResult:
    output = frame.copy(deep=True)
    mask = pd.Series(True, index=output.index) if city is None or "city" not in output else output["city"].eq(city)
    output.loc[mask, target] = output.loc[mask, target] * (1 + shift)
    return CorruptionResult(output, "market_shift", {"shift": shift, "city": city}, int(mask.sum()))
