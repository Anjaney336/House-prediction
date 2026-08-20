from __future__ import annotations

import numpy as np
import pandas as pd


def model_confidence_score(
    test_r2: float,
    mape_percent: float,
    input_completeness: float,
    quality_score: int,
    similarity_score: float | None = None,
    stability_score: float | None = None,
    generalization_score: float | None = None,
) -> tuple[int, str, dict[str, str]]:
    """Create a transparent heuristic quality score, not a probability."""
    accuracy = np.clip((test_r2 + 0.2) / 1.2, 0, 1) * 0.55 + np.clip(1 - mape_percent / 100, 0, 1) * 0.45
    similarity = 0.6 if similarity_score is None else float(np.clip(similarity_score, 0, 1))
    stability = 0.7 if stability_score is None else float(np.clip(stability_score, 0, 1))
    generalization = 0.7 if generalization_score is None else float(np.clip(generalization_score, 0, 1))
    score = int(round(100 * (0.28 * accuracy + 0.17 * stability + 0.12 * generalization + 0.16 * input_completeness + 0.17 * quality_score / 100 + 0.10 * similarity)))
    label = "High" if score >= 80 else "Moderate" if score >= 60 else "Low"
    factors = {
        "Data completeness": "High" if input_completeness >= 0.9 else "Medium" if input_completeness >= 0.7 else "Low",
        "Model accuracy": "High" if accuracy >= 0.8 else "Medium" if accuracy >= 0.6 else "Low",
        "Dataset quality": "High" if quality_score >= 80 else "Medium" if quality_score >= 60 else "Low",
        "Comparable coverage": "High" if similarity >= 0.8 else "Medium" if similarity >= 0.55 else "Low",
        "Validation stability": "High" if stability >= 0.85 else "Medium" if stability >= 0.65 else "Low",
        "Generalization": "High" if generalization >= 0.85 else "Medium" if generalization >= 0.65 else "Low",
    }
    return score, label, factors


def coverage_by_group(
    y_true,
    prediction,
    radius: float,
    groups,
    minimum_rows: int = 5,
) -> pd.DataFrame:
    """Audit fixed-radius interval coverage across observed test subgroups."""
    frame = pd.DataFrame({
        "actual": np.asarray(y_true, dtype=float),
        "prediction": np.asarray(prediction, dtype=float),
        "group": pd.Series(groups).astype("string").fillna("Missing").to_numpy(),
    })
    frame["covered"] = (frame["actual"] - frame["prediction"]).abs() <= radius
    summary = frame.groupby("group", dropna=False)["covered"].agg(["mean", "count"]).reset_index()
    summary = summary.loc[summary["count"] >= minimum_rows].rename(columns={"mean": "coverage", "count": "rows"})
    return summary.sort_values("coverage")
