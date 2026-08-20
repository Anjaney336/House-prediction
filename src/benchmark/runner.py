from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.benchmark.corruption import inject_missingness
from src.benchmark.generators import SyntheticDataset
from src.features.leakage import detect_leakage
from src.models.trainer import TrainingResult, train_regressors
from src.utils.schema import infer_column_roles, suggest_features


BENCHMARK_COLUMNS = [
    "experiment_id", "dataset_type", "dataset_seed", "dataset_size", "corruption_level",
    "target", "model", "model_family", "validation_strategy", "cv_rmse", "cv_mae",
    "cv_r2", "cv_std", "holdout_rmse", "holdout_mae", "holdout_r2",
    "interval_coverage", "interval_width", "calibration_rows", "train_time",
    "prediction_time", "memory_estimate", "status", "failure_reason", "selected",
]


@dataclass
class BenchmarkRun:
    experiment_id: str
    results: pd.DataFrame
    manifest: dict
    training_result: TrainingResult
    automl_regret_percent: float
    ground_truth_recovery: pd.DataFrame


def _benchmark_id() -> str:
    return f"BENCH-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"


def _safe_features(dataset: SyntheticDataset) -> list[str]:
    features, _ = suggest_features(dataset.frame, dataset.target)
    datetime_columns = set(infer_column_roles(dataset.frame).datetime)
    high_leakage = {warning.column for warning in detect_leakage(dataset.frame, dataset.target, features) if warning.severity == "high"}
    return [column for column in features if column not in high_leakage and column not in datetime_columns]


def ground_truth_recovery(result: TrainingResult, expected: dict[str, float]) -> pd.DataFrame:
    if not expected:
        return pd.DataFrame(columns=["True Feature", "Expected Importance", "Observed Importance", "Observed Rank"])
    scored = permutation_importance(
        result.active_pipeline, result.X_test, result.y_test,
        scoring="neg_root_mean_squared_error", n_repeats=3, random_state=42, n_jobs=1,
    )
    observed = pd.Series(np.maximum(scored.importances_mean, 0), index=result.X_test.columns)
    if observed.sum() > 0:
        observed = observed / observed.sum()
    rank = observed.rank(ascending=False, method="min")
    rows = [{
        "True Feature": feature,
        "Expected Importance": weight,
        "Observed Importance": float(observed.get(feature, 0.0)),
        "Observed Rank": int(rank.get(feature, len(observed) + 1)),
    } for feature, weight in expected.items()]
    return pd.DataFrame(rows).sort_values("Expected Importance", ascending=False)


def run_benchmark(
    dataset: SyntheticDataset,
    mode: str = "Balanced",
    selected_models: list[str] | None = None,
    corruption_level: str = "clean",
    cv_folds: int = 3,
    include_optional: bool = False,
    include_ensembles: bool = False,
    validation_strategy: str = "auto",
    ranking_weights: dict[str, float] | None = None,
) -> BenchmarkRun:
    experiment_id, started = _benchmark_id(), time.perf_counter()
    features = _safe_features(dataset)
    result = train_regressors(
        dataset.frame, dataset.target, features, cv_folds=cv_folds, tune=False,
        mode=mode, selected_models=selected_models, include_optional_boosters=include_optional,
        include_ensembles=include_ensembles, validation_strategy=validation_strategy,
        ranking_weights=ranking_weights,
        metadata={"benchmark_id": experiment_id, "dataset_type": dataset.dataset_type, "dataset_seed": dataset.seed},
    )
    memory = int(dataset.frame.memory_usage(index=True, deep=True).sum())
    records = []
    for _, row in result.leaderboard.iterrows():
        records.append({
            "experiment_id": experiment_id, "dataset_type": dataset.dataset_type,
            "dataset_seed": dataset.seed, "dataset_size": len(dataset.frame),
            "corruption_level": corruption_level, "target": dataset.target,
            "model": row["Model"], "model_family": row["Family"],
            "validation_strategy": result.metadata["validation_strategy"],
            "cv_rmse": row["CV RMSE"], "cv_mae": row["CV MAE"], "cv_r2": row["CV R²"],
            "cv_std": row["CV RMSE Std"], "holdout_rmse": row["Test RMSE"],
            "holdout_mae": row["Test MAE"], "holdout_r2": row["Test R²"],
            "interval_coverage": row.get("95% Interval Coverage"),
            "interval_width": row.get("95% Interval Width"),
            "calibration_rows": result.metadata.get("calibration_rows"),
            "train_time": row["Training Time (s)"], "prediction_time": row["Prediction Time (ms/100 rows)"],
            "memory_estimate": memory, "status": "SUCCESS", "failure_reason": "",
            "selected": row["Model"] == result.active_model_name,
        })
    recorded_models = {record["model"] for record in records}
    for _, item in result.availability.iterrows():
        if item["Model"] in recorded_models or item["Status"] in {"Eligible"}:
            continue
        records.append({
            "experiment_id": experiment_id, "dataset_type": dataset.dataset_type,
            "dataset_seed": dataset.seed, "dataset_size": len(dataset.frame), "corruption_level": corruption_level,
            "target": dataset.target, "model": item["Model"], "model_family": item["Family"],
            "validation_strategy": result.metadata["validation_strategy"], "memory_estimate": memory,
            "status": str(item["Status"]).upper(), "failure_reason": item["Reason"], "selected": False,
            **{column: np.nan for column in [
                "cv_rmse", "cv_mae", "cv_r2", "cv_std", "holdout_rmse", "holdout_mae",
                "holdout_r2", "interval_coverage", "interval_width", "calibration_rows",
                "train_time", "prediction_time",
            ]},
        })
    results = pd.DataFrame(records).reindex(columns=BENCHMARK_COLUMNS)
    success = results.loc[results["status"] == "SUCCESS"]
    selected_rmse = float(success.loc[success["selected"], "holdout_rmse"].iloc[0])
    oracle_rmse = float(success["holdout_rmse"].min())
    regret = max(0.0, (selected_rmse - oracle_rmse) / max(oracle_rmse, 1e-9) * 100)
    recovery = ground_truth_recovery(result, dataset.ground_truth)
    manifest = {
        "experiment_id": experiment_id, "timestamp": datetime.now(timezone.utc).isoformat(),
        **dataset.manifest(), "features": features, "corruption_scenario": corruption_level,
        "models_tested": success["model"].tolist(), "validation_strategy": result.metadata["validation_strategy"],
        "uncertainty_method": result.metadata.get("uncertainty_method"),
        "calibration_rows": result.metadata.get("calibration_rows"),
        "auto_ml_regret_percent": regret, "total_runtime_seconds": time.perf_counter() - started,
        "software_versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    return BenchmarkRun(experiment_id, results, manifest, result, regret, recovery)


def robustness_curve(
    dataset: SyntheticDataset,
    levels: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
    selected_models: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    for level in levels:
        corrupted = inject_missingness(dataset.frame, level, seed=dataset.seed, mechanism="MAR")
        variant = SyntheticDataset(corrupted.frame, dataset.dataset_type, dataset.target, dataset.seed, dataset.parameters, dataset.ground_truth)
        run = run_benchmark(variant, selected_models=selected_models, corruption_level=f"MAR {level:.0%}")
        success = run.results.loc[run.results["status"] == "SUCCESS"].copy()
        success["missingness"] = level
        rows.append(success)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate_distribution_shift(result: TrainingResult, shifted_frame: pd.DataFrame, target: str) -> dict[str, float]:
    rows = shifted_frame.reindex(columns=result.X_test.columns)
    actual = pd.to_numeric(shifted_frame[target], errors="coerce")
    valid = actual.notna()
    prediction = result.active_pipeline.predict(rows.loc[valid])
    return {
        "rmse": float(mean_squared_error(actual.loc[valid], prediction) ** 0.5),
        "mae": float(mean_absolute_error(actual.loc[valid], prediction)),
        "r2": float(r2_score(actual.loc[valid], prediction)),
    }


def save_benchmark(run: BenchmarkRun, directory: str | Path) -> tuple[Path, Path]:
    output = Path(directory); output.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = output / f"{run.experiment_id}.csv", output / f"{run.experiment_id}.json"
    run.results.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(run.manifest, indent=2, default=str), encoding="utf-8")
    run.ground_truth_recovery.to_csv(output / f"{run.experiment_id}-ground-truth-recovery.csv", index=False)
    return csv_path, json_path


def ranking_sensitivity(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Re-rank one experiment under documented alternative weighting policies."""
    scenarios = {
        "60/20/10/10": (0.60, 0.20, 0.10, 0.10),
        "70/15/10/5": (0.70, 0.15, 0.10, 0.05),
        "50/25/10/15": (0.50, 0.25, 0.10, 0.15),
    }
    complexity = leaderboard["Complexity"].astype(float)
    simplicity = 1 - (complexity - complexity.min()) / max(complexity.max() - complexity.min(), 1e-9)
    gap = leaderboard["Overfit Gap"].astype(float)
    generalization = 1 - (gap - gap.min()) / max(gap.max() - gap.min(), 1e-9)
    rows = []
    for label, weights in scenarios.items():
        score = (
            leaderboard["Performance Score"] / 100 * weights[0]
            + leaderboard["Stability Score"] / 100 * weights[1]
            + simplicity * weights[2] + generalization * weights[3]
        )
        winner = int(score.idxmax())
        rows.append({"Weighting": label, "Selected Model": leaderboard.loc[winner, "Model"], "Score": float(score.loc[winner] * 100)})
    return pd.DataFrame(rows)
