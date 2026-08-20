from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.benchmark.corruption import apply_market_shift
from src.benchmark.generators import GENERATORS, SyntheticDataset, generate_dataset
from src.benchmark.runner import evaluate_distribution_shift, run_benchmark, save_benchmark
from src.data.quality import assess_quality
from src.domain.domain_detector import analyze_domain
from src.features.leakage import detect_leakage
from src.utils.schema import suggest_features


OUTPUT = Path("data") / "benchmarks"
MODELS = ["Linear Regression", "Ridge", "Decision Tree", "Random Forest", "Gradient Boosting", "Histogram Gradient Boosting"]


def quality_score(dataset: SyntheticDataset) -> int:
    features, _ = suggest_features(dataset.frame, dataset.target)
    domain = analyze_domain(dataset.frame)
    leakage = detect_leakage(dataset.frame, dataset.target, features)
    return assess_quality(dataset.frame, dataset.target, features, domain, leakage).overall


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    california_frame = pd.read_csv("data/sample_datasets/housing_sample.csv")
    datasets = [SyntheticDataset(california_frame, "california_block", "median_house_value", 42, {"source": "bundled California Housing"}, {})]
    datasets.extend(generate_dataset(kind, rows=500, seed=42) for kind in GENERATORS)
    summaries, shift_rows = [], []
    for dataset in datasets:
        run = run_benchmark(dataset, mode="Comprehensive", selected_models=MODELS, validation_strategy="auto")
        save_benchmark(run, OUTPUT)
        success = run.results.loc[run.results.status == "SUCCESS"]
        winner = success.loc[success.selected].iloc[0]
        summaries.append({
            "Dataset": dataset.dataset_type, "Best Model": winner.model,
            "CV RMSE": winner.cv_rmse, "Holdout RMSE": winner.holdout_rmse,
            "R²": winner.holdout_r2, "Stability": winner.cv_std,
            "Runtime": run.manifest["total_runtime_seconds"],
            "AutoML Regret (%)": run.automl_regret_percent,
            "Data Quality": quality_score(dataset),
            "Validation": run.manifest["validation_strategy"],
        })
        if dataset.dataset_type != "california_block":
            baseline_metrics = evaluate_distribution_shift(run.training_result, dataset.frame, dataset.target)
            shifted = apply_market_shift(dataset.frame, dataset.target, shift=0.20).frame
            shifted_metrics = evaluate_distribution_shift(run.training_result, shifted, dataset.target)
            baseline_rmse = baseline_metrics["rmse"]
            shift_rows.append({"Dataset": dataset.dataset_type, "In-distribution RMSE": baseline_rmse, "Shifted-market RMSE": shifted_metrics["rmse"], "Degradation (%)": (shifted_metrics["rmse"] / baseline_rmse - 1) * 100})
    summary = pd.DataFrame(summaries)
    shifts = pd.DataFrame(shift_rows)
    summary.to_csv(OUTPUT / "final_validation_matrix.csv", index=False)
    shifts.to_csv(OUTPUT / "distribution_shift_results.csv", index=False)
    (OUTPUT / "final_validation_manifest.json").write_text(json.dumps({"models": MODELS, "datasets": [dataset.manifest() for dataset in datasets]}, indent=2, default=str), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nDISTRIBUTION SHIFT\n", shifts.to_string(index=False))


if __name__ == "__main__":
    main()
