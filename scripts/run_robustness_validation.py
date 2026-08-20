from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmark.generators import generate_dataset
from src.benchmark.runner import robustness_curve, run_benchmark


def main() -> None:
    output = Path("data") / "benchmarks"; output.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset("Apartments", rows=500, seed=42)
    curve = robustness_curve(dataset, levels=(0.0, 0.05, 0.10, 0.20), selected_models=["Ridge", "Gradient Boosting"])
    curve.to_csv(output / "missingness_robustness_curve.csv", index=False)

    comparisons = []
    for strategy in ["random", "auto", "geographic"]:
        run = run_benchmark(
            dataset, mode="Comprehensive", selected_models=["Gradient Boosting"],
            validation_strategy=strategy,
        )
        row = run.results.loc[run.results.status == "SUCCESS"].iloc[0]
        comparisons.append({"Requested Strategy": strategy, "Applied Strategy": row.validation_strategy, "CV RMSE": row.cv_rmse, "Holdout RMSE": row.holdout_rmse, "CV Std": row.cv_std})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "validation_strategy_comparison.csv", index=False)
    print("MISSINGNESS ROBUSTNESS\n", curve[["model", "missingness", "cv_rmse", "holdout_rmse"]].to_string(index=False))
    print("\nVALIDATION COMPARISON\n", comparison.to_string(index=False))


if __name__ == "__main__":
    main()
