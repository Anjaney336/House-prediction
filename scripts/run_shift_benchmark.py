from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmark.corruption import apply_market_shift
from src.benchmark.generators import GENERATORS, generate_dataset
from src.benchmark.runner import evaluate_distribution_shift, run_benchmark


def main() -> None:
    output = Path("data") / "benchmarks"
    summary = pd.read_csv(output / "final_validation_matrix.csv")
    rows = []
    for kind in GENERATORS:
        dataset = generate_dataset(kind, rows=500, seed=42)
        dataset_name = dataset.dataset_type
        model = str(summary.loc[summary["Dataset"] == dataset_name, "Best Model"].iloc[0])
        run = run_benchmark(dataset, mode="Comprehensive", selected_models=[model], validation_strategy="auto")
        baseline = evaluate_distribution_shift(run.training_result, dataset.frame, dataset.target)
        shifted_frame = apply_market_shift(dataset.frame, dataset.target, shift=0.20).frame
        shifted = evaluate_distribution_shift(run.training_result, shifted_frame, dataset.target)
        rows.append({
            "Dataset": dataset_name, "Model": model,
            "In-distribution RMSE": baseline["rmse"], "Shifted-market RMSE": shifted["rmse"],
            "Degradation (%)": (shifted["rmse"] / baseline["rmse"] - 1) * 100,
        })
    results = pd.DataFrame(rows)
    results.to_csv(output / "distribution_shift_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
