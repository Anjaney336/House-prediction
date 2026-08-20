from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.generators import generate_dataset
from src.benchmark.runner import run_benchmark, save_benchmark


def main() -> None:
    output = Path("data") / "benchmarks"
    dataset = generate_dataset("Apartments", rows=500, seed=42)
    models = ["Linear Regression", "Ridge", "Decision Tree", "Random Forest", "Gradient Boosting", "Histogram Gradient Boosting"]
    run = run_benchmark(dataset, mode="Comprehensive", selected_models=models, validation_strategy="auto")
    save_benchmark(run, output)
    recovery = run.ground_truth_recovery.copy()
    recovery["Expected Rank"] = recovery["Expected Importance"].rank(ascending=False, method="min")
    rank_correlation = float(recovery["Expected Rank"].corr(recovery["Observed Rank"], method="spearman"))
    expected_top3 = set(recovery.nsmallest(3, "Expected Rank")["True Feature"])
    observed_top3 = set(recovery.nsmallest(3, "Observed Rank")["True Feature"])
    summary = {
        "experiment_id": run.experiment_id,
        "selected_model": run.training_result.active_model_name,
        "spearman_rank_correlation": rank_correlation,
        "top_3_driver_overlap": len(expected_top3 & observed_top3),
        "expected_top_3": sorted(expected_top3),
        "observed_top_3": sorted(observed_top3),
        "scope": "Synthetic scenario evidence only; not proof of real-world causal explanations.",
    }
    (output / "explainability_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(recovery.to_string(index=False))


if __name__ == "__main__":
    main()
