from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmark.generators import generate_dataset
from src.benchmark.runner import ranking_sensitivity, run_benchmark


MODELS = ["Linear Regression", "Ridge", "Decision Tree", "Random Forest", "Gradient Boosting", "Histogram Gradient Boosting"]


def main() -> None:
    rows, policies = [], []
    for seed in [1, 7, 21, 42, 100, 999]:
        dataset = generate_dataset("Mixed Residential", rows=500, seed=seed)
        run = run_benchmark(dataset, mode="Comprehensive", selected_models=MODELS, validation_strategy="auto")
        success = run.results.loc[run.results.status == "SUCCESS"]
        selected = success.loc[success.selected].iloc[0]
        oracle = success.loc[success.holdout_rmse.idxmin()]
        rows.append({
            "Seed": seed, "Selected": selected.model, "Oracle": oracle.model,
            "Selected Holdout RMSE": selected.holdout_rmse, "Oracle Holdout RMSE": oracle.holdout_rmse,
            "AutoML Regret (%)": run.automl_regret_percent,
            "Interval Coverage": selected.interval_coverage,
        })
        oracle_rmse = float(oracle.holdout_rmse)
        for _, policy in ranking_sensitivity(run.training_result.leaderboard).iterrows():
            policy_model = policy["Selected Model"]
            policy_rmse = float(success.loc[success.model == policy_model, "holdout_rmse"].iloc[0])
            policies.append({"Seed": seed, "Weighting": policy["Weighting"], "Selected": policy_model, "Regret (%)": max(0.0, (policy_rmse / oracle_rmse - 1) * 100)})
    evidence = pd.DataFrame(rows)
    evidence.to_csv(Path("data") / "benchmarks" / "multiseed_selection_validation.csv", index=False)
    policy_evidence = pd.DataFrame(policies)
    policy_evidence.to_csv(Path("data") / "benchmarks" / "ranking_policy_validation.csv", index=False)
    print(evidence.to_string(index=False))
    print("\nRegret mean/median/max:", evidence["AutoML Regret (%)"].mean(), evidence["AutoML Regret (%)"].median(), evidence["AutoML Regret (%)"].max())
    print("Coverage mean:", evidence["Interval Coverage"].mean())
    print("\nRANKING POLICY REGRET\n", policy_evidence.groupby("Weighting")["Regret (%)"].agg(["mean", "median", "max"]).to_string())


if __name__ == "__main__":
    main()
