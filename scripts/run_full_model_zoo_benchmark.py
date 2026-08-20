from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.generators import generate_dataset
from src.benchmark.runner import run_benchmark, save_benchmark


def main() -> None:
    output = Path("data") / "benchmarks"
    dataset = generate_dataset("Apartments", rows=200, seed=7)
    run = run_benchmark(
        dataset, mode="Comprehensive", selected_models=None,
        include_optional=True, include_ensembles=True, validation_strategy="random",
    )
    save_benchmark(run, output)
    status = run.results.groupby("status").size().to_dict()
    evidence = {
        "experiment_id": run.experiment_id,
        "dataset": "synthetic_apartments_seed7",
        "rows": 200,
        "status_counts": status,
        "selected_model": run.training_result.active_model_name,
        "automl_regret_percent": run.automl_regret_percent,
        "ensemble_decisions": run.training_result.metadata.get("ensemble_decisions", {}),
    }
    (output / "full_model_zoo_summary.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    print(run.results[["model", "model_family", "status", "failure_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
