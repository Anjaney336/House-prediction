import numpy as np
import pandas as pd

from src.models.model_catalog import assess_eligibility, model_catalog
from src.models.model_catalog import ModelSpec
from src.models.trainer import train_regressors
from src.models.validation_strategy import recommend_validation


def frame(rows=80):
    rng = np.random.default_rng(7)
    area = rng.normal(1200, 180, rows)
    locality = rng.choice(["A", "B", "C"], rows)
    price = area * 200 + (locality == "A") * 25000 + rng.normal(0, 5000, rows)
    return pd.DataFrame({"area": area, "locality": locality, "price": price})


def test_catalog_is_formal_and_dataset_aware():
    catalog = model_catalog()
    assert len(catalog) >= 25
    assert {"Linear", "Boosting", "Kernel", "Polynomial", "Probabilistic"}.issubset({item.family for item in catalog.values()})
    _, availability = assess_eligibility(frame(2000).drop(columns="price"), mode="Comprehensive")
    gaussian = availability.loc[availability["Model"] == "Gaussian Process"].iloc[0]
    assert gaussian["Status"] == "Excluded"
    assert "1,500" in gaussian["Reason"]


def test_repeated_property_ids_trigger_group_validation():
    data = frame(80)
    data["property_id"] = np.repeat(np.arange(40), 2)
    plan = recommend_validation(data, ["area", "locality"], folds=4)
    assert plan.name == "Grouped K-fold"
    assert plan.groups is not None


def test_balanced_lab_records_availability_folds_and_reproducibility():
    data = frame(70)
    result = train_regressors(
        data, "price", ["area", "locality"], cv_folds=3, tune=False,
        mode="Balanced", selected_models=["Linear Regression", "Ridge", "Decision Tree"],
        include_optional_boosters=False, metadata={"dataset_name": "unit-test"},
    )
    assert set(result.pipelines) == {"Linear Regression", "Ridge", "Decision Tree"}
    assert result.metadata["experiment_id"].startswith("exp-")
    assert result.metadata["validation_strategy"] == "Shuffled K-fold"
    assert len(result.fold_scores[result.active_model_name]) == 3
    assert {"Overall Score", "CV RMSE Std", "Family"}.issubset(result.leaderboard.columns)


def test_configurable_performance_only_ranking_selects_lowest_cv_rmse():
    data = frame(80)
    result = train_regressors(
        data, "price", ["area", "locality"], cv_folds=3, tune=False,
        ranking_weights={"performance": 1.0, "stability": 0.0, "simplicity": 0.0, "generalization": 0.0},
    )
    assert result.active_model_name == result.leaderboard.loc[result.leaderboard["CV RMSE"].idxmin(), "Model"]


def test_ensembles_are_rejected_without_material_validation_gain():
    data = frame(100)
    result = train_regressors(
        data, "price", ["area", "locality"], cv_folds=3, tune=False,
        mode="Comprehensive", selected_models=["Linear Regression", "Ridge", "Decision Tree"],
        include_ensembles=True, minimum_ensemble_improvement=1.0,
    )
    decisions = result.metadata["ensemble_decisions"]
    assert decisions
    assert not any(item["accepted"] for item in decisions.values())
    assert not any("Ensemble" in name or name == "Weighted Blend" for name in result.pipelines)


def test_one_failed_model_does_not_abort_automl(monkeypatch):
    import src.models.trainer as trainer

    broken = ModelSpec("Broken Model", "Test", lambda: (_ for _ in ()).throw(RuntimeError("injected failure")), ("Core",), {}, 1)
    ridge = model_catalog()["Ridge"]
    availability = pd.DataFrame([
        {"Model": "Broken Model", "Family": "Test", "Status": "Eligible", "Reason": "Fault injection", "Compute cost": "Low"},
        {"Model": "Ridge", "Family": "Linear", "Status": "Eligible", "Reason": "Control", "Compute cost": "Low"},
    ])
    monkeypatch.setattr(trainer, "assess_eligibility", lambda *args, **kwargs: ([broken, ridge], availability.copy()))
    result = trainer.train_regressors(frame(70), "price", ["area", "locality"], cv_folds=3, tune=False)
    assert list(result.pipelines) == ["Ridge"]
    assert result.availability.loc[result.availability.Model == "Broken Model", "Status"].iloc[0] == "Failed"
