import numpy as np
import pandas as pd
import pytest

from src.benchmark.corruption import inject_duplicates, inject_leakage, inject_missingness, inject_outliers
from src.benchmark.faults import audit_faults, fault_scenarios
from src.benchmark.generators import GENERATORS, generate_dataset
from src.benchmark.runner import run_benchmark
from src.domain.domain_detector import analyze_domain
from src.domain.target_detector import detect_targets
from src.features.leakage import detect_leakage


@pytest.mark.parametrize("kind", list(GENERATORS))
def test_all_market_generators_are_reproducible_and_semantically_detected(kind):
    first = generate_dataset(kind, rows=80, seed=7)
    second = generate_dataset(kind, rows=80, seed=7)
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.target in first.frame
    assert len(first.ground_truth) >= 5
    domain = analyze_domain(first.frame)
    assert domain.is_real_estate
    expected_asset = {"Land / Plots": "Land", "Commercial": "Commercial", "Rentals": "Rental"}.get(kind, "Residential")
    assert domain.asset_type == expected_asset
    assert detect_targets(first.frame)[0].column == first.target


def test_corruption_scenarios_preserve_source_and_are_detectable():
    dataset = generate_dataset("Apartments", rows=120, seed=42)
    source = dataset.frame.copy(deep=True)
    missing = inject_missingness(source, 0.1, mechanism="MAR", seed=1)
    impossible = inject_outliers(source, 0.05, impossible=True, seed=7)
    duplicates = inject_duplicates(source, 0.1, seed=21)
    leakage = inject_leakage(source, dataset.target)
    pd.testing.assert_frame_equal(dataset.frame, source)
    assert missing.frame.isna().sum().sum() > 0
    assert any(finding.severity == "HIGH" for finding in audit_faults(impossible.frame, dataset.target))
    assert duplicates.frame.duplicated().sum() > 0
    warnings = detect_leakage(leakage.frame, dataset.target, [column for column in leakage.frame if column != dataset.target])
    flagged = {warning.column for warning in warnings if warning.severity == "high"}
    assert {"price_per_sqft", "future_sale_price", "target_copy", "near_perfect_proxy"}.issubset(flagged)


def test_fault_levels_include_blocker_and_recovery_action():
    tiny = pd.DataFrame({"area_sqft": [1000, 1200], "sale_price": [np.nan, np.nan]})
    findings = audit_faults(tiny, "sale_price")
    assert any(item.severity == "BLOCKER" for item in findings)
    assert all(item.recommended_action for item in findings)


def test_fault_scenario_matrix_never_crashes_diagnostics():
    dataset = generate_dataset("Apartments", rows=80, seed=7)
    scenarios = fault_scenarios(dataset.frame, dataset.target)
    assert len(scenarios) >= 14
    for name, scenario in scenarios.items():
        findings = audit_faults(scenario, dataset.target)
        assert findings, name


def test_benchmark_records_manifest_regret_and_ground_truth():
    dataset = generate_dataset("Apartments", rows=120, seed=21)
    run = run_benchmark(
        dataset, selected_models=["Linear Regression", "Ridge", "Decision Tree"],
        validation_strategy="random",
    )
    successes = run.results.loc[run.results.status == "SUCCESS"]
    assert len(successes) == 3
    assert successes.selected.sum() == 1
    assert run.automl_regret_percent >= 0
    assert run.manifest["seed"] == 21
    assert run.experiment_id.startswith("BENCH-")
    assert not run.ground_truth_recovery.empty


def test_temporal_and_geographic_validation_are_explicit():
    dataset = generate_dataset("Apartments", rows=120, seed=100)
    from src.models.validation_strategy import recommend_validation

    temporal = recommend_validation(dataset.frame, ["area_sqft", "locality"], 3, strategy="auto")
    geographic = recommend_validation(dataset.frame, ["area_sqft", "locality"], 3, strategy="geographic", group_column="city")
    assert temporal.name == "Time-series split"
    assert geographic.name == "Geographic GroupKFold"


def test_geographic_training_reduces_cv_folds_after_group_holdouts():
    dataset = generate_dataset("Apartments", rows=120, seed=100)
    run = run_benchmark(
        dataset,
        selected_models=["Linear Regression", "Ridge"],
        validation_strategy="geographic",
    )
    metadata = run.training_result.metadata
    assert metadata["validation_strategy"] == "Geographic GroupKFold"
    config = metadata["training_config"]
    assert config["cv_folds_applied"] <= config["cv_folds_requested"]
    assert len(run.training_result.X_calibration) > 0
