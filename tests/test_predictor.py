import pandas as pd

from src.data.cleaner import CleaningConfig
from src.models.predictor import predict_batch, prediction_interval, similar_rows
from src.models.trainer import conformal_radius, train_regressors
from src.domain.domain_detector import analyze_domain
from tests.test_trainer import make_regression_frame
from src.validation.schema_contract import build_schema_contract
from src.models.uncertainty import coverage_by_group


def test_single_batch_interval_and_neighbors():
    frame = make_regression_frame()
    features = ["size", "bedrooms", "location"]
    result = train_regressors(
        frame,
        target="price",
        features=features,
        cleaning_config=CleaningConfig(),
        cv_folds=3,
        tune=False,
    )
    row = pd.DataFrame([{"size": 1600.0, "bedrooms": 3, "location": "North"}])
    prediction = predict_batch(result.active_pipeline, row, features)
    low, high = prediction_interval(float(prediction[0]), result.residual_std[result.active_model_name])
    neighbors = similar_rows(result.active_pipeline, result.X_train, row, n=4)
    assert len(prediction) == 1
    assert low < prediction[0] < high
    assert len(neighbors) == 4


def test_batch_schema_validation_is_clear():
    frame = make_regression_frame()
    result = train_regressors(frame, "price", ["size", "location"], cv_folds=3, tune=False)
    try:
        predict_batch(result.active_pipeline, pd.DataFrame({"size": [1000]}), ["size", "location"])
    except ValueError as exc:
        assert "location" in str(exc)
    else:
        raise AssertionError("Expected missing-schema validation error")


def test_contract_allows_optional_missing_prediction_input():
    frame = make_regression_frame()
    features = ["size", "bedrooms", "location"]
    result = train_regressors(frame, "price", features, cv_folds=3, tune=False)
    contract = build_schema_contract(frame, features, "price", analyze_domain(frame))
    partial = pd.DataFrame([{"size": 1200.0, "bedrooms": 2}])
    prediction = predict_batch(result.active_pipeline, partial, features, contract)
    assert len(prediction) == 1


def test_conformal_calibration_is_disjoint_and_finite_sample_correct():
    frame = make_regression_frame(120)
    result = train_regressors(frame, "price", ["size", "bedrooms", "location"], cv_folds=3, tune=False)
    name = result.active_model_name
    assert set(result.X_train.index).isdisjoint(result.X_calibration.index)
    assert set(result.X_train.index).isdisjoint(result.X_test.index)
    assert set(result.X_calibration.index).isdisjoint(result.X_test.index)
    assert result.conformal_radius[name] > 0
    assert 0 <= result.conformal_coverage[name] <= 1
    prediction = float(result.test_predictions[name][0])
    low, high = prediction_interval(prediction, calibrated_radius=result.conformal_radius[name])
    assert low == prediction - result.conformal_radius[name]
    assert high == prediction + result.conformal_radius[name]
    assert conformal_radius([0, 0, 0, 0], [1, 2, 3, 4], confidence=0.75) == 4


def test_legacy_residual_interval_remains_backward_compatible():
    assert prediction_interval(100.0, residual_std=10.0) == (80.4, 119.6)


def test_interval_coverage_can_be_audited_by_subgroup():
    audit = coverage_by_group(
        [100, 110, 200, 220], [105, 120, 205, 260], radius=15,
        groups=["A", "A", "B", "B"], minimum_rows=2,
    )
    assert dict(zip(audit.group, audit.coverage)) == {"B": 0.5, "A": 1.0}
