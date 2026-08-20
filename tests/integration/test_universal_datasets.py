import pytest
import pandas as pd
from src.utils.config import ROOT_DIR

from src.benchmark.generators import GENERATORS, generate_dataset
from src.domain.domain_detector import analyze_domain
from src.domain.target_detector import detect_targets
from src.models.trainer import train_regressors
from src.utils.schema import suggest_features
from src.validation.schema_contract import build_schema_contract


@pytest.mark.parametrize("market_kind", list(GENERATORS))
def test_same_training_workflow_accepts_every_property_market(market_kind):
    dataset = generate_dataset(market_kind, rows=80, seed=31)
    frame = dataset.frame
    domain = analyze_domain(frame)
    target = detect_targets(frame)[0].column
    features, _ = suggest_features(frame, target)
    contract = build_schema_contract(frame, features, target, domain, market="SYNTHETIC-QA")
    result = train_regressors(
        frame, target, features, cv_folds=3, tune=False,
        selected_models=["Linear Regression", "Ridge"], mode="Core",
    )
    assert result.active_pipeline.predict(frame[features].head(2)).shape == (2,)
    assert set(contract.feature_order) == set(features)


def test_california_demo_remains_aggregate_and_predictable_without_defining_defaults():
    frame = pd.read_csv(ROOT_DIR / "data" / "sample_datasets" / "housing_sample.csv").head(600)
    domain = analyze_domain(frame)
    target = detect_targets(frame)[0].column
    features, _ = suggest_features(frame, target)
    result = train_regressors(
        frame, target, features, cv_folds=3, tune=False,
        selected_models=["Ridge", "Histogram Gradient Boosting"], mode="Core",
    )
    assert domain.granularity == "Census Block / Geographic Area"
    assert domain.prediction_label == "Estimated Area/Block Housing Value"
    assert result.active_pipeline.predict(frame[features].head(2)).shape == (2,)
