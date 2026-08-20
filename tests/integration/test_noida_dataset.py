from pathlib import Path

import pandas as pd

from src.domain.domain_detector import analyze_domain
from src.domain.market_intelligence import detect_market
from src.domain.target_detector import detect_targets
from src.models.ood import assess_ood
from src.models.predictor import predict_batch
from src.models.trainer import train_regressors
from src.platform.ingestion import profile_contract
from src.utils.schema import suggest_features
from src.validation.schema_contract import build_schema_contract


NOIDA_DATASET = Path.home() / "Downloads" / "noida_real_estate_synthetic.csv"
CALIFORNIA_ONLY = {
    "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income", "ocean_proximity",
}


def test_supplied_noida_dataset_completes_universal_model_workflow():
    assert NOIDA_DATASET.exists(), "The mandatory supplied Noida dataset is unavailable."
    frame = pd.read_csv(NOIDA_DATASET)
    domain = analyze_domain(frame)
    profile = profile_contract(frame, NOIDA_DATASET.name)
    targets = detect_targets(frame)
    assert domain.is_real_estate and domain.domain == "REAL_ESTATE"
    assert domain.granularity == "Property Listing"
    assert {"Apartment", "Independent Floor", "Villa"}.issubset(domain.property_types)
    assert profile["market_hypothesis"]["candidate"] == "NOIDA"
    assert profile["market_hypothesis"]["requires_confirmation"]
    assert targets[0].column == "price_inr" and targets[0].currency_hypothesis == "INR"

    features, excluded = suggest_features(frame, "price_inr")
    assert "listing_id" in excluded
    assert not (CALIFORNIA_ONLY & set(features))
    contract = build_schema_contract(
        frame, features, "price_inr", domain, currency="INR", market="NOIDA",
        transaction_type="Sale", dataset_fingerprint=profile["dataset_id"],
    )
    result = train_regressors(
        frame, "price_inr", features, cv_folds=3, tune=False,
        selected_models=["Ridge", "Histogram Gradient Boosting"], mode="Core",
        metadata={"schema_contract": contract.to_dict()},
    )
    single = predict_batch(result.active_pipeline, frame[features].head(1), features, contract)
    batch = predict_batch(result.active_pipeline, frame[features].head(8), features, contract)
    assert len(single) == 1 and single[0] > 0
    assert len(batch) == 8 and (batch > 0).all()
    assert list(contract.feature_order) == features
    incompatible = frame[features].iloc[0].to_dict()
    incompatible["property_type"] = "Warehouse"
    assert not assess_ood(incompatible, contract).compatible
