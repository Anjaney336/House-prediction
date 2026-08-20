import pandas as pd

from src.domain.domain_detector import analyze_domain
from src.domain.target_detector import detect_targets
from src.models.predictor import predict_batch
from src.models.trainer import train_regressors
from src.utils.config import ROOT_DIR
from src.validation.prediction_validator import validate_and_prepare
from src.validation.schema_contract import build_schema_contract


SAMPLES = ROOT_DIR / "data" / "sample_datasets"


def test_individual_property_single_and_batch_prediction_contract():
    frame = pd.read_csv(SAMPLES / "individual_property_sample.csv")
    domain = analyze_domain(frame)
    target = detect_targets(frame)[0].column
    features = [column for column in frame.columns if column not in {target, "property_type"}]
    contract = build_schema_contract(frame, features, target, domain, currency="INR")
    result = train_regressors(frame, target, features, cv_folds=3, tune=False, metadata={"schema_contract": contract})

    partial = pd.DataFrame([{"city": "Pune", "locality": "Baner", "area_sqft": 1200, "bedrooms": 3, "bathrooms": 2, "floor": 4, "age": 6, "furnishing": "Semi-Furnished"}])
    validation = validate_and_prepare(partial, contract)
    assert validation.missing_columns == ("parking",)
    single = predict_batch(result.active_pipeline, validation.prepared, features, contract)
    assert len(single) == 1 and single[0] > 0

    batch = frame[features].head(4).copy()
    batch["ignored_note"] = "portfolio"
    batch_validation = validate_and_prepare(batch, contract)
    predictions = predict_batch(result.active_pipeline, batch_validation.prepared, features, contract)
    assert len(predictions) == 4
    assert batch_validation.unexpected_columns == ("ignored_note",)


def test_minimal_property_dataset_trains_without_fabricated_columns():
    frame = pd.read_csv(SAMPLES / "minimal_property_sample.csv")
    domain = analyze_domain(frame)
    features = ["area_sqft", "bedrooms", "bathrooms", "location"]
    contract = build_schema_contract(frame, features, "price", domain)
    result = train_regressors(frame, "price", features, cv_folds=3, tune=False)
    assert list(contract.feature_order) == features
    assert "parking" not in contract.feature_order
    assert "latitude" not in contract.feature_order
    assert len(result.active_pipeline.predict(frame[features].head(2))) == 2
