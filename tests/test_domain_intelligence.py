import pandas as pd

from src.data.loader import DataLoadError, load_csv
from src.data.quality import assess_quality
from src.domain.domain_detector import analyze_domain
from src.domain.property_ontology import map_column, schema_mapping
from src.domain.target_detector import detect_targets
from src.features.leakage import detect_leakage
from src.utils.config import SAMPLE_DATASET
from src.validation.prediction_validator import validate_and_prepare
from src.validation.schema_contract import build_schema_contract


def test_california_is_real_estate_block_level_with_target():
    frame = pd.read_csv(SAMPLE_DATASET)
    domain = analyze_domain(frame)
    candidates = detect_targets(frame)
    assert domain.is_real_estate
    assert domain.granularity == "Census Block / Geographic Area"
    assert domain.prediction_label == "Estimated Area/Block Housing Value"
    assert candidates[0].column == "median_house_value"
    assert not domain.comparables_supported


def test_indian_apartment_maps_roles_and_property_granularity():
    frame = pd.DataFrame(
        {
            "property_type": ["Flat"] * 60,
            "city": ["Pune"] * 60,
            "locality_name": ["Baner", "Wakad"] * 30,
            "area_sqft": range(900, 960),
            "bhk": [2, 3] * 30,
            "bath": [2] * 60,
            "parking": [1] * 60,
            "furnishing": ["Semi"] * 60,
            "price": range(7_000_000, 7_000_060),
        }
    )
    domain = analyze_domain(frame)
    mapping = schema_mapping(list(frame.columns))
    assert domain.is_real_estate and domain.granularity == "Individual Property"
    assert mapping["bhk"]["normalized_role"] == "bedrooms"
    assert mapping["area_sqft"]["group"] == "Size"
    assert domain.comparables_supported


def test_commercial_land_and_rental_assets_are_distinguished():
    commercial = pd.DataFrame({"commercial_area": [1000, 1200], "office_type": ["A", "B"], "location": ["CBD", "CBD"], "parking": [2, 3], "monthly_rent": [5000, 6000]})
    land = pd.DataFrame({"plot_area": [2000, 3000], "road_width": [30, 40], "zoning": ["R", "C"], "corner_plot": [True, False], "sale_price": [100000, 150000]})
    rental = pd.DataFrame({"area_sqft": [800, 900], "bedrooms": [2, 2], "city": ["A", "B"], "monthly_rent": [1500, 1800]})
    assert analyze_domain(commercial).asset_type == "Commercial"
    assert analyze_domain(land).asset_type == "Land"
    assert analyze_domain(rental).asset_type == "Rental"


def test_retail_dataset_is_not_misrepresented_as_property_data():
    frame = pd.DataFrame({"customer_id": range(50), "product": ["A"] * 50, "quantity": [2] * 50, "sales": range(100, 150), "discount": [0.1] * 50})
    domain = analyze_domain(frame)
    assert not domain.is_real_estate
    assert domain.domain == "GENERIC_REGRESSION"
    assert domain.asset_type == "Generic"


def test_real_estate_without_target_has_no_supervised_candidate():
    frame = pd.DataFrame({"area_sqft": [900, 1000], "bedrooms": [2, 3], "bathrooms": [2, 2], "location": ["A", "B"]})
    assert analyze_domain(frame).is_real_estate
    assert detect_targets(frame) == []


def test_leakage_quality_and_schema_contract_validation():
    frame = pd.DataFrame(
        {
            "area_sqft": [900.0, 1000.0, 1100.0, 1200.0] * 10,
            "bedrooms": [2, 2, 3, 3] * 10,
            "location": ["A", "B", "A", "B"] * 10,
            "price_per_sqft": [100, 101, 99, 102] * 10,
            "price": [90000, 101000, 108900, 122400] * 10,
        }
    )
    domain = analyze_domain(frame)
    features = ["area_sqft", "bedrooms", "location", "price_per_sqft"]
    leakage = detect_leakage(frame, "price", features)
    assert any(item.column == "price_per_sqft" and item.severity == "high" for item in leakage)
    quality = assess_quality(frame, "price", features, domain, leakage)
    assert quality.leakage_risk == "High"
    contract = build_schema_contract(frame, ["area_sqft", "bedrooms", "location"], "price", domain, "INR")
    incoming = pd.DataFrame({"area_sqft": [950], "bedrooms": [2], "extra": [1]})
    result = validate_and_prepare(incoming, contract)
    assert result.valid_rows == 1
    assert result.missing_columns == ("location",)
    assert result.unexpected_columns == ("extra",)
    assert list(result.prepared.columns) == list(contract.feature_order)


def test_loader_rejects_empty_and_one_column_csv(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("a,b\n", encoding="utf-8")
    one = tmp_path / "one.csv"
    one.write_text("a\n1\n", encoding="utf-8")
    for path in (empty, one):
        try:
            load_csv(path)
        except DataLoadError:
            pass
        else:
            raise AssertionError("Expected a clear DataLoadError")


def test_prediction_page_contains_no_california_specific_form(tmp_path):
    source = open("app/pages/5_Predict.py", encoding="utf-8").read()
    forbidden = ["housing_median_age", "total_rooms", "ocean_proximity"]
    assert not any(name in source for name in forbidden)


def test_production_code_contains_no_california_schema_assumptions():
    from pathlib import Path

    forbidden = {"housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity"}
    roots = [Path("app"), Path("backend"), Path("scripts"), Path("src")]
    violations = []
    for root in roots:
        for path in root.rglob("*.py"):
            if "src\\demo" in str(path) or "src/demo" in path.as_posix():
                continue
            text = path.read_text(encoding="utf-8").casefold()
            matched = sorted(term for term in forbidden if term in text)
            if matched:
                violations.append((str(path), matched))
    assert not violations, f"California-only schema terms leaked into production code: {violations}"
