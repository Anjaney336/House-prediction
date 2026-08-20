from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import app
from src.domain.locality_reference import resolve_locality
from src.platform import persistence, service
from src.platform.ingestion import profile_contract
from src.platform.retraining import drift_report, drift_status


@pytest.fixture()
def isolated_platform(tmp_path, monkeypatch):
    platform_dir = tmp_path / "platform"
    monkeypatch.setattr(persistence, "DATABASE_PATH", platform_dir / "platform.db")
    monkeypatch.setattr(persistence, "PLATFORM_DIR", platform_dir)
    monkeypatch.setattr(service, "PLATFORM_DIR", platform_dir)
    persistence.initialize_database()
    return platform_dir


def _international_dataset(rows: int = 54) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    living = rng.integers(650, 2600, rows)
    bedrooms = rng.integers(1, 6, rows)
    city = rng.choice(["Austin", "Denver", "Portland"], rows)
    city_effect = pd.Series(city).map({"Austin": 90000, "Denver": 60000, "Portland": 75000}).to_numpy()
    return pd.DataFrame({
        "listing_ref": [f"INT-{index:04d}" for index in range(rows)],
        "municipality": city,
        "zip_code": rng.choice(["78701", "80202", "97205"], rows),
        "living_area_sqft": living,
        "bedrooms": bedrooms,
        "year_built": rng.integers(1970, 2025, rows),
        "sale_price_usd": living * 310 + bedrooms * 18000 + city_effect + rng.normal(0, 25000, rows),
    })


def test_schema_inference_is_dataset_agnostic_and_flags_target_confirmation():
    frame = _international_dataset()
    frame["rent"] = frame["sale_price_usd"] * 0.004
    contract = profile_contract(frame)
    assert contract["domain"]["classification"] == "REAL_ESTATE"
    assert {candidate["column"] for candidate in contract["target_candidates"]} == {"sale_price_usd", "rent"}
    assert contract["operator_confirmation_required"]
    area = next(column for column in contract["columns"] if column["name"] == "living_area_sqft")
    assert area["unit"] == "sqft"


@pytest.mark.parametrize(
    ("filename", "frame", "target"),
    [
        ("noida_style.csv", pd.read_csv("data/sample_datasets/individual_property_sample.csv"), "price"),
        ("international.csv", _international_dataset(), "sale_price_usd"),
    ],
)
def test_full_ingest_train_predict_loop_for_different_schemas(isolated_platform, filename, frame, target):
    tenant = f"tenant-{target}"
    contract = service.ingest_dataset(frame.to_csv(index=False).encode(), filename, tenant)
    trained = service.train_dataset(contract["dataset_id"], tenant, target, lightweight=True)
    model = service.get_model(trained["model_id"], tenant)
    features = model["model_card"]["features"]
    values = {name: frame.iloc[0][name].item() if hasattr(frame.iloc[0][name], "item") else frame.iloc[0][name] for name in features}
    result = service.predict(trained["model_id"], tenant, values)
    assert result["range"]["lower"] < result["estimate"] < result["range"]["upper"]
    assert result["model_id"] == trained["model_id"]
    with pytest.raises(KeyError):
        service.get_model(trained["model_id"], "another-tenant")


def test_api_auth_tenant_isolation_and_raw_csv_upload(isolated_platform, monkeypatch):
    monkeypatch.setenv("PRICEPREDICT_API_KEY", "integration-secret")
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    frame = pd.read_csv("data/sample_datasets/minimal_property_sample.csv")
    response = client.post(
        "/datasets?filename=visitor.csv&owner_type=customer",
        content=frame.to_csv(index=False).encode(),
        headers={"X-Tenant-ID": "customer-a", "X-API-Key": "integration-secret", "Content-Type": "text/csv"},
    )
    assert response.status_code == 200, response.text
    dataset_id = response.json()["dataset_id"]
    assert client.get(f"/datasets/{dataset_id}", headers={"X-Tenant-ID": "customer-b", "X-API-Key": "integration-secret"}).status_code == 404
    assert client.get("/datasets", headers={"X-Tenant-ID": "customer-a", "X-API-Key": "wrong"}).status_code == 401
    assert client.delete(f"/datasets/{dataset_id}", headers={"X-Tenant-ID": "customer-a", "X-API-Key": "integration-secret"}).status_code == 204
    assert client.get(f"/datasets/{dataset_id}", headers={"X-Tenant-ID": "customer-a", "X-API-Key": "integration-secret"}).status_code == 404


def test_legacy_customer_page_redirects_to_shared_dashboard():
    response = TestClient(app).get("/customer", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "http://127.0.0.1:8501"


def test_customer_upload_uses_server_derived_isolated_tenant_and_retention(isolated_platform):
    client = TestClient(app)
    frame = pd.read_csv("data/sample_datasets/minimal_property_sample.csv")
    token = "a-secure-session-token-for-customer-one"
    response = client.post(
        "/customer/datasets?filename=comparables.csv", content=frame.to_csv(index=False).encode(),
        headers={"X-Session-Token": token, "Content-Type": "text/csv"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_tenant"].startswith("customer-")
    assert "24 hours" in body["schema_contract"]["retention_policy"]
    assert client.post(
        "/customer/datasets?filename=comparables.csv", content=frame.to_csv(index=False).encode(),
        headers={"X-Session-Token": "short"},
    ).status_code == 401


def test_drift_threshold_is_explicit():
    assert drift_status(100.0, 125.0, threshold=0.20)["alert"] is True
    assert drift_status(100.0, 115.0, threshold=0.20)["alert"] is False
    reference = pd.DataFrame({"area": [900, 950, 1000, 1050] * 20, "type": ["Apartment"] * 80})
    shifted = pd.DataFrame({"area": [1800, 1900, 2000, 2100] * 20, "type": ["Villa"] * 80})
    report = drift_report(reference, shifted, ["area", "type"])
    assert report["alert"] and report["retraining_recommended"]


def test_locality_reference_is_data_driven():
    assert resolve_locality("Sec-62 Noida")["normalized_name"] == "Sector 62 Noida"
    assert resolve_locality("An Unlisted Locality") is None


def test_production_lifecycle_router_and_ood_gate(isolated_platform):
    frame = _international_dataset(72)
    frame["property_type"] = "Apartment"
    contract = service.ingest_dataset(
        frame.to_csv(index=False).encode(), "licensed-austin-transactions.csv", "operator",
        source="Licensed broker transaction export", source_kind="real", permission="licensed",
        coverage="Austin transactions, 2024–2026",
    )
    trained = service.train_dataset(
        contract["dataset_id"], "operator", "sale_price_usd", lightweight=True,
        market="AUSTIN", market_confirmed=True, currency="USD", currency_confirmed=True,
        property_type="Apartment", transaction_type="Sale",
    )
    assert trained["model_card"]["status"] == "VALIDATED"
    assert service.approve_model(trained["model_id"], "operator")["status"] == "APPROVED"
    published = service.publish_model(trained["model_id"], "operator")
    assert published["status"] == "PUBLISHED" and published["is_active"] == 1
    routed = service.route_model("operator", "AUSTIN", "Residential", "Apartment", "Sale")
    assert routed["id"] == trained["model_id"]
    values = {name: frame.iloc[0][name].item() if hasattr(frame.iloc[0][name], "item") else frame.iloc[0][name] for name in trained["model_card"]["features"]}
    prediction = service.predict(trained["model_id"], "operator", values, require_active=True)
    assert prediction["market"] == "AUSTIN" and prediction["ood"]["compatible"]
    values["property_type"] = "Warehouse"
    with pytest.raises(ValueError, match="No compatible production model"):
        service.predict(trained["model_id"], "operator", values, require_active=True)
    with pytest.raises(KeyError, match="No compatible published"):
        service.route_model("operator", "GURUGRAM", "Residential", "Villa", "Sale")


def test_synthetic_dataset_cannot_be_published_as_platform_model(isolated_platform):
    frame = _international_dataset(60)
    frame["property_type"] = "Apartment"
    contract = service.ingest_dataset(
        frame.to_csv(index=False).encode(), "synthetic-market.csv", "operator",
        source_kind="real", permission="owned",
    )
    trained = service.train_dataset(
        contract["dataset_id"], "operator", "sale_price_usd", lightweight=True,
        market="AUSTIN", market_confirmed=True, currency="USD", currency_confirmed=True,
        property_type="Apartment",
    )
    service.approve_model(trained["model_id"], "operator")
    with pytest.raises(ValueError, match="real approved dataset"):
        service.publish_model(trained["model_id"], "operator")


def test_customer_private_model_activates_without_entering_platform_router(isolated_platform):
    frame = _international_dataset(60)
    frame["property_type"] = "Apartment"
    contract = service.ingest_dataset(frame.to_csv(index=False).encode(), "private-comparables.csv", "customer-private", owner_type="customer")
    trained = service.train_dataset(
        contract["dataset_id"], "customer-private", "sale_price_usd", lightweight=True,
        market="DENVER", currency="USD", property_type="Apartment", model_scope="private",
    )
    service.approve_model(trained["model_id"], "customer-private")
    activated = service.publish_model(trained["model_id"], "customer-private")
    assert activated["status"] == "PUBLISHED" and activated["model_scope"] == "private"
    with pytest.raises(KeyError):
        service.route_model("customer-private", "DENVER", "Residential", "Apartment", "Sale")
    with pytest.raises(KeyError):
        service.get_model(trained["model_id"], "another-customer")
