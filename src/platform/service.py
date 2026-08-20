from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.domain.domain_detector import analyze_domain
from src.domain.market_intelligence import region_for_market
from src.features.leakage import detect_leakage
from src.models.predictor import prediction_interval
from src.models.ood import assess_ood
from src.models.production_validation import production_validation_evidence, shadow_decision
from src.models.registry import save_active_model
from src.models.trainer import train_regressors
from src.models.uncertainty import model_confidence_score
from src.platform.ingestion import load_tabular, profile_contract
from src.platform.persistence import PLATFORM_DIR, connection, get_one, insert, list_records, update, utc_now
from src.validation.prediction_validator import validate_and_prepare
from src.validation.schema_contract import ModelSchemaContract, build_schema_contract


logger = logging.getLogger("pricepredict.platform")
QUICK_MODELS = ["Linear Regression", "Ridge", "Random Forest", "Histogram Gradient Boosting"]


def _safe_segment(value: str) -> str:
    cleaned = "".join(character for character in value if character.isalnum() or character in "-_")
    if not cleaned:
        raise ValueError("Tenant identifiers must contain letters or numbers.")
    return cleaned[:80]


def ingest_dataset(
    content: bytes, filename: str, tenant_id: str, owner_type: str = "operator",
    *, source: str = "upload", source_kind: str = "unverified", permission: str = "unverified",
    coverage: str | None = None,
) -> dict[str, Any]:
    tenant = _safe_segment(tenant_id)
    if owner_type not in {"operator", "customer"}:
        raise ValueError("owner_type must be operator or customer.")
    frame = load_tabular(content, filename)
    contract = profile_contract(frame, filename)
    dataset_fingerprint = contract["dataset_id"]
    if "synthetic" in filename.casefold():
        source_kind = "synthetic"
    contract["provenance"] = {
        "source": source, "source_kind": source_kind, "owner": tenant,
        "permission": permission, "coverage": coverage,
        "dataset_fingerprint": dataset_fingerprint,
    }
    dataset_id = f"ds-{uuid.uuid4().hex[:12]}"
    contract["dataset_id"] = dataset_id
    retention_until = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat() if owner_type == "customer" else None
    contract["retention_policy"] = "Customer upload and derived artifacts are retained for 24 hours unless deleted sooner." if owner_type == "customer" else "Operator-managed; retained until explicitly deleted."
    directory = PLATFORM_DIR / "tenants" / tenant / "datasets" / dataset_id
    directory.mkdir(parents=True, exist_ok=False)
    data_path = directory / "source.csv"
    contract_path = directory / "schema-contract.json"
    frame.to_csv(data_path, index=False)
    contract_path.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")
    insert("datasets", {
        "id": dataset_id, "tenant_id": tenant, "owner_type": owner_type,
        "name": Path(filename).name, "file_path": str(data_path), "file_format": Path(filename).suffix.lower(),
        "row_count": len(frame), "schema_contract": contract, "created_at": utc_now(), "retention_until": retention_until,
    })
    logger.info("dataset_ingested", extra={"dataset_id": dataset_id, "tenant_id": tenant, "rows": len(frame)})
    return contract


def get_dataset(dataset_id: str, tenant_id: str) -> dict[str, Any]:
    record = get_one("datasets", dataset_id, _safe_segment(tenant_id))
    if not record:
        raise KeyError("Dataset not found.")
    record["schema_contract"] = json.loads(record["schema_contract"])
    record.pop("file_path", None)
    return record


def datasets_for_tenant(tenant_id: str) -> list[dict[str, Any]]:
    records = list_records("datasets", _safe_segment(tenant_id))
    for record in records:
        record["schema_contract"] = json.loads(record["schema_contract"])
        record.pop("file_path", None)
    return records


def delete_dataset(dataset_id: str, tenant_id: str) -> None:
    """Delete one tenant-owned dataset and its derived records/artifacts."""
    tenant = _safe_segment(tenant_id)
    record = get_one("datasets", dataset_id, tenant)
    if not record:
        raise KeyError("Dataset not found.")
    with connection() as database:
        model_ids = [row["id"] for row in database.execute(
            "SELECT id FROM models WHERE dataset_id = ? AND tenant_id = ?", (dataset_id, tenant)
        ).fetchall()]
        for model_id in model_ids:
            database.execute("DELETE FROM leads WHERE model_id = ? AND tenant_id = ?", (model_id, tenant))
            database.execute("DELETE FROM predictions WHERE model_id = ? AND tenant_id = ?", (model_id, tenant))
        database.execute("DELETE FROM models WHERE dataset_id = ? AND tenant_id = ?", (dataset_id, tenant))
        database.execute("DELETE FROM training_jobs WHERE dataset_id = ? AND tenant_id = ?", (dataset_id, tenant))
        database.execute("DELETE FROM datasets WHERE id = ? AND tenant_id = ?", (dataset_id, tenant))
    tenant_root = (PLATFORM_DIR / "tenants" / tenant).resolve()
    for target in (tenant_root / "datasets" / dataset_id, tenant_root / "models" / dataset_id):
        resolved = target.resolve()
        if resolved != tenant_root and tenant_root in resolved.parents and resolved.exists():
            shutil.rmtree(resolved)
    logger.info("dataset_deleted", extra={"dataset_id": dataset_id, "tenant_id": tenant})


def purge_expired_customer_datasets(at: str | None = None) -> int:
    cutoff = at or utc_now()
    with connection() as database:
        expired = database.execute(
            "SELECT id, tenant_id FROM datasets WHERE owner_type = 'customer' AND retention_until IS NOT NULL AND retention_until <= ?",
            (cutoff,),
        ).fetchall()
    for record in expired:
        delete_dataset(record["id"], record["tenant_id"])
    return len(expired)


def _validated_features(frame: pd.DataFrame, target: str) -> tuple[list[str], list[dict[str, Any]]]:
    if target not in frame:
        raise ValueError("The confirmed target is not present in this dataset.")
    target_values = pd.to_numeric(frame[target], errors="coerce")
    if target_values.notna().sum() < 30:
        raise ValueError("At least 30 rows with a numeric target are required.")
    if target_values.nunique(dropna=True) < 5 or float(target_values.var()) <= 0:
        raise ValueError("The target has insufficient variance for a defensible valuation model.")
    candidates = [str(column) for column in frame.columns if str(column) != target]
    warnings = detect_leakage(frame, target, candidates)
    blocked = {warning.column for warning in warnings if warning.severity == "high"}
    identifiers = {
        warning.column for warning in warnings
        if warning.severity == "medium" and any(token in warning.reason.lower() for token in ("identifier", "post-event"))
    }
    features = [column for column in candidates if column not in blocked | identifiers and frame[column].nunique(dropna=True) > 1]
    if not features:
        raise ValueError("No usable non-leaky predictive feature remains after validation.")
    return features, [warning.__dict__ for warning in warnings]


def train_dataset(
    dataset_id: str, tenant_id: str, target: str, lightweight: bool = False,
    *, market: str | None = None, market_confirmed: bool = False,
    region: str | None = None,
    currency: str | None = None, currency_confirmed: bool = False,
    property_type: str | None = None, transaction_type: str | None = None,
    model_scope: str | None = None, allow_region_fallback: bool = False,
) -> dict[str, Any]:
    tenant = _safe_segment(tenant_id)
    record = get_one("datasets", dataset_id, tenant)
    if not record:
        raise KeyError("Dataset not found.")
    frame = pd.read_csv(record["file_path"])
    dataset_contract = json.loads(record["schema_contract"])
    domain = analyze_domain(frame)
    if not domain.is_real_estate:
        raise ValueError(f"This dataset has low real-estate confidence ({domain.confidence:.0%}); training was blocked.")
    features, leakage = _validated_features(frame, target)
    market_hypothesis = dataset_contract.get("market_hypothesis", {})
    selected_market = (market or market_hypothesis.get("candidate") or "UNCONFIRMED").upper()
    selected_region = (region or region_for_market(selected_market) or "UNCONFIRMED").upper()
    target_candidates = {candidate["column"]: candidate for candidate in dataset_contract.get("target_candidates", [])}
    target_info = target_candidates.get(target, {})
    selected_currency = (currency or target_info.get("currency_hypothesis") or "UNCONFIRMED").upper()
    inferred_transaction = "Rent" if target_info.get("target_type") == "rent" else "Sale"
    selected_transaction = transaction_type or inferred_transaction
    selected_property_type = property_type or (domain.property_types[0] if len(domain.property_types) == 1 else "Mixed")
    scope = model_scope or ("private" if record["owner_type"] == "customer" else "platform")
    contract = build_schema_contract(
        frame, features, target, domain, currency=selected_currency, market=selected_market,
        region=selected_region, transaction_type=selected_transaction, target_unit=target_info.get("target_unit", "currency amount"),
        dataset_fingerprint=dataset_id,
    )
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    insert("training_jobs", {
        "id": job_id, "tenant_id": tenant, "dataset_id": dataset_id,
        "status": "RUNNING", "detail": None, "created_at": utc_now(), "completed_at": None,
    })
    try:
        result = train_regressors(
            frame, target, features,
            mode="Core", selected_models=QUICK_MODELS if lightweight else None,
            cv_folds=3 if lightweight else 5, tune=not lightweight,
            tuning_iterations=2 if lightweight else 5, tune_top_n=1 if lightweight else 3,
            validation_strategy="auto",
            metadata={
                "dataset_name": record["name"], "dataset_id": dataset_id,
                "tenant_id": tenant, "owner_type": record["owner_type"],
                "asset_type": domain.asset_type, "prediction_granularity": domain.granularity,
                "market": selected_market, "market_confirmed": market_confirmed,
                "region": selected_region,
                "property_type": selected_property_type, "transaction_type": selected_transaction,
                "currency": selected_currency, "currency_confirmed": currency_confirmed,
                "target_unit": target_info.get("target_unit", "currency amount"),
                "provenance": dataset_contract.get("provenance", {}), "model_scope": scope,
                "schema_contract": contract.to_dict(), "leakage_exclusions": leakage,
                "known_limitations": [
                    "Model-based estimate only; not a legal valuation or guaranteed appraisal.",
                    "Split-conformal coverage is marginal and may not hold for every subgroup or shifted market.",
                ],
            },
        )
        validation_evidence = production_validation_evidence(result, frame, target, features)
        result.metadata["production_validation"] = validation_evidence
        artifact_path = save_active_model(
            result, dataset_id, target, features,
            output_dir=PLATFORM_DIR / "tenants" / tenant / "models" / dataset_id,
        )
        payload = joblib.load(artifact_path)
        model_id = payload["model_id"]
        model_card = payload["model_card"]
        model_card.update({
            "training_date": payload["created_at"], "dataset_id": dataset_id,
            "prediction_contract": contract.to_dict(), "status": "VALIDATED",
            "market_confirmed": market_confirmed, "currency_confirmed": currency_confirmed,
            "model_scope": scope, "allow_region_fallback": allow_region_fallback,
            "production_validation": validation_evidence,
            "version": "1.0", "data_coverage": {
                "rows": len(frame), "property_types": list(domain.property_types),
                "market": selected_market, "region": selected_region,
                "coverage_statement": dataset_contract.get("provenance", {}).get("coverage"),
            },
            "disclaimer": "Not a legal valuation or guaranteed appraisal; model-based estimate only.",
        })
        insert("models", {
            "id": model_id, "tenant_id": tenant, "dataset_id": dataset_id,
            "artifact_path": str(artifact_path), "status": "VALIDATED", "is_active": 0,
            "model_card": model_card, "created_at": payload["created_at"],
            "market": selected_market, "asset_type": domain.asset_type,
            "region": selected_region,
            "property_type": selected_property_type, "transaction_type": selected_transaction,
            "model_scope": scope, "allow_region_fallback": int(allow_region_fallback),
        })
        update("training_jobs", job_id, tenant, {"status": "COMPLETED", "detail": model_id, "completed_at": utc_now()})
        logger.info("training_completed", extra={"job_id": job_id, "model_id": model_id, "tenant_id": tenant})
        return {"job_id": job_id, "model_id": model_id, "model_card": model_card}
    except Exception as exc:
        update("training_jobs", job_id, tenant, {"status": "FAILED", "detail": f"{type(exc).__name__}: {exc}", "completed_at": utc_now()})
        logger.exception("training_failed", extra={"job_id": job_id, "tenant_id": tenant})
        raise


def get_model(model_id: str, tenant_id: str) -> dict[str, Any]:
    record = get_one("models", model_id, _safe_segment(tenant_id))
    if not record:
        raise KeyError("Model not found.")
    record["model_card"] = json.loads(record["model_card"])
    record.pop("artifact_path", None)
    return record


def models_for_tenant(tenant_id: str) -> list[dict[str, Any]]:
    records = list_records("models", _safe_segment(tenant_id))
    for record in records:
        record["model_card"] = json.loads(record["model_card"])
        record.pop("artifact_path", None)
    return records


MODEL_TRANSITIONS = {
    "DRAFT": {"TRAINING", "ARCHIVED"},
    "TRAINING": {"VALIDATED", "ARCHIVED"},
    "VALIDATED": {"APPROVED", "ARCHIVED"},
    "APPROVED": {"PUBLISHED", "ARCHIVED"},
    "PUBLISHED": {"DEPRECATED", "ARCHIVED"},
    "DEPRECATED": {"ARCHIVED"},
    "READY": {"APPROVED", "ARCHIVED"},
}


def transition_model(model_id: str, tenant_id: str, new_status: str) -> dict[str, Any]:
    tenant = _safe_segment(tenant_id)
    record = get_one("models", model_id, tenant)
    if not record:
        raise KeyError("Model not found.")
    current, requested = str(record["status"]).upper(), new_status.upper()
    if requested not in MODEL_TRANSITIONS.get(current, set()):
        raise ValueError(f"Invalid model lifecycle transition: {current} → {requested}.")
    card = json.loads(record["model_card"])
    card["status"] = requested
    update("models", model_id, tenant, {"status": requested, "model_card": card})
    return get_model(model_id, tenant)


def approve_model(model_id: str, tenant_id: str) -> dict[str, Any]:
    return transition_model(model_id, tenant_id, "APPROVED")


def publish_model(model_id: str, tenant_id: str) -> dict[str, Any]:
    tenant = _safe_segment(tenant_id)
    record = get_one("models", model_id, tenant)
    if not record:
        raise KeyError("Model not found.")
    if str(record["status"]).upper() != "APPROVED":
        raise ValueError("Only an approved model can be published.")
    card = json.loads(record["model_card"])
    provenance = card.get("provenance", {})
    if record.get("model_scope") == "platform":
        if provenance.get("source_kind") != "real" or provenance.get("permission") not in {"owned", "licensed", "authorized"}:
            raise ValueError("Platform publication requires a real approved dataset with owned, licensed, or authorized provenance.")
        if not card.get("market_confirmed") or not card.get("currency_confirmed"):
            raise ValueError("Market and currency must be explicitly confirmed before publication.")
    with connection() as database:
        current = database.execute(
            "SELECT model_card FROM models WHERE tenant_id = ? AND market = ? AND asset_type = ? AND transaction_type = ? AND model_scope = ? AND status = 'PUBLISHED' AND is_active = 1 LIMIT 1",
            (tenant, record.get("market"), record.get("asset_type"), record.get("transaction_type"), record.get("model_scope")),
        ).fetchone()
    if current:
        decision = shadow_decision(json.loads(current["model_card"]), card)
        if not decision["approved"]:
            raise ValueError(f"Candidate failed production shadow gate: {decision}.")
        card["shadow_decision"] = decision
    with connection() as database:
        database.execute(
            "UPDATE models SET is_active = 0, status = CASE WHEN status = 'PUBLISHED' THEN 'DEPRECATED' ELSE status END WHERE tenant_id = ? AND market = ? AND asset_type = ? AND transaction_type = ? AND model_scope = ?",
            (tenant, record.get("market"), record.get("asset_type"), record.get("transaction_type"), record.get("model_scope")),
        )
        database.execute("UPDATE models SET is_active = 1, status = 'PUBLISHED' WHERE id = ? AND tenant_id = ?", (model_id, tenant))
        card["status"] = "PUBLISHED"
        database.execute("UPDATE models SET model_card = ? WHERE id = ? AND tenant_id = ?", (json.dumps(card, default=str), model_id, tenant))
    return get_model(model_id, tenant)


def route_model(
    tenant_id: str, market: str, asset_type: str, property_type: str, transaction_type: str,
    *, region: str | None = None, allow_regional_fallback: bool = False,
) -> dict[str, Any]:
    tenant = _safe_segment(tenant_id)
    with connection() as database:
        rows = database.execute(
            "SELECT * FROM models WHERE tenant_id = ? AND status = 'PUBLISHED' AND is_active = 1 AND model_scope = 'platform' AND asset_type = ? AND transaction_type = ?",
            (tenant, asset_type, transaction_type),
        ).fetchall()
    records = [dict(row) for row in rows]
    exact = [row for row in records if str(row.get("market", "")).upper() == market.upper() and str(row.get("property_type", "")).casefold() == property_type.casefold()]
    broad = [row for row in records if str(row.get("market", "")).upper() == market.upper() and str(row.get("property_type", "")).casefold() in {"mixed", "all"}]
    requested_region = (region or region_for_market(market) or "").upper()
    fallback = [row for row in records if allow_regional_fallback and row.get("allow_region_fallback") and str(row.get("region", "")).upper() == requested_region]
    candidates = exact or broad or fallback
    if not candidates:
        raise KeyError("No compatible published production model exists for this market, asset, property type, and transaction.")
    selected = sorted(candidates, key=lambda row: row["created_at"], reverse=True)[0]
    return get_model(selected["id"], tenant)


def published_market_catalog(tenant_id: str) -> list[dict[str, Any]]:
    tenant = _safe_segment(tenant_id)
    with connection() as database:
        rows = database.execute(
            "SELECT id, market, region, asset_type, property_type, transaction_type, created_at FROM models WHERE tenant_id = ? AND status = 'PUBLISHED' AND is_active = 1 AND model_scope = 'platform' ORDER BY market, asset_type, property_type",
            (tenant,),
        ).fetchall()
    return [dict(row) for row in rows]


def predict(model_id: str, tenant_id: str, values: dict[str, Any], require_active: bool = False) -> dict[str, Any]:
    tenant = _safe_segment(tenant_id)
    record = get_one("models", model_id, tenant)
    if not record:
        raise KeyError("Model not found.")
    if require_active and (not record["is_active"] or record["status"] != "PUBLISHED"):
        raise ValueError("This model has not been published for public predictions.")
    payload = joblib.load(record["artifact_path"])
    contract = payload.get("schema_contract")
    if not contract:
        raise ValueError("This legacy model does not have a prediction contract.")
    schema = ModelSchemaContract.from_dict(contract)
    card = json.loads(record["model_card"])
    requested_property_type = values.get("property_type") or values.get("asset_type")
    covered_property_type = card.get("property_type")
    if requested_property_type and covered_property_type not in {None, "Mixed", "All"} and str(requested_property_type).casefold() != str(covered_property_type).casefold():
        raise ValueError(f"No compatible production model exists for property type '{requested_property_type}'. This model covers '{covered_property_type}'.")
    ood = assess_ood(values, schema)
    if not ood.compatible:
        raise ValueError("No compatible production model exists for these inputs. " + " ".join(ood.blockers))
    validation = validate_and_prepare(pd.DataFrame([values]), schema)
    if validation.invalid_rows:
        raise ValueError("; ".join(validation.warnings) or "Prediction inputs are invalid.")
    row = validation.prepared
    estimate = float(payload["pipeline"].predict(row)[0])
    low, high = prediction_interval(
        estimate, residual_std=payload.get("residual_std"), calibrated_radius=payload.get("conformal_radius")
    )
    prediction_id = f"pred-{uuid.uuid4().hex[:12]}"
    insert("predictions", {
        "id": prediction_id, "tenant_id": tenant, "model_id": model_id,
        "estimate": estimate, "lower_bound": low, "upper_bound": high,
        "input_json": values, "created_at": utc_now(),
    })
    metrics = card.get("metrics", {})
    completeness = sum(name in values and values[name] not in (None, "") for name in payload["features"]) / max(1, len(payload["features"]))
    cv_rmse = float(metrics.get("CV RMSE", 0) or 0)
    stability = max(0.0, 1.0 - float(metrics.get("CV RMSE Std", cv_rmse) or cv_rmse) / max(cv_rmse, 1e-9))
    generalization = max(0.0, 1.0 - float(metrics.get("Overfit Gap", 1) or 0))
    recorded_quality = card.get("data_quality")
    quality_score = int(recorded_quality.get("score", 80)) if isinstance(recorded_quality, dict) else int(recorded_quality or 80)
    confidence = model_confidence_score(
        float(metrics.get("Test R²", 0) or 0), float(metrics.get("Test MAPE (%)", 100) or 100),
        completeness, quality_score,
        similarity_score=ood.comparable_coverage, stability_score=stability, generalization_score=generalization,
    )
    return {
        "prediction_id": prediction_id, "estimate": estimate,
        "range": {"lower": low, "upper": high, "coverage_target": 0.95, "method": payload.get("uncertainty_method")},
        "model_id": model_id, "model_version": card.get("training_date"),
        "prediction_granularity": card.get("prediction_granularity"),
        "market": card.get("market"), "property_type": card.get("property_type"),
        "transaction_type": card.get("transaction_type"), "currency": card.get("currency"),
        "training_coverage": {"rows": schema.row_count, "property_types": list(schema.property_types)},
        "last_updated": card.get("training_date"), "ood": ood.to_dict(),
        "aggregate_data_caveat": card.get("prediction_granularity") in {"Census Block / Geographic Area", "Locality Aggregate", "Unknown"},
        "model_confidence": {"score": confidence[0], "label": confidence[1], "factors": confidence[2], "is_probability": False},
        "disclaimer": card.get("disclaimer", "Not a legal valuation or guaranteed appraisal; model-based estimate only."),
    }


def capture_lead(model_id: str, tenant_id: str, contact: dict[str, Any], consent: bool, retention_days: int = 90) -> str:
    if not consent:
        raise ValueError("Explicit consent is required before contact details can be stored.")
    if not get_one("models", model_id, _safe_segment(tenant_id)):
        raise KeyError("Model not found.")
    lead_id = f"lead-{uuid.uuid4().hex[:12]}"
    retention = datetime.now(timezone.utc) + timedelta(days=max(1, min(retention_days, 365)))
    insert("leads", {
        "id": lead_id, "tenant_id": _safe_segment(tenant_id), "model_id": model_id,
        "contact_json": contact, "consent": 1, "retention_until": retention.isoformat(), "created_at": utc_now(),
    })
    return lead_id
