from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid

import joblib

from src.models.trainer import TrainingResult
from src.utils.config import MODEL_DIR, ensure_runtime_directories


ASSET_CODES = {"Residential": "RES", "Housing": "HSG", "Land": "LAND", "Commercial": "COM", "Rental": "RENT", "Generic": "GEN"}


def generate_model_id(asset_type: str, dataset_key: str, market: str | None = None, property_type: str | None = None) -> str:
    now = datetime.now(timezone.utc)
    code = ASSET_CODES.get(asset_type, "GEN")
    dataset_fragment = "".join(character for character in dataset_key if character.isalnum())[-4:].upper() or "DATA"
    market_code = "".join(character for character in (market or "GLOBAL") if character.isalnum())[:10].upper() or "GLOBAL"
    property_code = "".join(character for character in (property_type or "ALL") if character.isalnum())[:8].upper() or "ALL"
    return f"{market_code}-{code}-{property_code}-{now:%Y%m%d%H%M%S}-{dataset_fragment}-{uuid.uuid4().hex[:4].upper()}"


def activate_model(result: TrainingResult, model_name: str) -> TrainingResult:
    if model_name not in result.pipelines:
        raise KeyError(f"Unknown model: {model_name}")
    result.active_model_name = model_name
    return result


def save_active_model(
    result: TrainingResult,
    dataset_key: str,
    target: str,
    features: list[str],
    output_dir: str | Path | None = None,
) -> Path:
    """Persist the complete fitted preprocessing/model pipeline and metadata."""
    ensure_runtime_directories()
    model_id = result.metadata.get("model_id") or generate_model_id(
        result.metadata.get("asset_type", "Generic"), dataset_key,
        result.metadata.get("market"), result.metadata.get("property_type"),
    )
    result.metadata["model_id"] = model_id
    destination = Path(output_dir) if output_dir is not None else MODEL_DIR
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{model_id}_{result.active_model_name.lower().replace(' ', '_')}.joblib"
    leaderboard_row = result.leaderboard.loc[result.leaderboard["Model"] == result.active_model_name].iloc[0].to_dict()
    model = result.active_pipeline.named_steps["model"]
    base_model = getattr(model, "regressor_", model)
    model_card = {
        "model_id": model_id,
        "dataset_name": result.metadata.get("dataset_name", dataset_key),
        "dataset_fingerprint": dataset_key,
        "target": target,
        "features": features,
        "asset_type": result.metadata.get("asset_type", "Generic"),
        "prediction_granularity": result.metadata.get("prediction_granularity", "Unknown"),
        "market": result.metadata.get("market", "UNCONFIRMED"),
        "region": result.metadata.get("region"),
        "property_type": result.metadata.get("property_type", "Mixed"),
        "transaction_type": result.metadata.get("transaction_type", "Sale"),
        "currency": result.metadata.get("currency", "UNCONFIRMED"),
        "target_unit": result.metadata.get("target_unit", "currency amount"),
        "provenance": result.metadata.get("provenance", {}),
        "algorithm": result.active_model_name,
        "hyperparameters": base_model.get_params(deep=False),
        "metrics": leaderboard_row,
        "data_quality": result.metadata.get("quality"),
        "validation_strategy": result.metadata.get("validation_strategy", "Random shuffled K-Fold"),
        "known_limitations": result.metadata.get("known_limitations", []),
        "uncertainty_method": result.metadata.get("uncertainty_method"),
        "conformal_radius": getattr(result, "conformal_radius", {}).get(result.active_model_name),
        "conformal_test_coverage": getattr(result, "conformal_coverage", {}).get(result.active_model_name),
    }
    payload = {
        "pipeline": result.active_pipeline,
        "model_name": result.active_model_name,
        "target": target,
        "features": features,
        "residual_std": result.residual_std[result.active_model_name],
        "conformal_radius": getattr(result, "conformal_radius", {}).get(result.active_model_name),
        "conformal_coverage": getattr(result, "conformal_coverage", {}).get(result.active_model_name),
        "uncertainty_method": result.metadata.get("uncertainty_method"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "schema_contract": result.metadata.get("schema_contract"),
        "model_card": model_card,
        "metadata": result.metadata,
    }
    joblib.dump(payload, path)
    return path


def load_model(path: str | Path) -> dict:
    payload = joblib.load(path)
    required = {"pipeline", "model_name", "target", "features"}
    if not required.issubset(payload):
        raise ValueError("This model artifact is missing required metadata.")
    return payload
