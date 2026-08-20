from __future__ import annotations

import logging
import os
import time
import hashlib
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from src.platform.persistence import initialize_database
from src.platform.service import (
    approve_model, capture_lead, datasets_for_tenant, delete_dataset, get_dataset, get_model, ingest_dataset,
    models_for_tenant, predict, publish_model, published_market_catalog, route_model, train_dataset,
)


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("pricepredict.api")
app = FastAPI(title="PricePredict AI API", version="1.0.0", description="Tenant-aware, dataset-agnostic real-estate valuation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")],
    allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-API-Key", "X-Tenant-ID", "X-Session-Token"],
)
initialize_database()


class SlidingWindowLimiter:
    def __init__(self, limit: int, seconds: int):
        self.limit, self.seconds = limit, seconds
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        events = self.events[key]
        while events and events[0] <= now - self.seconds:
            events.popleft()
        if len(events) >= self.limit:
            raise HTTPException(429, "Rate limit exceeded. Try again later.")
        events.append(now)


public_limiter = SlidingWindowLimiter(int(os.getenv("PUBLIC_RATE_LIMIT", "60")), 60)
training_limiter = SlidingWindowLimiter(int(os.getenv("TRAIN_RATE_LIMIT", "5")), 3600)


def tenant_header(x_tenant_id: str = Header(..., alias="X-Tenant-ID")) -> str:
    return x_tenant_id


def operator_auth(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    configured = os.getenv("PRICEPREDICT_API_KEY")
    if not configured:
        raise HTTPException(503, "Operator API key is not configured.")
    import secrets
    if not secrets.compare_digest(x_api_key, configured):
        raise HTTPException(401, "Invalid API key.")


class TrainRequest(BaseModel):
    dataset_id: str
    target: str
    lightweight: bool = False
    market: str | None = None
    market_confirmed: bool = False
    region: str | None = None
    currency: str | None = None
    currency_confirmed: bool = False
    property_type: str | None = None
    transaction_type: str | None = None
    allow_region_fallback: bool = False


class PredictionRequest(BaseModel):
    model_id: str
    values: dict[str, Any]


class BatchPredictionRequest(BaseModel):
    model_id: str
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


class LeadRequest(BaseModel):
    model_id: str
    contact: dict[str, Any]
    consent: bool
    retention_days: int = Field(default=90, ge=1, le=365)


class RoutedPredictionRequest(BaseModel):
    market: str
    asset_type: str
    property_type: str
    region: str | None = None
    transaction_type: str = "Sale"
    values: dict[str, Any]
    allow_regional_fallback: bool = False


def customer_tenant(x_session_token: str = Header(..., alias="X-Session-Token")) -> str:
    if len(x_session_token) < 20:
        raise HTTPException(401, "A strong customer session token is required.")
    return "customer-" + hashlib.sha256(x_session_token.encode()).hexdigest()[:24]


@app.middleware("http")
async def request_observability(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed", extra={"path": request.url.path, "method": request.method})
        raise
    response.headers["X-Response-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    logger.info("request_completed", extra={"path": request.url.path, "method": request.method, "status": response.status_code})
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "pricepredict-api"}


@app.post("/datasets", dependencies=[Depends(operator_auth)])
async def create_dataset(
    request: Request, filename: str = Query(...), owner_type: str = Query("operator"),
    source: str = Query("upload"), source_kind: str = Query("unverified"),
    permission: str = Query("unverified"), coverage: str | None = Query(None),
    tenant_id: str = Depends(tenant_header),
):
    try:
        return ingest_dataset(
            await request.body(), filename, tenant_id, owner_type,
            source=source, source_kind=source_kind, permission=permission, coverage=coverage,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/customer/datasets")
async def create_customer_dataset(request: Request, filename: str = Query(...), tenant_id: str = Depends(customer_tenant)):
    public_limiter.check(f"upload:{tenant_id}:{request.client.host if request.client else 'unknown'}")
    try:
        contract = ingest_dataset(await request.body(), filename, tenant_id, "customer")
        return {"session_tenant": tenant_id, "schema_contract": contract}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/datasets", dependencies=[Depends(operator_auth)])
def list_datasets(tenant_id: str = Depends(tenant_header)):
    return datasets_for_tenant(tenant_id)


@app.get("/datasets/{dataset_id}", dependencies=[Depends(operator_auth)])
def dataset_detail(dataset_id: str, tenant_id: str = Depends(tenant_header)):
    try:
        return get_dataset(dataset_id, tenant_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.delete("/datasets/{dataset_id}", status_code=204, dependencies=[Depends(operator_auth)])
def remove_dataset(dataset_id: str, tenant_id: str = Depends(tenant_header)):
    try:
        delete_dataset(dataset_id, tenant_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/train", dependencies=[Depends(operator_auth)])
def train(body: TrainRequest, request: Request, tenant_id: str = Depends(tenant_header)):
    training_limiter.check(f"{tenant_id}:{request.client.host if request.client else 'unknown'}")
    try:
        return train_dataset(
            body.dataset_id, tenant_id, body.target, body.lightweight,
            market=body.market, market_confirmed=body.market_confirmed,
            region=body.region,
            currency=body.currency, currency_confirmed=body.currency_confirmed,
            property_type=body.property_type, transaction_type=body.transaction_type,
            allow_region_fallback=body.allow_region_fallback,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/customer/train")
def customer_train(body: TrainRequest, request: Request, tenant_id: str = Depends(customer_tenant)):
    training_limiter.check(f"customer:{tenant_id}:{request.client.host if request.client else 'unknown'}")
    try:
        return train_dataset(
            body.dataset_id, tenant_id, body.target, lightweight=True,
            market=body.market, market_confirmed=body.market_confirmed,
            region=body.region,
            currency=body.currency, currency_confirmed=body.currency_confirmed,
            property_type=body.property_type, transaction_type=body.transaction_type,
            model_scope="private",
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/customer/predict")
def customer_prediction(body: PredictionRequest, request: Request, tenant_id: str = Depends(customer_tenant)):
    public_limiter.check(f"customer-predict:{tenant_id}:{request.client.host if request.client else 'unknown'}")
    try:
        return predict(body.model_id, tenant_id, body.values)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/models", dependencies=[Depends(operator_auth)])
def list_models(tenant_id: str = Depends(tenant_header)):
    return models_for_tenant(tenant_id)


@app.get("/models/{model_id}")
def model_detail(model_id: str, tenant_id: str = Depends(tenant_header)):
    try:
        return get_model(model_id, tenant_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/models/{model_id}/schema")
@app.get("/schema")
def prediction_schema(model_id: str, tenant_id: str = Depends(tenant_header)):
    try:
        return get_model(model_id, tenant_id)["model_card"]["prediction_contract"]
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/markets")
def market_catalog(tenant_id: str = Depends(tenant_header)):
    return published_market_catalog(tenant_id)


@app.get("/route")
def route_contract(
    market: str, asset_type: str, property_type: str, transaction_type: str = "Sale",
    region: str | None = None, allow_regional_fallback: bool = False, tenant_id: str = Depends(tenant_header),
):
    try:
        return route_model(tenant_id, market, asset_type, property_type, transaction_type, region=region, allow_regional_fallback=allow_regional_fallback)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/models/{model_id}/publish", dependencies=[Depends(operator_auth)])
def publish(model_id: str, tenant_id: str = Depends(tenant_header)):
    try:
        return publish_model(model_id, tenant_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/models/{model_id}/approve", dependencies=[Depends(operator_auth)])
def approve(model_id: str, tenant_id: str = Depends(tenant_header)):
    try:
        return approve_model(model_id, tenant_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/customer/models/{model_id}/activate")
def activate_customer_model(model_id: str, tenant_id: str = Depends(customer_tenant)):
    try:
        approve_model(model_id, tenant_id)
        return publish_model(model_id, tenant_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/predict")
def single_prediction(body: PredictionRequest, request: Request, tenant_id: str = Depends(tenant_header)):
    public_limiter.check(f"{tenant_id}:{request.client.host if request.client else 'unknown'}")
    try:
        return predict(body.model_id, tenant_id, body.values, require_active=True)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/valuation")
def routed_valuation(body: RoutedPredictionRequest, request: Request, tenant_id: str = Depends(tenant_header)):
    public_limiter.check(f"valuation:{tenant_id}:{request.client.host if request.client else 'unknown'}")
    try:
        model = route_model(
            tenant_id, body.market, body.asset_type, body.property_type, body.transaction_type,
            region=body.region, allow_regional_fallback=body.allow_regional_fallback,
        )
        return predict(model["id"], tenant_id, body.values, require_active=True)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/predict/batch", dependencies=[Depends(operator_auth)])
def batch_prediction(body: BatchPredictionRequest, tenant_id: str = Depends(tenant_header)):
    try:
        return {"predictions": [predict(body.model_id, tenant_id, row) for row in body.rows]}
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/leads")
def lead(body: LeadRequest, request: Request, tenant_id: str = Depends(tenant_header)):
    public_limiter.check(f"lead:{tenant_id}:{request.client.host if request.client else 'unknown'}")
    try:
        return {"lead_id": capture_lead(body.model_id, tenant_id, body.contact, body.consent, body.retention_days)}
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/widget.js", include_in_schema=False)
def widget_script():
    return FileResponse(Path(__file__).resolve().parent / "static" / "widget.js", media_type="application/javascript")


@app.get("/customer", include_in_schema=False)
def customer_dashboard_redirect():
    """Keep old bookmarks useful while serving one shared product interface."""
    return RedirectResponse("http://127.0.0.1:8501", status_code=307)
