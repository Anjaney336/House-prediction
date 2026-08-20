from __future__ import annotations

import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.sidebar import initialize_state, render_sidebar
from src.platform.persistence import initialize_database
from src.platform.service import approve_model, datasets_for_tenant, models_for_tenant, publish_model


st.set_page_config(page_title="Platform API · PricePredict AI", page_icon="🔌", layout="wide")
initialize_state(); render_sidebar(10)
st.title("🔌 Embeddable Platform API")
if not st.session_state.get("expert_mode", False):
    st.info("Enable Expert Mode to inspect platform delivery and tenant-scoped records.")
    st.stop()

st.caption("The public widget and FastAPI service are separate from this analyst console and share the same validated ML core.")
initialize_database()
tenant = st.text_input("Tenant ID", value="operator", help="Records are always queried within this tenant boundary.")

cards = st.columns(4)
cards[0].metric("API", "Configured" if os.getenv("PRICEPREDICT_API_KEY") else "Key required")
cards[1].metric("Single prediction", "Rate limited")
cards[2].metric("Training / batch", "API key required")
cards[3].metric("Storage", "Durable SQL + tenant files")

if tenant:
    try:
        datasets = datasets_for_tenant(tenant)
        models = models_for_tenant(tenant)
        st.subheader("Datasets")
        st.dataframe(pd.DataFrame([{key: value for key, value in row.items() if key != "schema_contract"} for row in datasets]), width="stretch", hide_index=True)
        st.subheader("Models")
        st.dataframe(pd.DataFrame([{key: value for key, value in row.items() if key != "model_card"} for row in models]), width="stretch", hide_index=True)
        if models:
            selected_id = st.selectbox("Lifecycle model", [row["id"] for row in models])
            selected = next(row for row in models if row["id"] == selected_id)
            st.json(selected["model_card"])
            action_left, action_right = st.columns(2)
            if action_left.button("Approve validated model", disabled=selected["status"] not in {"VALIDATED", "READY"}):
                try:
                    approve_model(selected_id, tenant); st.success("Model approved. Review provenance and confirmations before publication."); st.rerun()
                except (KeyError, ValueError) as exc:
                    st.error(str(exc))
            if action_right.button("Publish approved model", disabled=selected["status"] != "APPROVED"):
                try:
                    publish_model(selected_id, tenant); st.success("Model published to the customer router."); st.rerun()
                except (KeyError, ValueError) as exc:
                    st.error(str(exc))
    except ValueError as exc:
        st.error(str(exc))

st.subheader("Run the API")
st.code("$env:PRICEPREDICT_API_KEY='replace-with-a-secret'\npython -m uvicorn backend.api:app --host 127.0.0.1 --port 8765", language="powershell")
st.subheader("Embed a published model")
st.code('<div id="pricepredict-widget"></div>\n<script src="http://127.0.0.1:8765/widget.js" data-api="http://127.0.0.1:8765" data-tenant="operator" data-routing="true" data-mount="pricepredict-widget"></script>', language="html")
st.subheader("Production Risk Register")
risk_path = Path("data") / "production_risk_register.json"
if risk_path.exists():
    st.dataframe(pd.DataFrame(json.loads(risk_path.read_text(encoding="utf-8"))), width="stretch", hide_index=True)
st.warning("Use HTTPS, a strong secret, restricted CORS origins, managed SQL/object storage, and an external rate-limit store before internet deployment.")
