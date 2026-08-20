from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.sidebar import initialize_state, render_sidebar
from src.models.model_catalog import model_catalog


st.set_page_config(page_title="System Health · PricePredict AI", page_icon="🩺", layout="wide")
initialize_state(); render_sidebar(9)
st.title("🩺 System Health")
if not st.session_state.get("expert_mode", False):
    st.info("Enable Expert Mode to inspect engineering diagnostics.")
    st.stop()

quality_path = Path("data") / "quality_gate.json"
quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
explanation_path = Path("data") / "benchmarks" / "explainability_validation_summary.json"
explanation = json.loads(explanation_path.read_text(encoding="utf-8")) if explanation_path.exists() else {}
explanation_healthy = find_spec("shap") and explanation.get("spearman_rank_correlation", 0) >= 0.4
zoo_path = Path("data") / "benchmarks" / "full_model_zoo_summary.json"
zoo = json.loads(zoo_path.read_text(encoding="utf-8")) if zoo_path.exists() else {}
zoo_failures = int(zoo.get("status_counts", {}).get("FAILED", 0))
selection_path = Path("data") / "benchmarks" / "multiseed_selection_validation.csv"
selection = pd.read_csv(selection_path) if selection_path.exists() else pd.DataFrame()
mean_regret = float(selection["AutoML Regret (%)"].mean()) if not selection.empty else None
mean_coverage = float(selection["Interval Coverage"].mean()) if not selection.empty else None
engine_rows = [
    {"Engine": "Data Engine", "Status": "Healthy" if st.session_state.dataframe is not None else "Warning", "Detail": "Dataset loaded" if st.session_state.dataframe is not None else "Awaiting dataset"},
    {"Engine": "Model Engine", "Status": "Warning" if zoo_failures else "Healthy", "Detail": f"{zoo_failures} optional model failure(s) isolated in latest full-zoo benchmark" if zoo_failures else "All benchmarked models healthy"},
    {"Engine": "Prediction Engine", "Status": "Healthy" if st.session_state.schema_contract is not None else "Warning", "Detail": "Schema contract active" if st.session_state.schema_contract is not None else "Awaiting trained contract"},
    {"Engine": "Validation Engine", "Status": "Healthy", "Detail": "Shuffled, grouped, chronological, and geographic strategies available"},
    {"Engine": "Uncertainty Engine", "Status": "Healthy" if mean_coverage is not None and 0.90 <= mean_coverage <= 0.99 else "Warning", "Detail": f"Split-conformal mean final-test coverage: {mean_coverage:.1%}" if mean_coverage is not None else "Calibration evidence not available"},
    {"Engine": "Selection Engine", "Status": "Warning" if mean_regret is not None and mean_regret > 10 else "Healthy", "Detail": f"Multi-seed mean holdout regret: {mean_regret:.1f}% (holdout is audit-only)" if mean_regret is not None else "Multi-seed evidence not available"},
    {"Engine": "Explainability Engine", "Status": "Healthy" if explanation_healthy else "Degraded", "Detail": f"Synthetic driver rank correlation: {explanation.get('spearman_rank_correlation', 'not tested')}"},
    {"Engine": "Benchmark Engine", "Status": "Healthy", "Detail": "Six synthetic markets and fault scenarios available"},
]
st.dataframe(pd.DataFrame(engine_rows), width="stretch", hide_index=True)

cards = st.columns(4)
cards[0].metric("Tests Passed", quality.get("tests_passed", "Run quality gate"))
cards[1].metric("Base Models", len(model_catalog()))
cards[2].metric("Last Benchmark", getattr(st.session_state.last_benchmark, "experiment_id", "None"))
cards[3].metric("Last Training", st.session_state.training_result.metadata.get("experiment_id", "None") if st.session_state.training_result else "None")

st.subheader("Optional Dependencies")
dependencies = pd.DataFrame([{
    "Package": package,
    "Status": ("Installed · latest benchmark failed safely" if package == "lightgbm" and zoo_failures else "Available") if find_spec(package) else "Unavailable · handled gracefully",
} for package in ["xgboost", "lightgbm", "catboost", "shap"]])
st.dataframe(dependencies, width="stretch", hide_index=True)
if quality:
    st.subheader("Last Quality Gate")
    st.json(quality)
