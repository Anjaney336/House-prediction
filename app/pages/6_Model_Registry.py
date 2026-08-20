from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.components.sidebar import initialize_state, render_sidebar
from src.utils.config import MODEL_DIR


st.set_page_config(page_title="Model Registry · PricePredict AI", page_icon="🗂️", layout="wide")
initialize_state()
render_sidebar(6)
st.title("🗂️ Model registry")
st.caption("Governance metadata for models trained inside this application. External joblib uploads are intentionally not accepted.")

result = st.session_state.training_result
if result is not None:
    model_id = result.metadata.get("model_id", "Pending version")
    row = result.leaderboard.loc[result.leaderboard["Model"] == result.active_model_name].iloc[0]
    cards = st.columns(5)
    cards[0].metric("Model ID", model_id)
    cards[1].metric("Algorithm", result.active_model_name)
    cards[2].metric("CV RMSE", f"{row['CV RMSE']:,.2f}")
    cards[3].metric("Holdout R²", f"{row['Test R²']:.3f}")
    cards[4].metric("Status", "Active")
    with st.expander("Model card", expanded=True):
        st.json(
            {
                "model_id": model_id,
                "dataset": result.metadata.get("dataset_name"),
                "asset_type": result.metadata.get("asset_type"),
                "prediction_granularity": result.metadata.get("prediction_granularity"),
                "validation_strategy": result.metadata.get("validation_strategy"),
                "target": st.session_state.target,
                "features": st.session_state.features,
                "data_quality": result.metadata.get("quality"),
                "known_limitations": result.metadata.get("known_limitations"),
                "uncertainty_method": result.metadata.get("uncertainty_method"),
                "calibration_rows": result.metadata.get("calibration_rows"),
                "final_test_coverage": getattr(result, "conformal_coverage", {}).get(result.active_model_name),
            }
        )
else:
    st.info("No active model in this session. Train one in Model Lab to create a versioned model card.")

st.subheader("Trusted local artifacts")
artifacts = sorted(MODEL_DIR.glob("PP-*.joblib"), key=lambda path: path.stat().st_mtime, reverse=True)
if artifacts:
    st.dataframe(
        [
            {"Artifact": path.name, "Size (MB)": round(path.stat().st_size / 1024**2, 2), "Trusted source": "Trained by this application"}
            for path in artifacts
        ],
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No versioned artifacts have been trained yet.")

st.warning("Only load model artifacts produced and retained by this application. Joblib/pickle files from untrusted sources can execute code during deserialization.")
