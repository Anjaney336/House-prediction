from __future__ import annotations

import plotly.express as px
import pandas as pd
import streamlit as st

from app.components.charts import importance_chart, predicted_actual_chart, residual_chart
from app.components.sidebar import initialize_state, render_sidebar, require_model
from src.models.explainer import feature_importance, shap_summary_data
from src.models.uncertainty import coverage_by_group


st.set_page_config(page_title="Model Insights · PricePredict AI", page_icon="📊", layout="wide")
initialize_state()
render_sidebar(4)
expert = bool(st.session_state.get("expert_mode", False))
st.title("📈 Why the Model Makes Its Estimates" if not expert else "📊 Model Insights")
if not require_model():
    st.stop()

result = st.session_state.training_result
name = result.active_model_name
prediction = result.test_predictions[name]
actual = result.y_test
row = result.leaderboard.loc[result.leaderboard["Model"] == name].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active model", name)
c2.metric("Test R²", f"{row['Test R²']:.3f}")
c3.metric("RMSE", f"{row['Test RMSE']:,.0f}")
c4.metric("MAPE", f"{row['Test MAPE (%)']:.1f}%")

try:
    importance = feature_importance(result.active_pipeline, result.X_test, result.y_test)
    st.subheader("Key Model Drivers")
    st.plotly_chart(importance_chart(importance), width="stretch")
    if not expert:
        top_features = importance.head(5)["feature"].astype(str).str.replace("_", " ").str.title().tolist()
        st.write("The model relies most on: " + ", ".join(top_features) + ".")
        st.caption("These are predictive associations found in the historical dataset, not proof of cause and effect.")
except Exception as exc:
    st.info(f"Feature importance is unavailable for this model: {exc}")

if not expert:
    st.stop()

with st.expander("SHAP global explanation", expanded=False):
    st.caption("SHAP can take several seconds; it runs only when requested and uses a bounded holdout sample.")
    if st.button("Generate SHAP summary"):
        try:
            with st.spinner("Calculating SHAP values…"):
                shap_global, shap_points = shap_summary_data(result.active_pipeline, result.X_test)
            shap_bar = px.bar(
                shap_global.sort_values("mean_abs_shap"),
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title="Mean absolute SHAP value",
            )
            st.plotly_chart(shap_bar, width="stretch")
            beeswarm = px.strip(
                shap_points,
                x="shap_value",
                y="feature",
                color="feature_value",
                color_continuous_scale="RdBu",
                title="SHAP contribution distribution",
            )
            st.plotly_chart(beeswarm, width="stretch")
        except Exception as exc:
            st.info(f"SHAP summary is unavailable for this model: {exc}")

left, right = st.columns(2)
with left:
    st.subheader("Predicted vs actual")
    st.plotly_chart(predicted_actual_chart(actual, prediction), width="stretch")
with right:
    st.subheader("Residuals")
    st.plotly_chart(residual_chart(actual, prediction), width="stretch")

residuals = actual.to_numpy() - prediction
st.subheader("Error distribution")
st.plotly_chart(px.histogram(x=residuals, nbins=40, labels={"x": "Residual"}), width="stretch")

st.subheader("Adaptive error analysis")
error_frame = result.X_test.copy()
error_frame["actual"] = actual
error_frame["predicted"] = prediction
error_frame["absolute_error"] = abs(error_frame["actual"] - error_frame["predicted"])
try:
    error_frame["target_segment"] = pd.qcut(error_frame["actual"], q=min(4, error_frame["actual"].nunique()), duplicates="drop")
    segment = error_frame.groupby("target_segment", observed=True)["absolute_error"].mean().reset_index()
    segment["target_segment"] = segment["target_segment"].astype(str)
    st.plotly_chart(px.bar(segment, x="target_segment", y="absolute_error", title="Mean absolute error by target-value segment"), width="stretch")
except ValueError:
    st.info("The target has too few unique values for segmented error analysis.")

categorical = result.X_test.select_dtypes(exclude="number").columns.tolist()
if categorical:
    segment_column = st.selectbox("Analyze error by categorical segment", categorical)
    grouped = error_frame.groupby(segment_column, dropna=False)["absolute_error"].agg(["mean", "count"]).reset_index().sort_values("count", ascending=False).head(20)
    st.plotly_chart(px.bar(grouped, x=segment_column, y="mean", hover_data=["count"], title=f"Mean absolute error by {segment_column}"), width="stretch")
st.caption("A narrow distribution centered near zero indicates well-calibrated errors. Large train–test gaps on the leaderboard may indicate overfitting.")

radius = getattr(result, "conformal_radius", {}).get(name)
if radius is not None:
    st.subheader("Calibrated interval audit")
    coverage = getattr(result, "conformal_coverage", {}).get(name, float("nan"))
    i1, i2, i3 = st.columns(3)
    i1.metric("Coverage target", "95%")
    i2.metric("Final-test coverage", f"{coverage:.1%}")
    i3.metric("Interval width", f"{2 * radius:,.0f}")
    target_bands = pd.qcut(actual, q=min(4, actual.nunique()), duplicates="drop").astype(str)
    band_coverage = coverage_by_group(actual, prediction, radius, target_bands, minimum_rows=3)
    if not band_coverage.empty:
        st.plotly_chart(px.bar(band_coverage, x="group", y="coverage", hover_data=["rows"], title="Interval coverage by target-value segment"), width="stretch")
    st.caption("Split-conformal coverage is a marginal guarantee under exchangeability, not a guarantee for every location, property type, or shifted future market.")
