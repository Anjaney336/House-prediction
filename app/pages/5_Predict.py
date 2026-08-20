from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.forms import contract_input_form
from app.components.sidebar import initialize_state, render_sidebar, require_model
from src.models.explainer import local_contributions
from src.models.uncertainty import model_confidence_score
from src.reports.valuation_report import build_valuation_report
from src.utils.formatting import format_value
from src.utils.errors import user_error
from src.validation.prediction_validator import validate_and_prepare
from src.utils.runtime import require_module_api


predictor_module = require_module_api("src.models.predictor", required_parameters={"predict_batch": {"contract"}})
predict_batch = predictor_module.predict_batch
prediction_interval = predictor_module.prediction_interval
similar_rows = predictor_module.similar_rows


st.set_page_config(page_title="Predict · PricePredict AI", page_icon="🔮", layout="wide")
initialize_state()
render_sidebar(5)
st.title("🏷️ Property Valuation")
if not require_model():
    st.stop()

df = st.session_state.dataframe
features = st.session_state.features
target = st.session_state.target
result = st.session_state.training_result
pipeline = result.active_pipeline
contract = st.session_state.schema_contract or result.metadata.get("schema_contract")
if contract is None:
    st.error("This model predates the schema-contract upgrade. Retrain it once from Model Lab before making predictions.")
    st.stop()
domain = st.session_state.domain_analysis
quality = st.session_state.quality_assessment
model_id = result.metadata.get("model_id", "Unversioned model")

st.caption(f"Model `{model_id}` · {contract.asset_type} · {contract.prediction_granularity}")
if contract.prediction_granularity == "Census Block / Geographic Area":
    st.warning("This is geographic/block-level data. The output estimates an area-level housing value pattern, not the exact market value of an individual house.")
elif contract.dataset_domain != "REAL_ESTATE":
    st.info("Generic Regression Mode is active. Property valuation terminology and appraisal claims are intentionally disabled.")

single, batch = st.tabs(["Single prediction", "Batch prediction"])
with single:
    row, imputed = contract_input_form(contract)
    if row is not None:
        try:
            validation = validate_and_prepare(row, contract)
            prediction = float(predict_batch(pipeline, validation.prepared, features, contract)[0])
            calibrated_radius = getattr(result, "conformal_radius", {}).get(result.active_model_name)
            low, high = prediction_interval(
                prediction, result.residual_std.get(result.active_model_name),
                calibrated_radius=calibrated_radius,
            )
            metric_row = result.leaderboard.loc[result.leaderboard["Model"] == result.active_model_name].iloc[0]
            completeness = 1 - len(imputed) / max(len(features), 1)
            confidence = model_confidence_score(
                float(metric_row["Test R²"]),
                float(metric_row["Test MAPE (%)"]),
                completeness,
                quality.overall if quality else 50,
                stability_score=max(0.0, 1 - float(metric_row["CV RMSE Std"]) / max(float(metric_row["CV RMSE"]), 1e-9)),
                generalization_score=max(0.0, 1 - float(metric_row["Overfit Gap"])),
            )
            value_col, range_col, confidence_col = st.columns(3)
            value_col.metric(contract.prediction_label, format_value(prediction, contract.currency))
            range_col.metric("Calibrated 95% range" if calibrated_radius is not None else "Model-based range", f"{format_value(low, contract.currency)} – {format_value(high, contract.currency)}")
            confidence_col.metric("Model Confidence Score", f"{confidence[0]}/100 · {confidence[1]}")
            st.info(f"Model performance: R² {metric_row['Test R²']:.3f} · Data reliability: {quality.overall if quality else 50}/100")
            if calibrated_radius is not None:
                empirical_coverage = getattr(result, "conformal_coverage", {}).get(result.active_model_name)
                st.caption(f"The range uses a held-out calibration split with a 95% marginal coverage target. Final-test coverage was {empirical_coverage:.1%}. Coverage is not guaranteed for every property subgroup or shifted market.")
            else:
                st.caption("The confidence score is a transparent model/data-quality heuristic, not a probability. This legacy range reflects residual dispersion and is not guaranteed future coverage.")
            if imputed:
                labels = {spec.name: spec.label for spec in contract.features}
                st.warning(f"{len(imputed)} input(s) were intentionally left unknown and imputed by the trained pipeline: {', '.join(labels[name] for name in imputed)}.")

            st.subheader("Why this prediction?")
            try:
                contributions = local_contributions(pipeline, row, result.X_train)
                top = contributions.head(15).sort_values("contribution")
                figure = px.bar(top, x="contribution", y="feature", orientation="h", color="contribution", color_continuous_scale="RdBu", title="Features that contributed to the model prediction")
                st.plotly_chart(figure, width="stretch")
                positive = top.nlargest(3, "contribution")["feature"].tolist()
                negative = top.nsmallest(3, "contribution")["feature"].tolist()
                st.write("**Top positive contributors:** " + (", ".join(positive) or "None"))
                st.write("**Top negative contributors:** " + (", ".join(negative) or "None"))
                st.caption("These are predictive associations in the fitted model, not causal claims.")
            except Exception as exc:
                st.info("Detailed contribution analysis is not available for this model. The estimate and validation metrics remain available.")

            if domain and domain.comparables_supported:
                st.subheader("Similar historical records")
                indices = similar_rows(pipeline, result.X_train, validation.prepared, n=5)
                similar = df.loc[indices, features + [target]].copy()
                similar["model_prediction"] = pipeline.predict(similar[features])
                st.dataframe(similar, width="stretch")
                st.caption("Similarity is measured in the trained feature space. These records are context, not certified appraisal comparables.")
            else:
                st.info("Comparable analysis is unavailable because this dataset is not clearly property-level or lacks sufficient valuation signals.")

            report_warnings = [
                "This is not a legal or guaranteed market appraisal.",
                *( ["Area/block-level data cannot support an exact individual-property claim."] if contract.prediction_granularity == "Census Block / Geographic Area" else []),
            ]
            report = build_valuation_report(
                row,
                prediction,
                (low, high),
                confidence,
                contract,
                result.active_model_name,
                model_id,
                metric_row.to_dict(),
                quality.overall if quality else 50,
                report_warnings,
                interval_method=result.metadata.get("uncertainty_method", "Model-based residual range"),
            )
            st.download_button("Download valuation report", report.encode("utf-8"), file_name=f"{model_id}_valuation_report.md", mime="text/markdown")
        except Exception as exc:
            message, incident = user_error(exc, "Property valuation")
            st.error(message)
            st.caption(f"Diagnostic reference: {incident}")

with batch:
    upload = st.file_uploader("Upload rows using the same feature schema", type=["csv"], key="batch_file")
    if upload is not None:
        try:
            batch_df = pd.read_csv(upload)
            validation = validate_and_prepare(batch_df, contract)
            summary = st.columns(4)
            summary[0].metric("Valid rows", validation.valid_rows)
            summary[1].metric("Invalid rows", validation.invalid_rows)
            summary[2].metric("Missing columns", len(validation.missing_columns))
            summary[3].metric("Ignored columns", len(validation.unexpected_columns))
            for warning in validation.warnings:
                st.warning(warning)
            if validation.valid_rows == 0:
                st.error("No rows satisfy the trained model schema. Correct the reported fields and upload again.")
                st.stop()
            predictions = predict_batch(pipeline, validation.prepared, features, contract)
            output = batch_df.loc[validation.prepared.index].copy()
            output[f"predicted_{target}"] = predictions
            st.dataframe(output.head(100), width="stretch")
            st.download_button(
                "Download predictions CSV",
                output.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            message, incident = user_error(exc, "Batch valuation")
            st.error(message)
            st.caption(f"Diagnostic reference: {incident}")
