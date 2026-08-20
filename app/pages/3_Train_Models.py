from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app.components.charts import model_leaderboard_chart
from app.components.sidebar import initialize_state, render_sidebar, require_data
from src.data.cleaner import CleaningConfig
from src.models.model_catalog import assess_eligibility, model_catalog
from src.utils.errors import user_error
from src.utils.runtime import require_module_api
from src.utils.schema import validate_training_frame
from src.validation.schema_contract import build_schema_contract


trainer_module = require_module_api("src.models.trainer", required_parameters={"train_regressors": {"metadata", "mode", "tune_top_n"}})
registry_module = require_module_api("src.models.registry", required_attributes={"generate_model_id"})
train_regressors = trainer_module.train_regressors
activate_model = registry_module.activate_model
save_active_model = registry_module.save_active_model


st.set_page_config(page_title="Model Lab · PricePredict AI", page_icon="🧪", layout="wide")
initialize_state()
render_sidebar(3)
expert = bool(st.session_state.get("expert_mode", False))
st.title("✨ Build Model" if not expert else "🧪 Real Estate Regression & AutoML Model Lab")
st.caption("Choose how much optimization you want. Balanced is recommended for most datasets." if not expert else "Dataset-aware model screening, focused optimization, robust validation, and auditable selection.")
if not require_data():
    st.stop()

df, target, features = st.session_state.dataframe, st.session_state.target, st.session_state.features
errors = validate_training_frame(df, target, features) if target else ["Confirm a target on the Upload Data page."]
if errors:
    for error in errors:
        st.error(error)
    st.stop()

high_leakage = [warning for warning in st.session_state.leakage_warnings if warning.severity == "high"]
if high_leakage:
    st.error("High-risk leakage features are selected and may produce unrealistically strong validation scores.")
    if not st.checkbox("I confirm these fields are available before the valuation outcome."):
        st.stop()

summary_columns = st.columns(4)
summary_columns[0].metric("Rows", f"{len(df):,}")
summary_columns[1].metric("Features", len(features))
training_matrix = df.loc[:, features]
summary_columns[2].metric("Numeric", len(training_matrix.select_dtypes("number").columns))
summary_columns[3].metric("Categorical", len(features) - len(training_matrix.select_dtypes("number").columns))

st.subheader("Choose a build mode")
mode_label = st.segmented_control("Build mode", ["Quick", "Balanced", "Advanced"], default="Balanced")
mode = {"Quick": "Fast", "Balanced": "Balanced", "Advanced": "Comprehensive"}[mode_label or "Balanced"]
descriptions = st.columns(3)
descriptions[0].caption("Quick · Fast screening for an immediate baseline.")
descriptions[1].caption("Balanced · Recommended validation and focused optimization.")
descriptions[2].caption("Advanced · Wider model laboratory and ensemble experiments.")
catalog = model_catalog()
selected_models = None
if expert:
    families = sorted({spec.family for spec in catalog.values()})
    selected_families = st.multiselect("Model families", families, default=families)
    choices = [name for name, spec in catalog.items() if spec.family in selected_families]
    selected_models = st.multiselect("Models", choices, default=choices)

test_percent, folds, transform = 20, 5, "Automatic"
include_optional, tune = mode != "Fast", mode != "Fast"
tune_top_n, iterations, include_ensembles = 3, 5, mode == "Comprehensive"
validation_strategy, ranking_weights = "auto", None
if expert:
    c1, c2, c3, c4 = st.columns(4)
    test_percent = c1.slider("Holdout set (%)", 10, 35, 20)
    folds = c2.slider("Validation folds", 3, 10, 5)
    transform = c3.selectbox("Target transform", ["Automatic", "None", "log1p"], index=0)
    include_optional = c4.toggle("External boosters", value=mode != "Fast")
    with st.expander("Advanced validation and optimization", expanded=False):
        o1, o2, o3, o4 = st.columns(4)
        tune = o1.toggle("Optimize finalists", value=mode != "Fast")
        tune_top_n = o2.slider("Finalists to optimize", 1, 8, 3, disabled=not tune)
        iterations = o3.slider("Search trials per finalist", 1, 30, 5, disabled=not tune)
        include_ensembles = o4.toggle("Evaluate ensembles", value=mode == "Comprehensive")
        validation_strategy = st.selectbox("Validation strategy", ["auto", "random", "time", "group", "geographic"])
        st.caption("Ranking weights can be changed for sensitivity analysis; values are normalized automatically.")
        w1, w2, w3, w4 = st.columns(4)
        ranking_weights = {
            "performance": w1.number_input("Performance", 0.0, 1.0, 0.60, 0.05),
            "stability": w2.number_input("Stability", 0.0, 1.0, 0.20, 0.05),
            "simplicity": w3.number_input("Simplicity", 0.0, 1.0, 0.10, 0.05),
            "generalization": w4.number_input("Generalization", 0.0, 1.0, 0.10, 0.05),
        }

eligible, preview_availability = assess_eligibility(training_matrix, mode or "Balanced", selected_models, include_optional)
if expert:
    with st.expander(f"Model availability matrix · {len(eligible)} eligible", expanded=False):
        st.dataframe(preview_availability, width="stretch", hide_index=True)

st.info(f"Training {len(eligible)} eligible models with consistent validation. The final test set remains separate from model tuning.")

if st.button("Run AutoML experiment", type="primary", width="stretch", disabled=not eligible):
    progress_bar = st.progress(0, text="Preparing leakage-safe training data…")

    def update(value: float, label: str) -> None:
        progress_bar.progress(min(1.0, value), text=label)

    try:
        domain, quality = st.session_state.domain_analysis, st.session_state.quality_assessment
        cleaning = st.session_state.cleaning_config or CleaningConfig()
        contract = build_schema_contract(
            df, features, target, domain, currency=st.session_state.currency,
            numeric_strategy=cleaning.numeric_imputation,
            categorical_strategy=cleaning.categorical_imputation,
        )
        governance = {
            "dataset_name": st.session_state.dataset_name,
            "asset_type": domain.asset_type,
            "prediction_granularity": domain.granularity,
            "dataset_domain": domain.domain,
            "schema_contract": contract,
            "quality": quality.to_dict() if quality else None,
            "known_limitations": [
                "Predictions depend on the uploaded historical data and are not guaranteed appraisals.",
                "Unobserved market shifts and omitted location variables can reduce out-of-sample accuracy.",
            ],
        }
        log_target = None if transform == "Automatic" else transform == "log1p"
        result = train_regressors(
            df, target, features, cleaning_config=cleaning,
            test_size=test_percent / 100, cv_folds=folds, log_target=log_target,
            tune=tune, tuning_iterations=iterations, progress=update, metadata=governance,
            mode=mode or "Balanced", selected_models=selected_models, tune_top_n=tune_top_n,
            include_optional_boosters=include_optional, include_ensembles=include_ensembles,
            validation_strategy=validation_strategy, ranking_weights=ranking_weights,
        )
        st.session_state.training_result = result
        st.session_state.schema_contract = contract
        st.session_state.model_path = str(save_active_model(result, st.session_state.dataset_key, target, features))
        progress_bar.empty()
        st.success(f"Experiment {result.metadata['experiment_id']} complete. {result.active_model_name} is active.")
    except Exception as exc:
        progress_bar.empty()
        message, incident = user_error(exc, "Model training")
        st.error(message)
        st.caption(f"Diagnostic reference: {incident}")

result = st.session_state.training_result
if result is None:
    st.stop()

st.divider()
st.subheader("Best validated model found")
meta = result.metadata
selected_row = result.leaderboard.loc[result.leaderboard["Model"] == result.active_model_name].iloc[0]
result_metrics = st.columns(6)
result_metrics[0].metric("Best Model", result.active_model_name)
result_metrics[1].metric("MAE", f"{selected_row['Test MAE']:,.0f}")
result_metrics[2].metric("RMSE", f"{selected_row['Test RMSE']:,.0f}")
result_metrics[3].metric("R²", f"{selected_row['Test R²']:.3f}")
stability = max(0.0, 100 * (1 - selected_row["CV RMSE Std"] / max(selected_row["CV RMSE"], 1e-9)))
result_metrics[4].metric("Stability", f"{stability:.0f}/100")
result_metrics[5].metric("Training Time", f"{selected_row['Training Time (s)']:.1f}s")
st.caption(meta.get("validation_reason", ""))
quality = st.session_state.quality_assessment
st.markdown("#### Why this model?")
st.write(f"{result.active_model_name} achieved the strongest combined score across predictive error, consistency between validation folds, model simplicity, and train-to-test generalization.")
reliability = quality.overall if quality else 50
coverage = selected_row.get("95% Interval Coverage")
coverage_text = f" · Calibrated-range test coverage: {coverage:.1%}" if coverage is not None else ""
st.info(f"Model performance: R² {selected_row['Test R²']:.3f} · Data reliability: {reliability}/100{coverage_text}. These are separate signals and should be interpreted together.")
if not expert:
    st.page_link("pages/5_Predict.py", label="Value a Property", icon="🏷️")
    st.stop()

tab_leaderboard, tab_stability, tab_tuning, tab_availability, tab_governance = st.tabs(
    ["Leaderboard", "Fold stability", "Optimization", "Availability", "Governance"]
)
with tab_leaderboard:
    display = result.leaderboard.copy()
    numeric_columns = display.select_dtypes("number").columns
    display[numeric_columns] = display[numeric_columns].round(3)
    st.caption(meta.get("ranking_formula", "Composite performance ranking"))
    st.dataframe(display, width="stretch", hide_index=True)
    st.plotly_chart(model_leaderboard_chart(result.leaderboard), width="stretch")

with tab_stability:
    fold_frame = pd.DataFrame(result.fold_scores)
    if not fold_frame.empty:
        fold_frame.index = [f"Fold {index + 1}" for index in range(len(fold_frame))]
        st.line_chart(fold_frame)
        st.dataframe(fold_frame.round(3), width="stretch")
    else:
        st.info("Fold-level metrics are unavailable for this legacy experiment.")

with tab_tuning:
    if result.tuning_results:
        for model_name, details in result.tuning_results.items():
            with st.expander(model_name):
                st.json(details)
    else:
        st.info("No optimization was run. Screening estimates were used directly.")

with tab_availability:
    st.dataframe(result.availability, width="stretch", hide_index=True)

with tab_governance:
    st.json({key: value for key, value in meta.items() if key != "schema_contract"})
    report = [
        "# PricePredict AI — Model Selection Report", "",
        f"- Experiment: {meta.get('experiment_id')}",
        f"- Dataset: {meta.get('dataset_name')}",
        f"- Selected model: {result.active_model_name}",
        f"- Validation: {meta.get('validation_strategy')} — {meta.get('validation_reason')}",
        f"- Target transformation: {meta.get('target_transformation')}",
        f"- Ranking: {meta.get('ranking_formula')}", "", "## Leaderboard", "",
        "```csv", result.leaderboard.to_csv(index=False), "```", "", "## Model availability", "",
        "```csv", result.availability.to_csv(index=False), "```", "", "## Reproducibility metadata", "",
        "```json", json.dumps(meta.get("software", {}), indent=2), "```",
    ]
    st.download_button("Download model-selection report", "\n".join(report), file_name=f"{meta.get('experiment_id', 'experiment')}-report.md", mime="text/markdown")

choice = st.selectbox("Active model override", result.leaderboard["Model"].tolist(), index=result.leaderboard["Model"].tolist().index(result.active_model_name))
if st.button("Activate selected model"):
    activate_model(result, choice)
    st.session_state.model_path = str(save_active_model(result, st.session_state.dataset_key, target, features))
    st.success(f"{choice} is now active.")

if st.session_state.model_path:
    path = st.session_state.model_path
    with open(path, "rb") as artifact:
        st.download_button("Download active model artifact", artifact.read(), file_name=path.split("\\")[-1], mime="application/octet-stream")
