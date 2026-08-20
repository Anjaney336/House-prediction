from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.sidebar import initialize_state, render_sidebar
from src.benchmark.corruption import inject_duplicates, inject_leakage, inject_missingness, inject_outliers
from src.benchmark.faults import audit_faults, findings_frame
from src.benchmark.generators import MarketConfig, SyntheticDataset, generate_dataset
from src.benchmark.runner import ranking_sensitivity, run_benchmark, save_benchmark
from src.models.model_catalog import model_catalog
from src.utils.errors import user_error


st.set_page_config(page_title="Synthetic Benchmark · PricePredict AI", page_icon="🧬", layout="wide")
initialize_state(); render_sidebar(7)
st.title("🧬 Synthetic Real Estate Benchmark Lab")
if not st.session_state.get("expert_mode", False):
    st.info("Enable Expert Mode in the sidebar to use controlled synthetic-market experiments.")
    st.stop()
st.caption("Synthetic experiments test engineering behavior under known assumptions. They do not prove accuracy in a real market.")

with st.expander("Dataset Generator", expanded=True):
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox("Market", ["Apartments", "Villas", "Mixed Residential", "Land / Plots", "Commercial", "Rentals"])
    size_label = c2.selectbox("Size", ["Small · 500", "Medium · 5,000", "Large · 25,000", "Custom"])
    default_rows = {"Small · 500": 500, "Medium · 5,000": 5000, "Large · 25,000": 25000}.get(size_label, 1000)
    rows = c2.number_input("Rows", 50, 150000, default_rows, 50, disabled=size_label != "Custom") if size_label == "Custom" else default_rows
    seed = c3.selectbox("Seed", [1, 7, 21, 42, 100, 999], index=3)
    p1, p2, p3, p4 = st.columns(4)
    config = MarketConfig(
        base_market_level=p1.slider("Market level", 0.5, 2.0, 1.0, 0.05),
        location_premium=p2.slider("Location sensitivity", 0.5, 1.8, 1.0, 0.05),
        market_inflation=p3.slider("Annual inflation", 0.0, 0.20, 0.055, 0.005),
        noise=p4.slider("Market noise", 0.01, 0.30, 0.08, 0.01),
    )
    if st.button("Generate reproducible market", type="primary"):
        st.session_state.synthetic_dataset = generate_dataset(kind, int(rows), int(seed), config)

dataset = st.session_state.get("synthetic_dataset")
if dataset is None:
    st.stop()

st.subheader("Generated Dataset")
cards = st.columns(5)
cards[0].metric("Market", dataset.dataset_type.replace("_", " ").title())
cards[1].metric("Rows", f"{len(dataset.frame):,}")
cards[2].metric("Features", len(dataset.frame.columns) - 1)
cards[3].metric("Target", dataset.target)
cards[4].metric("Seed", dataset.seed)
st.dataframe(dataset.frame.head(100), width="stretch")
file_stem = f"synthetic_{dataset.dataset_type}_seed{dataset.seed}"
d1, d2 = st.columns(2)
d1.download_button("Download CSV", dataset.frame.to_csv(index=False), f"{file_stem}.csv", "text/csv")
d2.download_button("Download generation manifest", json.dumps(dataset.manifest(), indent=2, default=str), f"{file_stem}.json", "application/json")

st.subheader("Data Quality Scenario")
q1, q2, q3 = st.columns(3)
scenario = q1.selectbox("Scenario", ["Clean", "MCAR missingness", "MAR missingness", "Feature-dependent missingness", "Valid extremes", "Impossible values", "Exact duplicates", "Near-duplicate listings", "Target leakage"])
level = q2.slider("Affected fraction", 0.0, 0.20, 0.05, 0.01)
corruption_seed = q3.selectbox("Corruption seed", [1, 7, 21, 42, 100, 999], index=1)
frame = dataset.frame
if scenario == "MCAR missingness": frame = inject_missingness(frame, level, corruption_seed, "MCAR").frame
elif scenario == "MAR missingness": frame = inject_missingness(frame, level, corruption_seed, "MAR").frame
elif scenario == "Feature-dependent missingness": frame = inject_missingness(frame, level, corruption_seed, "FEATURE_DEPENDENT").frame
elif scenario == "Valid extremes": frame = inject_outliers(frame, level, corruption_seed, False).frame
elif scenario == "Impossible values": frame = inject_outliers(frame, level, corruption_seed, True).frame
elif scenario == "Exact duplicates": frame = inject_duplicates(frame, level, corruption_seed, False).frame
elif scenario == "Near-duplicate listings": frame = inject_duplicates(frame, level, corruption_seed, True).frame
elif scenario == "Target leakage": frame = inject_leakage(frame, dataset.target).frame
variant = SyntheticDataset(frame, dataset.dataset_type, dataset.target, dataset.seed, dataset.parameters, dataset.ground_truth)
findings = audit_faults(frame, dataset.target)
st.dataframe(findings_frame(findings), width="stretch", hide_index=True)

st.subheader("Benchmark Configuration")
catalog = model_catalog()
default_models = ["Linear Regression", "Ridge", "Decision Tree", "Random Forest", "Gradient Boosting", "Histogram Gradient Boosting"]
selected = st.multiselect("Models", list(catalog), default=default_models)
b1, b2, b3 = st.columns(3)
validation = b1.selectbox("Validation", ["auto", "random", "time", "geographic"])
ensembles = b2.toggle("Evaluate ensembles", value=False)
optional = b3.toggle("Optional boosters", value=False)
if st.button("Run benchmark", type="primary", width="stretch", disabled=not selected):
    try:
        with st.spinner("Running controlled model experiment…"):
            run = run_benchmark(
                variant, mode="Comprehensive", selected_models=selected,
                corruption_level=scenario, include_optional=optional,
                include_ensembles=ensembles, validation_strategy=validation,
            )
            save_benchmark(run, Path("data") / "benchmarks")
            st.session_state.last_benchmark = run
    except Exception as exc:
        message, incident = user_error(exc, "Synthetic benchmark")
        st.error(message); st.caption(f"Diagnostic reference: {incident}")

run = st.session_state.last_benchmark
if run is not None:
    st.subheader("Benchmark Results")
    successful = run.results.loc[run.results.status == "SUCCESS"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Experiment", run.experiment_id)
    m2.metric("Selected", run.training_result.active_model_name)
    m3.metric("AutoML Regret", f"{run.automl_regret_percent:.2f}%")
    m4.metric("Successful Models", len(successful))
    st.dataframe(run.results, width="stretch", hide_index=True)
    chart1, chart2 = st.columns(2)
    chart1.plotly_chart(px.bar(successful, x="model", y="holdout_rmse", color="model_family", title="Holdout RMSE"), width="stretch")
    chart2.plotly_chart(px.scatter(successful, x="train_time", y="cv_rmse", color="model", title="Performance vs Runtime"), width="stretch")
    st.subheader("Ground-Truth Driver Recovery")
    st.dataframe(run.ground_truth_recovery, width="stretch", hide_index=True)
    st.caption("Observed importance is permutation-based and compared with known generator drivers; agreement is evidence about this synthetic scenario only.")
    st.subheader("Ranking Sensitivity")
    st.dataframe(ranking_sensitivity(run.training_result.leaderboard), width="stretch", hide_index=True)

evidence_dir = Path("data") / "benchmarks"
curve_path = evidence_dir / "missingness_robustness_curve.csv"
shift_path = evidence_dir / "distribution_shift_results.csv"
validation_path = evidence_dir / "validation_strategy_comparison.csv"
explanation_path = evidence_dir / "explainability_validation_summary.json"
selection_path = evidence_dir / "multiseed_selection_validation.csv"
ranking_policy_path = evidence_dir / "ranking_policy_validation.csv"
if curve_path.exists() or shift_path.exists() or validation_path.exists():
    st.subheader("Saved Robustness Evidence")
    robustness_tab, shift_tab, validation_tab, selection_tab = st.tabs(["Missingness", "Market Shift", "Validation Strategy", "Selection & Coverage"])
    with robustness_tab:
        if curve_path.exists():
            curve = pd.read_csv(curve_path)
            st.plotly_chart(px.line(curve, x="missingness", y="holdout_rmse", color="model", markers=True, title="RMSE vs Missingness"), width="stretch")
    with shift_tab:
        if shift_path.exists():
            shifts = pd.read_csv(shift_path)
            st.plotly_chart(px.bar(shifts, x="Dataset", y="Degradation (%)", color="Model", title="Controlled 20% Market-Shift Degradation"), width="stretch")
    with validation_tab:
        if validation_path.exists():
            comparison = pd.read_csv(validation_path)
            st.plotly_chart(px.bar(comparison, x="Applied Strategy", y="Holdout RMSE", title="Validation Strategy Comparison"), width="stretch")
    with selection_tab:
        if selection_path.exists():
            selection_evidence = pd.read_csv(selection_path)
            st.plotly_chart(px.bar(selection_evidence, x="Seed", y="AutoML Regret (%)", color="Selected", title="Holdout Selection Regret Across Seeds"), width="stretch")
            st.metric("Mean calibrated interval coverage", f"{selection_evidence['Interval Coverage'].mean():.1%}")
            st.caption("Holdout regret and coverage are audit diagnostics only. The final test split is never used to select or tune a model.")
        if ranking_policy_path.exists():
            st.dataframe(pd.read_csv(ranking_policy_path), width="stretch", hide_index=True)
    if explanation_path.exists():
        explanation_evidence = json.loads(explanation_path.read_text(encoding="utf-8"))
        correlation = explanation_evidence.get("spearman_rank_correlation", 0)
        (st.success if correlation >= 0.4 else st.warning)(
            f"Explainability recovery: rank correlation {correlation:.2f}; top-3 overlap {explanation_evidence.get('top_3_driver_overlap', 0)}/3. "
            "Weak recovery means feature explanations should be treated cautiously."
        )
