from __future__ import annotations

import streamlit as st


def initialize_state() -> None:
    defaults = {
        "dataframe": None,
        "dataset_key": None,
        "target": None,
        "features": [],
        "cleaning_config": None,
        "training_result": None,
        "model_path": None,
        "dataset_name": None,
        "domain_analysis": None,
        "market_analysis": None,
        "target_candidates": [],
        "operating_mode": None,
        "currency": "USD",
        "quality_assessment": None,
        "leakage_warnings": [],
        "schema_contract": None,
        "expert_mode": False,
        "last_benchmark": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state["state_schema_version"] = 2


def render_sidebar(current_step: int) -> None:
    initialize_state()
    st.sidebar.markdown("## PricePredict AI")
    st.sidebar.caption("AI-powered real estate valuation")
    expert_mode = bool(st.session_state.get("expert_mode", False))
    st.session_state["expert_mode"] = st.sidebar.toggle(
        "Expert Mode", value=expert_mode,
        help="Show model science, validation controls, benchmarks, and diagnostics.",
    )
    st.sidebar.page_link("main.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/1_Upload_Data.py", label="Data", icon="📁")
    st.sidebar.page_link("pages/2_Data_Cleaning.py", label="Data Preparation", icon="🧹")
    st.sidebar.page_link("pages/3_Train_Models.py", label="Model", icon="✨")
    st.sidebar.page_link("pages/5_Predict.py", label="Valuation", icon="🏷️")
    st.sidebar.page_link("pages/4_Model_Insights.py", label="Insights", icon="📈")
    st.sidebar.page_link("pages/6_Model_Registry.py", label="Reports & Models", icon="📄")
    if st.session_state.get("expert_mode", False):
        st.sidebar.markdown("#### Engineering Lab")
        st.sidebar.page_link("pages/7_Synthetic_Benchmark.py", label="Synthetic Benchmark", icon="🧬")
        st.sidebar.page_link("pages/8_Model_Science.py", label="Model Science", icon="🔬")
        st.sidebar.page_link("pages/9_System_Health.py", label="System Health", icon="🩺")
        st.sidebar.page_link("pages/10_Platform_API.py", label="Platform API", icon="🔌")
    st.sidebar.divider()
    if st.session_state.get("dataset_key") and st.session_state.get("expert_mode", False):
        st.sidebar.caption(f"Dataset ID: `{st.session_state.dataset_key}`")
    if st.session_state.target:
        st.sidebar.caption(f"Target: `{st.session_state.target}`")
    if st.session_state.get("operating_mode") and st.session_state.get("expert_mode", False):
        st.sidebar.caption(st.session_state.operating_mode)
    if st.session_state.domain_analysis:
        st.sidebar.caption(f"Granularity: {st.session_state.domain_analysis.granularity}")
    if st.session_state.get("market_analysis") and st.session_state.market_analysis.candidate:
        st.sidebar.caption(f"Market hypothesis: {st.session_state.market_analysis.candidate}")
    if st.session_state.training_result:
        st.sidebar.success(f"Active: {st.session_state.training_result.active_model_name}")


def require_data() -> bool:
    if st.session_state.get("dataframe") is None:
        st.warning("Upload or choose a dataset on the Upload Data page first.")
        return False
    return True


def require_model() -> bool:
    if st.session_state.get("training_result") is None:
        st.warning("Train at least one model on the Train Models page first.")
        return False
    return True
