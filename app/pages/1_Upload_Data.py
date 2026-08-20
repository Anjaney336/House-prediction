from __future__ import annotations

import streamlit as st

from app.components.charts import missingness_chart
from app.components.sidebar import initialize_state, render_sidebar
from src.data.loader import DataLoadError, dataset_hash, load_csv
from src.data.profiler import data_quality_summary, profile_dataframe
from src.data.quality import assess_quality
from src.domain.domain_detector import analyze_domain
from src.domain.market_intelligence import detect_market
from src.domain.target_detector import detect_targets
from src.features.leakage import detect_leakage
from src.utils.config import SAMPLE_DATASET
from src.utils.schema import suggest_features, suggest_target


st.set_page_config(page_title="Upload Data · PricePredict AI", page_icon="📂", layout="wide")
initialize_state()
render_sidebar(1)
expert = bool(st.session_state.get("expert_mode", False))
st.title("📂 Dataset Check")
st.caption("Upload property data or use a demo. PricePredict AI will check suitability before modeling.")


@st.cache_data(show_spinner=False)
def cached_upload(raw: bytes):
    return load_csv(raw)


@st.cache_data(show_spinner=False)
def cached_profile(frame):
    return profile_dataframe(frame)


def activate_dataset(frame, source_key: str, dataset_name: str) -> None:
    """Install a new dataset and invalidate every downstream schema/model artifact."""
    st.session_state.dataframe = frame
    st.session_state.source_key = source_key
    st.session_state.dataset_name = dataset_name
    for key, value in {
        "dataset_key": None, "target": None, "features": [], "cleaning_config": None,
        "training_result": None, "model_path": None, "domain_analysis": None, "market_analysis": None,
        "target_candidates": [], "operating_mode": None, "quality_assessment": None,
        "leakage_warnings": [], "schema_contract": None,
    }.items():
        st.session_state[key] = value

upload = st.file_uploader("Drop a CSV here", type=["csv"])
demo_options = {
    "California Housing · Block-level": SAMPLE_DATASET,
    "Individual Apartment Listings": SAMPLE_DATASET.parent / "individual_property_sample.csv",
    "Minimal Property Dataset": SAMPLE_DATASET.parent / "minimal_property_sample.csv",
    "Unrelated Retail · Rejection Demo": SAMPLE_DATASET.parent / "unrelated_retail_sample.csv",
    "Property Data Without Target": SAMPLE_DATASET.parent / "property_no_target_sample.csv",
}
demo_choice = st.selectbox("Bundled review datasets", list(demo_options))
sample = st.button("Load selected demo", width="content")

try:
    if upload is not None:
        raw = upload.getvalue()
        key = f"upload:{upload.name}:{len(raw)}"
        if st.session_state.get("source_key") != key:
            activate_dataset(cached_upload(raw), key, upload.name)
    elif sample:
        selected_demo = demo_options[demo_choice]
        activate_dataset(load_csv(selected_demo), f"sample:{selected_demo.name}", demo_choice)
except DataLoadError as exc:
    st.error(str(exc))

df = st.session_state.dataframe
if df is None:
    st.stop()

st.session_state.dataset_key = dataset_hash(df)
domain = analyze_domain(df)
market = detect_market(df, st.session_state.get("dataset_name"))
candidates = detect_targets(df)
st.session_state.domain_analysis = domain
st.session_state.market_analysis = market
st.session_state.target_candidates = candidates
summary = data_quality_summary(df)
if expert:
    metrics = st.columns(5)
    for box, (label, value) in zip(metrics, summary.items()):
        box.metric(label.replace("_", " ").title(), value)
domain_cards = st.columns(5)
domain_cards[0].metric("Dataset", st.session_state.dataset_name or "Uploaded CSV")
domain_cards[1].metric("Rows / Columns", f"{len(df):,} / {len(df.columns)}")
domain_cards[2].metric("Real Estate Confidence", f"{domain.confidence:.0%}")
domain_cards[3].metric("Asset Type", domain.asset_type)
domain_cards[4].metric("Market", f"{market.candidate or 'Unconfirmed'} · {market.confidence}")
if market.evidence:
    st.caption("Market evidence: " + " ".join(market.evidence) + " Admin confirmation is required before publication.")
if domain.is_real_estate:
    st.success("Ready · Real-estate structure detected." if domain.confidence >= 0.65 else "Warning · Some property signals are limited.")
    if domain.granularity == "Census Block / Geographic Area":
        st.warning("Dataset detected as geographic/block-level housing data. It can estimate area-level housing value patterns, but cannot support an exact valuation claim for a specific house.")
    mode = st.radio("Operating mode", ["Real Estate Intelligence", "Generic Regression Mode"], horizontal=True)
else:
    strongest = max(domain.scores, key=domain.scores.get).replace("_", " ").title()
    st.warning(f"Dataset not recognized as real-estate data. It appears closer to {strongest.lower()} data. Property-specific language and valuation claims are disabled.")
    mode = st.radio("Choose how to proceed", ["Stop / upload a different dataset", "Generic Regression Mode"], horizontal=True)
    if mode.startswith("Stop"):
        st.info("Upload a property dataset, or explicitly choose Generic Regression Mode.")
        st.stop()
st.session_state.operating_mode = mode

profile = cached_profile(df)
numeric_targets = df.select_dtypes("number").columns.tolist()
if not numeric_targets:
    st.error("This dataset has no numeric column that can be used as a regression target.")
    st.stop()
recommended_columns = [candidate.column for candidate in candidates]
if mode == "Real Estate Intelligence" and not recommended_columns:
    st.error("No supervised valuation target detected. This dataset has property features but no recognized historical price, value, or rent column. Use it for analytics, upload a valued dataset, or switch to Generic Regression Mode.")
    st.stop()
target_default = st.session_state.target if st.session_state.target in numeric_targets else (recommended_columns[0] if recommended_columns else suggest_target(df))
target = st.selectbox("Select prediction target", numeric_targets, index=numeric_targets.index(target_default)) if expert else target_default
if not expert:
    st.metric("Target Candidate", target)
if candidates and target == candidates[0].column:
    st.caption(f"Recommended target: **{target}** — {candidates[0].reason}")
elif mode == "Real Estate Intelligence":
    st.warning("The selected target is not a strongly recognized valuation target. Confirm that it represents a historical value or rent outcome.")
currencies = ["USD", "INR", "GBP", "EUR", "AED", "CAD", "AUD", "OTHER"]
currency_hypothesis = next((candidate.currency_hypothesis for candidate in candidates if candidate.column == target), None)
currency_default = st.session_state.get("currency") if st.session_state.get("currency") in currencies else currency_hypothesis
if currency_hypothesis and st.session_state.get("target") is None:
    currency_default = currency_hypothesis
currency = st.selectbox("Currency / value format · confirm before publishing", currencies, index=currencies.index(currency_default) if currency_default in currencies else 0)
default_features, excluded = suggest_features(df, target)
features = (
    st.multiselect(
        "Feature columns", [column for column in df.columns if column != target],
        default=[column for column in st.session_state.features if column in df.columns and column != target] or default_features,
    ) if expert else default_features
)
leakage = detect_leakage(df, target, features)
if leakage:
    for warning in leakage:
        (st.error if warning.severity == "high" else st.warning)(f"Potential target leakage — {warning.column}: {warning.reason}")
quality = assess_quality(df, target, features, domain, leakage)
quality_cols = st.columns(4 if expert else 3)
quality_cols[0].metric("Dataset Quality", f"{quality.overall}/100")
quality_cols[1].metric("Valuation Suitability", f"{quality.valuation_suitability}/100 · {quality.suitability_label}" if mode == "Real Estate Intelligence" else "Generic mode")
quality_cols[2].metric("Completeness", f"{quality.completeness}/100")
if expert:
    quality_cols[3].metric("Leakage Risk", quality.leakage_risk)

if expert:
    with st.expander("Available and unavailable valuation signals", expanded=False):
        available, unavailable = st.columns(2)
        available.markdown("**Available valuation signals**")
        available.write("\n".join(f"✓ {item.replace('_', ' ').title()}" for item in domain.available_signals) or "None detected")
        unavailable.markdown("**Unavailable signals**")
        unavailable.write("\n".join(f"○ {item.replace('_', ' ').title()}" for item in domain.unavailable_signals))
        st.caption("The model uses only available columns. Missing property concepts are never fabricated or filled with arbitrary defaults.")
if excluded and expert:
    st.warning(f"Auto-excluded likely IDs or near-constant columns: {', '.join(excluded)}")
if st.button("Confirm dataset schema", type="primary"):
    st.session_state.target = target
    st.session_state.features = features
    st.session_state.currency = currency
    st.session_state.quality_assessment = quality
    st.session_state.leakage_warnings = leakage
    st.session_state.cleaning_config = None
    st.session_state.training_result = None
    st.success("Dataset check complete. Continue to Data Preparation.")

if expert:
    tab_preview, tab_profile, tab_missing = st.tabs(["Data preview", "Column profile", "Missingness"])
    with tab_preview:
        st.dataframe(df.head(100), width="stretch")
    with tab_profile:
        st.dataframe(profile, width="stretch", hide_index=True)
    with tab_missing:
        if profile["missing"].sum():
            st.plotly_chart(missingness_chart(profile), width="stretch")
        else:
            st.success("No missing values detected.")
