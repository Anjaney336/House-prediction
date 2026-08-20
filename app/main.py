from __future__ import annotations

import streamlit as st

from app.components.sidebar import initialize_state, render_sidebar


st.set_page_config(page_title="PricePredict AI", page_icon="🏠", layout="wide")
initialize_state()
render_sidebar(0)

st.markdown(
    """
    <style>
      .stApp {background: linear-gradient(145deg, #07111f 0%, #0c1b2a 55%, #102536 100%);}
      [data-testid="stMetric"] {background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.09); padding: 1rem; border-radius: 14px;}
      .hero {padding: 3rem 2.5rem; border-radius: 24px; background: linear-gradient(120deg, rgba(22,166,161,.22), rgba(124,92,252,.18)); border: 1px solid rgba(255,255,255,.12); margin-bottom: 2rem;}
      .hero h1 {font-size: 3.3rem; margin: 0; letter-spacing: -.04em;}
      .hero p {font-size: 1.15rem; color: #b7c8d9; max-width: 720px;}
    </style>
    <section class="hero">
      <div style="color:#4fe0d6;font-weight:700;letter-spacing:.12em">REAL ESTATE VALUATION & PREDICTIVE INTELLIGENCE</div>
      <h1>AI-Powered Property Valuation</h1>
      <p>Upload property data, train a validated valuation model, understand its drivers, and estimate values for one property or an entire portfolio.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

cta, demo, previous, spacer = st.columns([1, 1, 1.3, 2.7])
with cta:
    st.page_link("pages/1_Upload_Data.py", label="Upload Dataset", icon="📤")
with demo:
    st.page_link("pages/1_Upload_Data.py", label="Use Demo", icon="🏙️")
with previous:
    st.page_link("pages/6_Model_Registry.py", label="View Previous Model", icon="📄")

if st.session_state.dataframe is not None:
    st.subheader("Workspace status")
    domain = st.session_state.domain_analysis
    quality = st.session_state.quality_assessment
    result = st.session_state.training_result
    cards = st.columns(6)
    cards[0].metric("Active dataset", st.session_state.dataset_name or "Uploaded CSV")
    cards[1].metric("Rows", f"{len(st.session_state.dataframe):,}")
    cards[2].metric("Domain", "Real Estate" if domain and domain.is_real_estate else "Generic")
    cards[3].metric("Data quality", f"{quality.overall}/100" if quality else "Pending")
    cards[4].metric("Best model", result.active_model_name if result else "Not trained")
    cards[5].metric("Model status", "Ready" if result else "Setup")

steps = st.columns(4)
for box, title, description in zip(
    steps,
    ["1 · Check Data", "2 · Build Model", "3 · Value Property", "4 · Explain Result"],
    ["Confirm that the dataset is suitable.", "Choose Quick, Balanced, or Advanced.", "Enter one property or upload a portfolio.", "See drivers, range, confidence, and comparables."],
):
    with box:
        st.subheader(title)
        st.write(description)

st.info("The dataset defines the model schema, and the saved model schema defines every valuation input. The California dataset is a block-level demo—not the product schema.")
st.caption("Predictions are historical, data-dependent estimates. They are not guaranteed market prices, legal valuations, or substitutes for licensed appraisal where required.")
