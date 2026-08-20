from __future__ import annotations

import streamlit as st

from app.components.sidebar import initialize_state, render_sidebar


st.set_page_config(page_title="Model Science · PricePredict AI", page_icon="🔬", layout="wide")
initialize_state(); render_sidebar(8)
st.title("🔬 Model Science")
if not st.session_state.get("expert_mode", False):
    st.info("Enable Expert Mode to inspect the methodology behind model selection.")
    st.stop()

sections = {
    "Why multiple models are tested": "Real-estate tables combine nonlinear size effects, location categories, interactions, missing values, and market noise. No single algorithm is universally strongest, so candidates are compared under identical validation folds.",
    "Why linear baselines matter": "Linear and regularized models are fast, interpretable reference points. If a complex model cannot improve on them consistently, its added operational cost is difficult to justify.",
    "Why boosting often performs strongly": "Boosted trees can recover nonlinear thresholds and interactions common in tabular valuation data. This is an empirical tendency, not a guaranteed winner.",
    "Why categorical handling matters": "Locality, property type, furnishing, and zoning carry market structure. Encoders are fitted inside each training fold so validation data cannot influence learned categories or imputation.",
    "Why validation strategy matters": "Random folds estimate performance on similar observations. Chronological, property-grouped, and geographic folds answer harder questions about future markets, repeated listings, and unseen locations.",
    "Why ensembles are not always accepted": "Voting, blending, and stacking are retained only when their cross-validated RMSE improves beyond a configured threshold. Otherwise the simpler single model remains preferred.",
    "Why synthetic benchmarks are used": "Synthetic markets provide known drivers and controlled corruption, shift, scale, and noise. They validate software and methodology but cannot establish real-world appraisal accuracy.",
}
for title, body in sections.items():
    with st.expander(title, expanded=True):
        st.write(body)
st.warning("All displayed explanations are predictive associations. Neither feature importance nor SHAP establishes causality.")
