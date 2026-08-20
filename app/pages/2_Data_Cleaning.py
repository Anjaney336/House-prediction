from __future__ import annotations

import plotly.express as px
import pandas as pd
import streamlit as st

from app.components.sidebar import initialize_state, render_sidebar, require_data
from src.data.cleaner import CleaningConfig, cleaning_preview
from src.data.validator import detect_data_issues
from src.utils.schema import validate_columns


st.set_page_config(page_title="Data Cleaning · PricePredict AI", page_icon="🧹", layout="wide")
initialize_state()
render_sidebar(2)
expert = bool(st.session_state.get("expert_mode", False))
st.title("🧹 Data Preparation")
if not require_data():
    st.stop()

df = st.session_state.dataframe
target = st.session_state.get("target")
if not target or target not in df.columns:
    st.warning("Confirm the target and feature schema on the Data page before preparing this dataset.")
    st.stop()
requested_features = list(st.session_state.get("features") or [])
projection = validate_columns(df, requested_features)
features = list(projection.available_columns)
if projection.missing_columns:
    st.warning(f"The active dataset does not contain previously selected columns: {', '.join(projection.missing_columns)}. They were excluded safely; reconfirm the schema on the Data page.")
    st.session_state.features = features
if not features:
    st.error("No validated model features are available. Return to Data and confirm at least one feature.")
    st.stop()
working = df.loc[:, features]

st.subheader("Dataset Structure")
structure = st.columns(4)
structure[0].metric("Rows", f"{len(df):,}")
structure[1].metric("Columns", len(df.columns))
structure[2].metric("Detected Domain", f"{st.session_state.domain_analysis.domain.replace('_', ' ').title()} · {st.session_state.domain_analysis.confidence:.0%}" if st.session_state.get("domain_analysis") else "Pending")
structure[3].metric("Candidate Target", target)
property_types = list(getattr(st.session_state.get("domain_analysis"), "property_types", ()))
if property_types:
    st.caption("Observed property types: " + ", ".join(property_types))
available_col, excluded_col = st.columns(2)
with available_col:
    st.markdown("**Available model features**")
    st.write(", ".join(features))
with excluded_col:
    excluded = [column for column in df.columns if column not in features and column != target]
    st.markdown("**Excluded or unselected columns**")
    st.write(", ".join(excluded) or "None")

issue_columns = list(dict.fromkeys([*features, target]))
issues = detect_data_issues(df.loc[:, issue_columns])
if issues and expert:
    st.subheader("Data-quality findings")
    st.dataframe(pd.DataFrame([issue.to_dict() for issue in issues]), width="stretch", hide_index=True)
    st.caption("Invalid values are separated from valid-but-extreme observations. No records are removed automatically.")

if not expert:
    severe = [issue for issue in issues if issue.severity == "high"]
    status = "Needs Attention" if severe else "Warning" if issues else "Ready"
    cards = st.columns(3)
    cards[0].metric("Preparation Status", status)
    cards[1].metric("Issues Found", len(issues))
    cards[2].metric("High Priority", len(severe))
    for issue in severe[:5]:
        st.error(f"{issue.column}: {issue.message} {issue.recommendation}")
    st.info("Automatic data preprocessing will fill missing numeric values with medians, use the most frequent category for missing labels, and standardize numeric inputs inside each training fold.")
    recommended = CleaningConfig()
    if st.button("Use recommended preparation", type="primary", width="stretch"):
        st.session_state.cleaning_config = recommended
        st.session_state.training_result = None
        st.success("Data preparation saved. Continue to Model.")
    st.stop()

left, middle, right = st.columns(3)
numeric_default = left.selectbox("Default numeric imputation", ["median", "mean", "constant"])
categorical_default = middle.selectbox("Default categorical imputation", ["most_frequent", "constant"])
scaling = right.selectbox("Numeric scaling", ["standard", "minmax", "none"])
outlier_strategy = st.radio("Outlier handling", ["none", "cap_iqr"], horizontal=True, help="IQR limits are learned from training data only.")

numeric_by_column: dict[str, str] = {}
categorical_by_column: dict[str, str] = {}
with st.expander("Per-column imputation overrides"):
    for column in working.columns:
        if working[column].isna().sum() == 0:
            continue
        if working[column].dtype.kind in "biufc":
            numeric_by_column[column] = st.selectbox(column, ["median", "mean", "constant"], key=f"num_{column}")
        else:
            categorical_by_column[column] = st.selectbox(column, ["most_frequent", "constant"], key=f"cat_{column}")

config = CleaningConfig(
    numeric_imputation=numeric_default,
    categorical_imputation=categorical_default,
    scaling=scaling,
    outlier_strategy=outlier_strategy,
    numeric_by_column=numeric_by_column,
    categorical_by_column=categorical_by_column,
)

treatments = []
for column in working.columns:
    numeric = pd.api.types.is_numeric_dtype(working[column])
    strategy = numeric_by_column.get(column, numeric_default) if numeric else categorical_by_column.get(column, categorical_default)
    treatments.append({"Column": column, "Missing %": round(working[column].isna().mean() * 100, 2), "Treatment": f"{strategy.replace('_', ' ').title()} imputation", "Outliers": "IQR cap fitted on training data" if numeric and outlier_strategy == "cap_iqr" else "Retained"})
st.subheader("Cleaning decision register")
st.dataframe(pd.DataFrame(treatments), width="stretch", hide_index=True)
if st.button("Save cleaning configuration", type="primary"):
    st.session_state.cleaning_config = config
    st.session_state.training_result = None
    st.success("Cleaning configuration saved. It will be fitted only on training folds.")

before, after = st.tabs(["Before", "After preview"])
with before:
    st.dataframe(working.head(20), width="stretch")
with after:
    st.dataframe(cleaning_preview(working, config), width="stretch")

numeric_columns = working.select_dtypes("number").columns.tolist()
if numeric_columns:
    selected = st.selectbox("Inspect numeric outliers", numeric_columns)
    st.plotly_chart(px.box(working, x=selected, points="outliers"), width="stretch")
