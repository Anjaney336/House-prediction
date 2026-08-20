from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.utils.schema import infer_column_roles
from src.utils.schema import validate_columns
from src.validation.schema_contract import ModelSchemaContract


GROUP_ORDER = ["Location", "Property", "Size", "Rooms", "Building", "Amenities", "Accessibility", "Land", "Legal / Ownership", "Financial", "Neighborhood", "Transaction", "Additional Model Inputs"]


def contract_input_form(contract: ModelSchemaContract) -> tuple[pd.DataFrame | None, list[str]]:
    """Generate a grouped input workspace from the persisted model contract."""
    values: dict[str, object] = {}
    imputed: list[str] = []
    grouped: dict[str, list] = {}
    for spec in contract.features:
        grouped.setdefault(spec.group, []).append(spec)
    with st.form("contract_prediction_form"):
        for group in GROUP_ORDER:
            specs = grouped.get(group, [])
            if not specs:
                continue
            st.markdown(f"#### {group}")
            columns = st.columns(2)
            for index, spec in enumerate(specs):
                with columns[index % 2]:
                    help_text = f"Source column: {spec.name} · Missing values: {spec.imputation} imputation"
                    if spec.dtype == "numeric":
                        use_imputation = st.checkbox("Unknown — use trained imputation", key=f"missing_{spec.name}")
                        value = spec.median if spec.median is not None else 0.0
                        kwargs = {"value": float(value), "help": help_text, "key": f"value_{spec.name}"}
                        if spec.minimum is not None:
                            kwargs["min_value"] = float(spec.minimum)
                        if spec.maximum is not None:
                            kwargs["max_value"] = float(spec.maximum)
                        entered = st.number_input(spec.label, **kwargs)
                        values[spec.name] = np.nan if use_imputation else entered
                        if use_imputation:
                            imputed.append(spec.name)
                    elif spec.dtype == "boolean":
                        values[spec.name] = st.toggle(spec.label, help=help_text, key=f"value_{spec.name}")
                    elif len(spec.vocabulary) > 50:
                        entered = st.text_input(spec.label, value=spec.vocabulary[0] if spec.vocabulary else "", help=help_text, key=f"value_{spec.name}")
                        values[spec.name] = entered if entered.strip() else np.nan
                        if not entered.strip():
                            imputed.append(spec.name)
                    else:
                        unknown = "— Unknown / use trained imputation —"
                        selected = st.selectbox(spec.label, [unknown, *spec.vocabulary], index=1 if spec.vocabulary else 0, help=help_text, key=f"value_{spec.name}")
                        values[spec.name] = np.nan if selected == unknown else selected
                        if selected == unknown:
                            imputed.append(spec.name)
            st.divider()
        submitted = st.form_submit_button("Generate valuation", type="primary", width="stretch")
    return (pd.DataFrame([values], columns=contract.feature_order), imputed) if submitted else (None, [])


def dynamic_input_form(df: pd.DataFrame, features: list[str]) -> pd.DataFrame | None:
    """Render one input control per selected feature and return one row on submit."""
    projection = validate_columns(df, features)
    available = list(projection.available_columns)
    if not available:
        st.error("No selected prediction fields exist in the active dataset.")
        return None
    if projection.missing_columns:
        st.warning(f"Unavailable fields were excluded: {', '.join(projection.missing_columns)}")
    features = available
    roles = infer_column_roles(df.loc[:, features])
    values: dict[str, object] = {}
    precise = st.toggle("Use precise number entry", value=False)
    with st.form("prediction_form"):
        columns = st.columns(2)
        for index, feature in enumerate(features):
            container = columns[index % 2]
            series = df[feature]
            with container:
                if feature in roles.numeric:
                    clean = pd.to_numeric(series, errors="coerce").dropna()
                    low, high = float(clean.min()), float(clean.max())
                    median = float(clean.median())
                    if precise or low == high:
                        values[feature] = st.number_input(feature, min_value=low, max_value=high, value=median)
                    else:
                        step = max((high - low) / 200, 0.01)
                        values[feature] = st.slider(feature, low, high, median, step=step)
                elif feature in roles.boolean:
                    values[feature] = st.toggle(feature, value=bool(series.mode().iloc[0]) if not series.mode().empty else False)
                elif feature in roles.high_cardinality:
                    suggestions = series.dropna().astype(str).value_counts().head(8).index.tolist()
                    values[feature] = st.text_input(feature, value=suggestions[0] if suggestions else "", help=f"Examples: {', '.join(suggestions[:5])}")
                elif feature in roles.datetime:
                    parsed = pd.to_datetime(series, errors="coerce").dropna()
                    default = parsed.median().date() if len(parsed) else pd.Timestamp.today().date()
                    values[feature] = str(st.date_input(feature, value=default))
                else:
                    choices = series.dropna().astype(str).unique().tolist()
                    values[feature] = st.selectbox(feature, choices or ["Missing"])
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")
    return pd.DataFrame([values]) if submitted else None
