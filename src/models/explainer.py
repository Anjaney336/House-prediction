from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def feature_importance(pipeline, X: pd.DataFrame, y: pd.Series, max_features: int = 25) -> pd.DataFrame:
    """Return model-native importance when available, otherwise permutation importance."""
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed_names = preprocess.get_feature_names_out()
    base_model = getattr(model, "regressor_", model)
    values = getattr(base_model, "feature_importances_", None)
    if values is None:
        result = permutation_importance(pipeline, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        return pd.DataFrame({"feature": X.columns, "importance": result.importances_mean}).sort_values(
            "importance", ascending=False
        ).head(max_features)
    frame = pd.DataFrame({"feature": transformed_names, "importance": np.asarray(values)})
    return frame.sort_values("importance", ascending=False).head(max_features)


def local_contributions(pipeline, row: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """Estimate local feature contributions with SHAP when the fitted model supports it."""
    import shap

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    base_model = getattr(model, "regressor_", model)
    names = preprocess.get_feature_names_out()
    background_matrix = preprocess.transform(background.head(100))
    row_matrix = preprocess.transform(row)
    explainer = shap.Explainer(base_model, background_matrix, feature_names=names)
    explanation = explainer(row_matrix)
    values = np.asarray(explanation.values)[0]
    return pd.DataFrame({"feature": names, "contribution": values}).sort_values(
        "contribution", key=lambda s: s.abs(), ascending=False
    )


def shap_summary_data(pipeline, X: pd.DataFrame, max_rows: int = 150) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return global mean-absolute SHAP importance and a beeswarm-ready long table."""
    import shap

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    base_model = getattr(model, "regressor_", model)
    sample = X.sample(min(max_rows, len(X)), random_state=42)
    background = X.sample(min(100, len(X)), random_state=7)
    sample_matrix = np.asarray(preprocess.transform(sample))
    background_matrix = np.asarray(preprocess.transform(background))
    names = np.asarray(preprocess.get_feature_names_out())
    explanation = shap.Explainer(base_model, background_matrix, feature_names=names)(sample_matrix)
    values = np.asarray(explanation.values)
    global_frame = pd.DataFrame(
        {"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    top_names = global_frame.head(15)["feature"].tolist()
    indices = [int(np.where(names == name)[0][0]) for name in top_names]
    long_frame = pd.DataFrame(
        {
            "feature": np.repeat(names[indices], len(sample_matrix)),
            "shap_value": values[:, indices].T.reshape(-1),
            "feature_value": sample_matrix[:, indices].T.reshape(-1),
        }
    )
    return global_frame.head(25), long_frame
