from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Callable

import pandas as pd
from src.models.native_safety import probe_dependency
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    PassiveAggressiveRegressor,
    QuantileRegressor,
    Ridge,
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from src.utils.config import RANDOM_STATE


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    factory: Callable[[], object]
    modes: tuple[str, ...]
    search_space: dict[str, list]
    cost: int = 2
    min_rows: int = 20
    max_rows: int | None = None
    max_features: int | None = None
    dependency: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class Eligibility:
    name: str
    family: str
    status: str
    reason: str
    cost: str


def _xgboost():
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=180, learning_rate=0.05, max_depth=5, subsample=0.85,
        colsample_bytree=0.85, objective="reg:squarederror",
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def _lightgbm():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=180, learning_rate=0.05, num_leaves=31,
        # Some Windows LightGBM builds raise native access violations under
        # unrestricted OpenMP parallelism. Keep this optional model isolated
        # and deterministic; platform stability is more important than speed.
        random_state=RANDOM_STATE, n_jobs=1, verbosity=-1,
    )


def _catboost():
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=180, learning_rate=0.05, depth=6,
        random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
    )


def model_catalog() -> dict[str, ModelSpec]:
    """Return the single source of truth for supported regression models."""
    fast = ("Core", "Fast", "Balanced", "Comprehensive")
    balanced = ("Balanced", "Comprehensive")
    comprehensive = ("Comprehensive",)
    specs = [
        ModelSpec("Linear Regression", "Linear", LinearRegression, fast, {} , 1),
        ModelSpec("Ridge", "Linear", lambda: Ridge(alpha=1.0), fast, {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}, 1),
        ModelSpec("Lasso", "Linear", lambda: Lasso(alpha=1.0, max_iter=30000, random_state=RANDOM_STATE), fast, {"alpha": [0.01, 0.1, 1.0, 10.0]}, 1),
        ModelSpec("Elastic Net", "Linear", lambda: ElasticNet(max_iter=30000, random_state=RANDOM_STATE), balanced, {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]}, 1),
        ModelSpec("Bayesian Ridge", "Linear", BayesianRidge, balanced, {"alpha_1": [1e-7, 1e-6, 1e-5]}, 1),
        ModelSpec("Huber", "Robust linear", lambda: HuberRegressor(max_iter=1000), balanced, {"epsilon": [1.1, 1.35, 1.75], "alpha": [0.0001, 0.01]}, 2),
        ModelSpec("SGD", "Online linear", lambda: SGDRegressor(max_iter=3000, random_state=RANDOM_STATE), balanced, {"alpha": [0.00001, 0.0001, 0.001], "penalty": ["l2", "l1", "elasticnet"]}, 1),
        ModelSpec("Passive Aggressive", "Online linear", lambda: PassiveAggressiveRegressor(max_iter=3000, random_state=RANDOM_STATE), comprehensive, {"C": [0.01, 0.1, 1.0]}, 1),
        ModelSpec("Quantile Regression", "Robust linear", lambda: QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"), comprehensive, {"alpha": [0.0, 0.01, 0.1]}, 4, max_rows=5000),
        ModelSpec(
            "Polynomial Ridge", "Polynomial",
            lambda: Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("regressor", Ridge())]),
            comprehensive, {"poly__degree": [2, 3], "regressor__alpha": [0.1, 1.0, 10.0]},
            4, max_rows=10000, max_features=15,
        ),
        ModelSpec(
            "Polynomial Lasso", "Polynomial",
            lambda: Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("regressor", Lasso(max_iter=30000, random_state=RANDOM_STATE))]),
            comprehensive, {"poly__degree": [2, 3], "regressor__alpha": [0.01, 0.1, 1.0]},
            4, max_rows=10000, max_features=15,
        ),
        ModelSpec("K-Nearest Neighbors", "Instance-based", lambda: KNeighborsRegressor(n_neighbors=7, weights="distance"), balanced, {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"], "p": [1, 2]}, 2, max_rows=50000),
        ModelSpec("Linear SVR", "Kernel", lambda: SVR(kernel="linear", C=1.0), comprehensive, {"C": [0.1, 1.0, 10.0], "epsilon": [0.01, 0.1, 0.2]}, 3, max_rows=15000),
        ModelSpec("RBF SVR", "Kernel", lambda: SVR(kernel="rbf", C=10.0), comprehensive, {"C": [1.0, 10.0, 100.0], "gamma": ["scale", "auto"], "epsilon": [0.05, 0.1, 0.2]}, 4, max_rows=12000),
        ModelSpec("Polynomial SVR", "Kernel", lambda: SVR(kernel="poly", degree=2), comprehensive, {"C": [0.1, 1.0, 10.0], "degree": [2, 3]}, 5, max_rows=8000, max_features=30),
        ModelSpec("Decision Tree", "Decision tree", lambda: DecisionTreeRegressor(min_samples_leaf=2, random_state=RANDOM_STATE), balanced, {"max_depth": [None, 4, 8, 16], "min_samples_leaf": [1, 2, 5, 10]}, 1),
        ModelSpec("Extra Tree", "Decision tree", lambda: ExtraTreeRegressor(min_samples_leaf=2, random_state=RANDOM_STATE), balanced, {"max_depth": [None, 4, 8, 16], "min_samples_leaf": [1, 2, 5]}, 1),
        ModelSpec("Random Forest", "Bagging ensemble", lambda: RandomForestRegressor(n_estimators=140, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1), fast, {"n_estimators": [100, 180, 260], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.8, 1.0]}, 3),
        ModelSpec("Extra Trees", "Bagging ensemble", lambda: ExtraTreesRegressor(n_estimators=140, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1), balanced, {"n_estimators": [100, 180, 260], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 2, 4]}, 3),
        ModelSpec("Gradient Boosting", "Boosting", lambda: GradientBoostingRegressor(random_state=RANDOM_STATE), fast, {"n_estimators": [80, 140, 220], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [2, 3, 4]}, 3),
        ModelSpec("Histogram Gradient Boosting", "Boosting", lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE), balanced, {"learning_rate": [0.03, 0.07, 0.1], "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0.0, 0.1, 1.0]}, 3),
        ModelSpec("XGBoost", "External boosting", _xgboost, balanced, {"n_estimators": [100, 180, 260], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [3, 5, 7], "subsample": [0.7, 0.85, 1.0]}, 4, dependency="xgboost"),
        ModelSpec("LightGBM", "External boosting", _lightgbm, balanced, {"n_estimators": [100, 180, 260], "learning_rate": [0.03, 0.05, 0.1], "num_leaves": [15, 31, 63]}, 4, dependency="lightgbm"),
        ModelSpec("CatBoost", "External boosting", _catboost, balanced, {"iterations": [100, 180, 260], "learning_rate": [0.03, 0.05, 0.1], "depth": [4, 6, 8]}, 4, dependency="catboost"),
        ModelSpec("MLP Neural Network", "Neural network", lambda: MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=600, random_state=RANDOM_STATE), comprehensive, {"hidden_layer_sizes": [(32,), (64, 32), (128, 64)], "alpha": [0.0001, 0.001, 0.01]}, 4, min_rows=100),
        ModelSpec("Gaussian Process", "Probabilistic", lambda: GaussianProcessRegressor(kernel=Matern() + WhiteKernel(), normalize_y=True, random_state=RANDOM_STATE), comprehensive, {"alpha": [1e-10, 1e-6, 1e-3]}, 5, max_rows=1500, max_features=25),
    ]
    return {spec.name: spec for spec in specs}


def assess_eligibility(
    X: pd.DataFrame,
    mode: str = "Balanced",
    selected_models: list[str] | None = None,
    include_optional: bool = True,
) -> tuple[list[ModelSpec], pd.DataFrame]:
    catalog = model_catalog()
    requested = set(selected_models or catalog)
    eligible: list[ModelSpec] = []
    rows: list[dict[str, str]] = []
    cost_labels = {1: "Very low", 2: "Low", 3: "Medium", 4: "High", 5: "Very high"}
    for spec in catalog.values():
        reason = "Eligible for this dataset and run mode."
        status = "Eligible"
        if spec.name not in requested:
            status, reason = "Not selected", "Not selected in Expert Mode."
        elif mode not in spec.modes:
            status, reason = "Excluded", f"Not part of {mode} mode."
        elif len(X) < spec.min_rows:
            status, reason = "Excluded", f"Needs at least {spec.min_rows:,} rows."
        elif spec.max_rows is not None and len(X) > spec.max_rows:
            status, reason = "Excluded", f"Limited to {spec.max_rows:,} rows to control runtime."
        elif spec.max_features is not None and X.shape[1] > spec.max_features:
            status, reason = "Excluded", f"Limited to {spec.max_features} input features to control dimensionality."
        elif spec.dependency and not include_optional:
            status, reason = "Disabled", "Optional external boosters are disabled."
        elif spec.dependency and find_spec(spec.dependency) is None:
            status, reason = "Unavailable", f"Optional package '{spec.dependency}' is not installed."
        elif spec.dependency:
            probe = probe_dependency(spec.dependency)
            if not probe.safe:
                status, reason = "Unavailable", probe.detail
        if status == "Eligible":
            eligible.append(spec)
        rows.append({"Model": spec.name, "Family": spec.family, "Status": status, "Reason": reason, "Compute cost": cost_labels[spec.cost]})
    return eligible, pd.DataFrame(rows)
