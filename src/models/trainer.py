from __future__ import annotations

import platform
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.data.cleaner import CleaningConfig, build_preprocessor
from src.domain.property_ontology import map_column
from src.models.model_catalog import ModelSpec, assess_eligibility
from src.models.validation_strategy import ValidationPlan, recommend_validation
from src.utils.config import RANDOM_STATE


@dataclass
class TrainingResult:
    leaderboard: pd.DataFrame
    pipelines: dict[str, Pipeline]
    active_model_name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_calibration: pd.DataFrame
    y_calibration: pd.Series
    test_predictions: dict[str, np.ndarray]
    residual_std: dict[str, float]
    conformal_radius: dict[str, float]
    conformal_coverage: dict[str, float]
    metadata: dict = field(default_factory=dict)
    availability: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_scores: dict[str, list[float]] = field(default_factory=dict)
    tuning_results: dict[str, dict] = field(default_factory=dict)

    @property
    def active_pipeline(self) -> Pipeline:
        return self.pipelines[self.active_model_name]


def _wrap_target(model: object, log_target: bool) -> object:
    if not log_target:
        return model
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1, check_inverse=False)


def _metrics(y_true, prediction) -> dict[str, float]:
    truth, predicted = np.asarray(y_true, dtype=float), np.asarray(prediction, dtype=float)
    safe = np.abs(truth) > max(1e-9, np.nanmedian(np.abs(truth)) * 1e-8)
    mape = float(np.mean(np.abs((truth[safe] - predicted[safe]) / truth[safe])) * 100) if safe.any() else np.nan
    return {
        "R²": r2_score(truth, predicted),
        "RMSE": mean_squared_error(truth, predicted) ** 0.5,
        "MAE": mean_absolute_error(truth, predicted),
        "Median AE": median_absolute_error(truth, predicted),
        "MAPE (%)": mape,
        "Explained Variance": explained_variance_score(truth, predicted),
    }


def conformal_radius(y_true, prediction, confidence: float = 0.95) -> float:
    """Finite-sample split-conformal absolute-residual quantile."""
    errors = np.abs(np.asarray(y_true, dtype=float) - np.asarray(prediction, dtype=float))
    if len(errors) == 0 or not 0 < confidence < 1:
        raise ValueError("Conformal calibration requires residuals and confidence strictly between 0 and 1.")
    quantile = min(1.0, np.ceil((len(errors) + 1) * confidence) / len(errors))
    return float(np.quantile(errors, quantile, method="higher"))


def _split_development(
    X: pd.DataFrame,
    y: pd.Series,
    plan: ValidationPlan,
    test_size: float,
    calibration_size: float,
):
    if not 0.05 <= calibration_size <= 0.4:
        raise ValueError("Calibration size must be between 5% and 40% of development data.")
    if plan.order is not None:
        X, y = X.loc[plan.order], y.loc[plan.order]
        test_at = max(1, int(len(X) * (1 - test_size)))
        X_development, X_test = X.iloc[:test_at], X.iloc[test_at:]
        y_development, y_test = y.iloc[:test_at], y.iloc[test_at:]
        calibration_at = max(1, int(len(X_development) * (1 - calibration_size)))
        return (
            X_development.iloc[:calibration_at], X_development.iloc[calibration_at:], X_test,
            y_development.iloc[:calibration_at], y_development.iloc[calibration_at:], y_test,
        )
    if plan.groups is not None:
        groups = plan.groups.reindex(X.index)
        outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
        development_positions, test_positions = next(outer.split(X, y, groups=groups))
        X_development, X_test = X.iloc[development_positions], X.iloc[test_positions]
        y_development, y_test = y.iloc[development_positions], y.iloc[test_positions]
        development_groups = groups.iloc[development_positions]
        inner = GroupShuffleSplit(n_splits=1, test_size=calibration_size, random_state=RANDOM_STATE + 1)
        train_positions, calibration_positions = next(inner.split(X_development, y_development, groups=development_groups))
        return (
            X_development.iloc[train_positions], X_development.iloc[calibration_positions], X_test,
            y_development.iloc[train_positions], y_development.iloc[calibration_positions], y_test,
        )
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    X_train, X_calibration, y_train, y_calibration = train_test_split(
        X_development, y_development, test_size=calibration_size, random_state=RANDOM_STATE + 1
    )
    return X_train, X_calibration, X_test, y_train, y_calibration, y_test


def _cv_kwargs(plan: ValidationPlan, index: pd.Index) -> dict:
    return {} if plan.groups is None else {"groups": plan.groups.reindex(index)}


def _parameter_space(spec: ModelSpec, pipeline: Pipeline) -> dict[str, list]:
    wrapped = isinstance(pipeline.named_steps["model"], TransformedTargetRegressor)
    prefix = "model__regressor__" if wrapped else "model__"
    return {f"{prefix}{parameter}": values for parameter, values in spec.search_space.items()}


def _evaluate(name, family, pipeline, X_train, y_train, X_test, y_test, plan, cost):
    started = time.perf_counter()
    scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "medae": "neg_median_absolute_error", "ev": "explained_variance"}
    scores = cross_validate(
        pipeline, X_train, y_train, scoring=scoring, cv=plan.splitter,
        n_jobs=1, error_score="raise", **_cv_kwargs(plan, X_train.index),
    )
    fitted = clone(pipeline).fit(X_train, y_train)
    train_prediction, test_prediction = fitted.predict(X_train), np.asarray(fitted.predict(X_test))
    prediction_started = time.perf_counter()
    fitted.predict(X_test.head(min(100, len(X_test))))
    prediction_ms = (time.perf_counter() - prediction_started) * 1000
    train_metrics, test_metrics = _metrics(y_train, train_prediction), _metrics(y_test, test_prediction)
    fold_rmse = (-scores["test_rmse"]).astype(float).tolist()
    row = {
        "Model": name, "Family": family,
        "CV R²": float(np.mean(scores["test_r2"])), "CV R² Std": float(np.std(scores["test_r2"])),
        "CV RMSE": float(np.mean(fold_rmse)), "CV RMSE Std": float(np.std(fold_rmse)),
        "CV MAE": float(-np.mean(scores["test_mae"])), "CV Median AE": float(-np.mean(scores["test_medae"])),
        "CV Explained Variance": float(np.mean(scores["test_ev"])),
        "Train R²": train_metrics["R²"], "Test R²": test_metrics["R²"],
        "Test RMSE": test_metrics["RMSE"], "Test MAE": test_metrics["MAE"],
        "Test Median AE": test_metrics["Median AE"], "Test MAPE (%)": test_metrics["MAPE (%)"],
        "Overfit Gap": max(0.0, train_metrics["R²"] - test_metrics["R²"]), "Complexity": cost,
        "Training Time (s)": time.perf_counter() - started, "Prediction Time (ms/100 rows)": prediction_ms,
    }
    return row, fitted, test_prediction, fold_rmse


def _scale_benefit(values: pd.Series, lower_is_better: bool = True) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan)
    clean = clean.fillna(clean.max() if lower_is_better else clean.min())
    spread = clean.max() - clean.min()
    if spread <= 1e-12:
        return pd.Series(1.0, index=values.index)
    normalized = (clean - clean.min()) / spread
    return 1.0 - normalized if lower_is_better else normalized


def _rank(rows: list[dict], weights: dict[str, float] | None = None) -> pd.DataFrame:
    board = pd.DataFrame(rows)
    weights = weights or {"performance": 0.60, "stability": 0.20, "simplicity": 0.10, "generalization": 0.10}
    required = {"performance", "stability", "simplicity", "generalization"}
    if set(weights) != required or any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError(f"Ranking weights must contain non-negative values for: {', '.join(sorted(required))}.")
    total_weight = sum(weights.values())
    weights = {key: value / total_weight for key, value in weights.items()}
    performance = _scale_benefit(board["CV RMSE"])
    stability = _scale_benefit(board["CV RMSE Std"])
    complexity = _scale_benefit(board["Complexity"])
    generalization = _scale_benefit(board["Overfit Gap"])
    board["Performance Score"] = performance * 100
    board["Stability Score"] = stability * 100
    board["Overall Score"] = (
        performance * weights["performance"] + stability * weights["stability"]
        + complexity * weights["simplicity"] + generalization * weights["generalization"]
    ) * 100
    board = board.sort_values(["Overall Score", "CV RMSE", "CV MAE"], ascending=[False, True, True]).reset_index(drop=True)
    board.insert(0, "Rank", np.arange(1, len(board) + 1))
    board["Badge"] = ""
    if len(board):
        board.loc[0, "Badge"] = "Best overall"
        predictive = board["CV RMSE"].idxmin()
        stable = board["CV RMSE Std"].idxmin()
        board.loc[predictive, "Badge"] = "Best predictive" if predictive else "Best overall · Best predictive"
        board.loc[stable, "Badge"] = (board.loc[stable, "Badge"] + " · Most stable").strip(" ·")
    return board


def _choose_log_target(y: pd.Series, requested: bool | None) -> tuple[bool, str]:
    if requested is not None:
        return requested, "log1p (user selected)" if requested else "None (user selected)"
    skew = float(y.skew())
    use_log = bool((y >= 0).all() and skew > 1.0)
    return use_log, f"{'log1p' if use_log else 'None'} (automatic; target skew={skew:.2f})"


def train_regressors(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    cleaning_config: CleaningConfig | None = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    log_target: bool | None = False,
    tune: bool = True,
    tuning_iterations: int = 5,
    include_xgboost: bool = False,
    progress: Callable[[float, str], None] | None = None,
    metadata: dict | None = None,
    mode: str = "Core",
    selected_models: list[str] | None = None,
    tune_top_n: int = 3,
    include_optional_boosters: bool | None = None,
    include_ensembles: bool = False,
    validation_strategy: str = "auto",
    validation_group_column: str | None = None,
    ranking_weights: dict[str, float] | None = None,
    minimum_ensemble_improvement: float = 0.005,
    calibration_size: float = 0.20,
    interval_confidence: float = 0.95,
) -> TrainingResult:
    """Screen, optimize, compare, and rank leakage-safe regression pipelines."""
    if target in features:
        raise ValueError("The target cannot also be used as a feature.")
    missing = [column for column in [*features, target] if column not in df.columns]
    if missing:
        raise ValueError(f"Training schema contains unavailable columns: {', '.join(dict.fromkeys(missing))}.")
    working = df.loc[:, list(dict.fromkeys(features + [target]))].dropna(subset=[target]).copy()
    # CSV readers expose dates as strings. Promote only ontology-recognized date
    # fields with a strong parse rate so they receive temporal preprocessing
    # instead of a high-cardinality categorical encoding.
    for column in features:
        role = map_column(column)
        if role and role.role in {"listing_date", "transaction_date"}:
            parsed = pd.to_datetime(working[column], errors="coerce")
            non_null = int(working[column].notna().sum())
            if non_null and int(parsed.notna().sum()) / non_null >= 0.8:
                working[column] = parsed
    if len(working) < max(20, cv_folds * 3):
        raise ValueError(f"At least {max(20, cv_folds * 3)} target-complete rows are required.")
    X, y = working[features], pd.to_numeric(working[target], errors="raise")
    use_log, transform_label = _choose_log_target(y, log_target)
    if use_log and (y < 0).any():
        raise ValueError("log1p target transformation requires non-negative target values.")

    plan = recommend_validation(
        df.loc[working.index], features, cv_folds,
        strategy=validation_strategy, group_column=validation_group_column,
    )
    X_train, X_calibration, X_test, y_train, y_calibration, y_test = _split_development(
        X, y, plan, test_size, calibration_size
    )
    effective_cv_folds = cv_folds
    if plan.groups is not None:
        training_groups = plan.groups.reindex(X_train.index).nunique(dropna=True)
        effective_cv_folds = min(cv_folds, int(training_groups))
        if effective_cv_folds < 2:
            raise ValueError("Grouped validation needs at least two groups after reserving calibration and test groups.")
        if effective_cv_folds != cv_folds:
            plan = replace(
                plan, splitter=GroupKFold(n_splits=effective_cv_folds),
                reason=f"{plan.reason} CV folds were reduced from {cv_folds} to {effective_cv_folds} because only {training_groups} training groups remain after calibration and holdout reservation.",
            )
    if len(X_train) < cv_folds * 3 or len(X_calibration) < 3 or len(X_test) < 3:
        raise ValueError("The train/calibration/test split is too small. Add rows or reduce validation folds.")
    include_optional = include_xgboost if include_optional_boosters is None else include_optional_boosters
    specs, availability = assess_eligibility(X_train, mode, selected_models, include_optional)
    if not specs:
        raise ValueError("No models are eligible. Change the mode or Expert Mode selection.")

    preprocessor = build_preprocessor(X_train, cleaning_config)
    rows, pipelines, predictions, residual_std, radii, coverages, fold_scores, tuning_results = [], {}, {}, {}, {}, {}, {}, {}
    spec_by_name = {spec.name: spec for spec in specs}
    for index, spec in enumerate(specs, start=1):
        try:
            pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", _wrap_target(spec.factory(), use_log))])
            row, fitted, prediction, folds = _evaluate(spec.name, spec.family, pipeline, X_train, y_train, X_test, y_test, plan, spec.cost)
        except Exception as exc:
            availability.loc[availability["Model"] == spec.name, ["Status", "Reason"]] = ["Failed", f"{type(exc).__name__}: {exc}"]
            continue
        rows.append(row)
        pipelines[spec.name], predictions[spec.name], fold_scores[spec.name] = fitted, prediction, folds
        calibration_prediction = fitted.predict(X_calibration)
        residual_std[spec.name] = float(np.std(y_calibration.to_numpy() - calibration_prediction, ddof=1))
        radii[spec.name] = conformal_radius(y_calibration, calibration_prediction, interval_confidence)
        coverages[spec.name] = float(np.mean(np.abs(y_test.to_numpy() - prediction) <= radii[spec.name]))
        row[f"{interval_confidence:.0%} Interval Coverage"] = coverages[spec.name]
        row[f"{interval_confidence:.0%} Interval Width"] = 2 * radii[spec.name]
        if progress:
            progress(index / (len(specs) + max(1, tune_top_n)), f"Screened {spec.name}")
    if not rows:
        failures = availability.loc[availability["Status"] == "Failed", "Reason"].tolist()
        raise RuntimeError("Every eligible model failed. " + (failures[0] if failures else "Review the dataset schema."))

    screening = _rank(rows, ranking_weights)
    if tune:
        tunable = [name for name in screening["Model"].tolist() if spec_by_name[name].search_space][:max(0, tune_top_n)]
        for tune_index, name in enumerate(tunable, start=1):
            spec = spec_by_name[name]
            started = time.perf_counter()
            try:
                base = Pipeline([("preprocess", clone(preprocessor)), ("model", _wrap_target(spec.factory(), use_log))])
                search = RandomizedSearchCV(
                    base, _parameter_space(spec, base), n_iter=max(1, tuning_iterations),
                    scoring="neg_root_mean_squared_error", cv=plan.splitter, random_state=RANDOM_STATE,
                    n_jobs=-1, refit=True, error_score="raise",
                )
                search.fit(X_train, y_train, **_cv_kwargs(plan, X_train.index))
                row, fitted, prediction, folds = _evaluate(name, spec.family, search.best_estimator_, X_train, y_train, X_test, y_test, plan, spec.cost)
                rows = [existing for existing in rows if existing["Model"] != name] + [row]
                pipelines[name], predictions[name], fold_scores[name] = fitted, prediction, folds
                calibration_prediction = fitted.predict(X_calibration)
                residual_std[name] = float(np.std(y_calibration.to_numpy() - calibration_prediction, ddof=1))
                radii[name] = conformal_radius(y_calibration, calibration_prediction, interval_confidence)
                coverages[name] = float(np.mean(np.abs(y_test.to_numpy() - prediction) <= radii[name]))
                row[f"{interval_confidence:.0%} Interval Coverage"] = coverages[name]
                row[f"{interval_confidence:.0%} Interval Width"] = 2 * radii[name]
                tuning_results[name] = {"Best parameters": search.best_params_, "Best search CV RMSE": float(-search.best_score_), "Search time (s)": time.perf_counter() - started}
            except Exception as exc:
                tuning_results[name] = {"Status": "Optimization failed; screened model retained", "Reason": str(exc)}
            if progress:
                progress((len(specs) + tune_index) / (len(specs) + max(1, len(tunable))), f"Optimized {name}")

    leaderboard = _rank(rows, ranking_weights)
    ensemble_decisions: dict[str, dict] = {}
    if include_ensembles and len(leaderboard) >= 2:
        best_single_rmse = float(leaderboard["CV RMSE"].min())
        complementary, families = [], set()
        for _, candidate in leaderboard.iterrows():
            if candidate["Family"] not in families or len(complementary) < 2:
                complementary.append(str(candidate["Model"])); families.add(str(candidate["Family"]))
            if len(complementary) == 3:
                break
        estimators = [(f"m{i}", clone(pipelines[name].named_steps["model"])) for i, name in enumerate(complementary)]
        ensemble_defs = {
            "Voting Ensemble": VotingRegressor(estimators=estimators),
            "Weighted Blend": VotingRegressor(
                estimators=estimators,
                weights=[1.0 / max(float(leaderboard.loc[leaderboard["Model"] == model_name, "CV RMSE"].iloc[0]), 1e-9) for model_name in complementary],
            ),
            "Stacking Ensemble": StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=3, n_jobs=-1),
        }
        for name, estimator in ensemble_defs.items():
            pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
            try:
                row, fitted, prediction, folds = _evaluate(name, "Ensemble", pipeline, X_train, y_train, X_test, y_test, plan, 5)
                improvement = (best_single_rmse - float(row["CV RMSE"])) / max(best_single_rmse, 1e-9)
                accepted = improvement >= minimum_ensemble_improvement
                ensemble_decisions[name] = {
                    "accepted": accepted, "improvement_percent": improvement * 100,
                    "minimum_required_percent": minimum_ensemble_improvement * 100,
                    "components": complementary,
                }
                if accepted:
                    rows.append(row)
                    pipelines[name], predictions[name], fold_scores[name] = fitted, prediction, folds
                    calibration_prediction = fitted.predict(X_calibration)
                    residual_std[name] = float(np.std(y_calibration.to_numpy() - calibration_prediction, ddof=1))
                    radii[name] = conformal_radius(y_calibration, calibration_prediction, interval_confidence)
                    coverages[name] = float(np.mean(np.abs(y_test.to_numpy() - prediction) <= radii[name]))
                    row[f"{interval_confidence:.0%} Interval Coverage"] = coverages[name]
                    row[f"{interval_confidence:.0%} Interval Width"] = 2 * radii[name]
                    reason = f"Accepted: validation RMSE improved {improvement:.2%}; built from {', '.join(complementary)}."
                    status = "Eligible"
                else:
                    reason = f"Rejected: validation improvement was {improvement:.2%}, below the {minimum_ensemble_improvement:.2%} threshold."
                    status = "Rejected"
                availability.loc[len(availability)] = [name, "Ensemble", status, reason, "Very high"]
            except Exception as exc:
                availability.loc[len(availability)] = [name, "Ensemble", "Failed", str(exc), "Very high"]
        leaderboard = _rank(rows, ranking_weights)

    active_name = str(leaderboard.iloc[0]["Model"])
    run_metadata = dict(metadata or {})
    run_metadata.update({
        "experiment_id": f"exp-{uuid.uuid4().hex[:10]}", "mode": mode,
        "validation_strategy": plan.name, "validation_reason": plan.reason,
        "target_transformation": transform_label,
        "uncertainty_method": f"Finite-sample split conformal ({interval_confidence:.0%} marginal coverage target)",
        "calibration_rows": len(X_calibration),
        "ranking_weights": ranking_weights or {"performance": 0.60, "stability": 0.20, "simplicity": 0.10, "generalization": 0.10},
        "ranking_formula": "Configurable composite of predictive performance, fold stability, simplicity, and generalization",
        "ensemble_decisions": ensemble_decisions,
        "software": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "training_config": {"test_size": test_size, "calibration_size": calibration_size, "interval_confidence": interval_confidence, "cv_folds_requested": cv_folds, "cv_folds_applied": effective_cv_folds, "tune": tune, "tuning_iterations": tuning_iterations, "tune_top_n": tune_top_n, "ensembles": include_ensembles, "validation_strategy": validation_strategy, "minimum_ensemble_improvement": minimum_ensemble_improvement},
    })
    if progress:
        progress(1.0, f"Selected {active_name}")
    return TrainingResult(
        leaderboard=leaderboard, pipelines=pipelines, active_model_name=active_name,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        X_calibration=X_calibration, y_calibration=y_calibration,
        test_predictions=predictions, residual_std=residual_std,
        conformal_radius=radii, conformal_coverage=coverages, metadata=run_metadata,
        availability=availability, fold_scores=fold_scores, tuning_results=tuning_results,
    )
