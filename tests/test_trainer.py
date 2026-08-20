import numpy as np
import pandas as pd

from src.data.cleaner import CleaningConfig
from src.models.trainer import train_regressors


def make_regression_frame(rows: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    size = rng.normal(1500, 300, rows)
    bedrooms = rng.integers(1, 6, rows)
    location = rng.choice(["North", "South", "Central"], rows)
    price = size * 180 + bedrooms * 12000 + (location == "North") * 25000 + rng.normal(0, 8000, rows)
    return pd.DataFrame({"size": size, "bedrooms": bedrooms, "location": location, "price": price})


def test_trainer_returns_ranked_fitted_pipelines():
    frame = make_regression_frame()
    result = train_regressors(
        frame,
        target="price",
        features=["size", "bedrooms", "location"],
        cleaning_config=CleaningConfig(),
        cv_folds=3,
        tune=False,
    )
    assert len(result.leaderboard) == 5
    assert result.active_model_name in result.pipelines
    assert result.leaderboard.iloc[0]["CV RMSE"] <= result.leaderboard.iloc[-1]["CV RMSE"]
    assert len(result.active_pipeline.predict(result.X_test)) == len(result.X_test)


def test_log_target_and_tuning_paths_work_together():
    frame = make_regression_frame(70)
    result = train_regressors(
        frame,
        target="price",
        features=["size", "bedrooms", "location"],
        cv_folds=3,
        log_target=True,
        tune=True,
        tuning_iterations=1,
    )
    prediction = result.active_pipeline.predict(result.X_test.head(2))
    assert len(prediction) == 2
    assert (prediction > 0).all()
