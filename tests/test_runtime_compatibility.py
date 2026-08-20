import inspect

import numpy as np
import pandas as pd
import pytest
import src.models.predictor as predictor_module
from src.utils.schema import suggest_features
from src.utils.runtime import require_module_api
from streamlit.testing.v1 import AppTest


def test_stale_streamlit_module_is_reloaded_to_required_api():
    def stale_predict_batch(pipeline, rows, features):
        return []

    predictor_module.predict_batch = stale_predict_batch
    refreshed = require_module_api(
        "src.models.predictor",
        required_parameters={"predict_batch": {"contract"}},
    )
    assert "contract" in inspect.signature(refreshed.predict_batch).parameters


def test_upload_page_migrates_a_session_missing_expert_mode():
    app = AppTest.from_file("app/main.py").run(timeout=30)
    del app.session_state["expert_mode"]
    app.switch_page("pages/1_Upload_Data.py").run(timeout=30)
    assert not app.exception
    assert app.session_state["expert_mode"] is False
    assert app.session_state["state_schema_version"] == 2


def test_data_preparation_excludes_stale_columns_without_keyerror():
    app = AppTest.from_file("app/main.py").run(timeout=30)
    app.session_state["dataframe"] = pd.DataFrame({"area": [900, 1000] * 20, "bedrooms": [2, 3] * 20, "price": range(40)})
    app.session_state["target"] = "price"
    app.session_state["features"] = ["longitude", "housing_median_age", "area", "bedrooms"]
    app.switch_page("pages/2_Data_Cleaning.py").run(timeout=30)
    assert not app.exception
    assert app.session_state["features"] == ["area", "bedrooms"]


def test_feature_suggestion_keeps_unique_continuous_measurements_but_excludes_ids():
    frame = pd.DataFrame({
        "listing_id": [f"L-{index}" for index in range(150)],
        "area_sqft": np.linspace(450, 2450, 150),
        "latitude": np.linspace(28.1, 28.9, 150),
        "sale_price": np.linspace(4_000_000, 20_000_000, 150),
    })
    features, excluded = suggest_features(frame, "sale_price")
    assert {"area_sqft", "latitude"}.issubset(features)
    assert "listing_id" in excluded


@pytest.mark.parametrize("page", [
    "pages/1_Upload_Data.py", "pages/2_Data_Cleaning.py", "pages/3_Train_Models.py",
    "pages/4_Model_Insights.py", "pages/5_Predict.py", "pages/6_Model_Registry.py",
    "pages/7_Synthetic_Benchmark.py", "pages/8_Model_Science.py",
    "pages/9_System_Health.py", "pages/10_Platform_API.py",
])
def test_every_streamlit_route_renders_without_an_uncaught_exception(page):
    app = AppTest.from_file("app/main.py").run(timeout=30)
    assert not app.exception
    app.switch_page(page).run(timeout=30)
    assert not app.exception
