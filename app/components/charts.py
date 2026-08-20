from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORS = ["#16A6A1", "#7C5CFC", "#F59E0B", "#EF476F", "#2D7FF9"]


def missingness_chart(profile: pd.DataFrame):
    data = profile.loc[profile["missing_%"] > 0].sort_values("missing_%")
    return px.bar(data, x="missing_%", y="column", orientation="h", color_discrete_sequence=[COLORS[0]])


def model_leaderboard_chart(leaderboard: pd.DataFrame):
    data = leaderboard.sort_values("Test RMSE", ascending=True)
    return px.bar(
        data,
        x="Test RMSE",
        y="Model",
        orientation="h",
        color="Test R²",
        color_continuous_scale="Tealgrn",
        title="Lower RMSE is better",
    )


def predicted_actual_chart(actual, predicted):
    frame = pd.DataFrame({"Actual": actual, "Predicted": predicted})
    figure = px.scatter(frame, x="Actual", y="Predicted", opacity=0.65)
    low = float(min(frame.min()))
    high = float(max(frame.max()))
    figure.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect prediction"))
    return figure


def residual_chart(actual, predicted):
    residuals = np.asarray(actual) - np.asarray(predicted)
    frame = pd.DataFrame({"Predicted": predicted, "Residual": residuals})
    figure = px.scatter(frame, x="Predicted", y="Residual", opacity=0.65)
    figure.add_hline(y=0, line_dash="dash")
    return figure


def importance_chart(frame: pd.DataFrame):
    data = frame.sort_values("importance")
    return px.bar(data, x="importance", y="feature", orientation="h", color_discrete_sequence=[COLORS[1]])
