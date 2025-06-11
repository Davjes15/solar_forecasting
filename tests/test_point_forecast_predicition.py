# tests/test_point_forecast_prediction.py

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# Adjust this import if you haven’t yet renamed `scripts/` → `solar_forecasting/`
from solar_forecasting.point_forecast_prediction import (
    prepare_training_data,
    train_regression_tree,
    forecast_point_generation,
    create_synthetic_forecast_data,
)


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """Create reproducible synthetic forecast data."""
    return create_synthetic_forecast_data(seed=42, hours=24)


def test_create_synthetic_reproducible():
    df1 = create_synthetic_forecast_data(seed=123, hours=10)
    df2 = create_synthetic_forecast_data(seed=123, hours=10)
    pd.testing.assert_frame_equal(df1, df2)

def test_create_synthetic_structure(synthetic_data):
    expected = {"timestamp", "temperature", "cloudiness", "humidity", "Ppccs", "Pm"}
    assert expected.issubset(synthetic_data.columns)
    assert len(synthetic_data) == 24
    assert pd.api.types.is_datetime64_any_dtype(synthetic_data["timestamp"])


def test_prepare_training_data_shapes_and_ranges(synthetic_data):
    feature_cols = ["temperature", "cloudiness", "humidity"]
    X, y = prepare_training_data(synthetic_data, feature_cols, target_col="Pm", reference_col="Ppccs")
    # X/y lengths match and features preserved
    assert len(X) == len(y)
    assert set(X.columns) == set(feature_cols)
    # y was clipped to [0, 1.5]
    assert (y >= 0).all() and (y <= 1.5).all()


def test_train_regression_tree_model_and_predict(synthetic_data):
    feature_cols = ["temperature", "cloudiness", "humidity"]
    X, y = prepare_training_data(synthetic_data, feature_cols)
    
    model = train_regression_tree(
        X, y,
        n_estimators=5,
        max_samples=0.8,
        bootstrap=True,
        random_state=0
    )

    assert isinstance(model, BaggingRegressor)

    # Predict on a few rows and validate shape
    preds = model.predict(X.head(3))
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3,)

    # Optional: check all base estimators are decision trees
    for est in model.estimators_:
        assert isinstance(est, DecisionTreeRegressor)


def test_forecast_point_generation_column_and_values(synthetic_data):
    feature_cols = ["temperature", "cloudiness", "humidity"]
    X, y = prepare_training_data(synthetic_data, feature_cols)
    model = train_regression_tree(X, y, n_estimators=5, max_samples=0.8, random_state=0)
    df_out = forecast_point_generation(
        model,
        synthetic_data,
        feature_cols,
        reference_col="Ppccs",
        output_col="Ppf"
    )
    # Column exists
    assert "Ppf" in df_out.columns
    # Values are between 0 and Ppccs
    assert (df_out["Ppf"] >= 0).all()
    assert (df_out["Ppf"] <= df_out["Ppccs"]).all()