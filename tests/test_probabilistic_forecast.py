import pytest
import pandas as pd
import numpy as np
from scipy.stats import rv_continuous

from solar_forecasting.probabilistic_forecast import (
    compute_relative_error,
    categorize_by_cloudiness,
    fit_best_distribution,
    generate_probabilistic_forecast,
    run_probabilistic_forecast_pipeline
)

@pytest.fixture
def synthetic_df():
    np.random.seed(42)
    hours = 24
    timestamps = pd.date_range("2022-06-01", periods=hours, freq="h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "Ppf": np.linspace(500, 1000, hours),
        "Pm": np.linspace(480, 980, hours),
        "cloudiness": np.clip(np.random.normal(0.5, 0.2, hours), 0, 1)
    })
    return df

def test_compute_relative_error(synthetic_df):
    result = compute_relative_error(synthetic_df)
    assert "lambda" in result.columns
    assert not result["lambda"].isnull().all()

def test_categorize_by_cloudiness(synthetic_df):
    df_with_lambda = compute_relative_error(synthetic_df)
    categorized, df_categorized = categorize_by_cloudiness(df_with_lambda)
    assert isinstance(categorized, dict)
    assert "lambda_class" in df_categorized.columns
    assert all(isinstance(k, int) for k in categorized)

def test_fit_best_distribution_returns_valid(synthetic_df):
    df_with_lambda = compute_relative_error(synthetic_df)
    categorized, _ = categorize_by_cloudiness(df_with_lambda)
    for data in categorized.values():
        dist, params = fit_best_distribution(data)
        assert isinstance(dist, rv_continuous)
        assert isinstance(params, tuple)

def test_generate_probabilistic_forecast(synthetic_df):
    df_with_lambda = compute_relative_error(synthetic_df)
    categorized, df_cat = categorize_by_cloudiness(df_with_lambda)
    fitted = {i: fit_best_distribution(data) for i, data in categorized.items()}
    quantiles = [0.1, 0.5, 0.9]
    result = generate_probabilistic_forecast(df_cat, fitted, quantiles)
    for q in quantiles:
        assert f"Ppp_{int(q * 100)}" in result.columns

def test_run_probabilistic_forecast_pipeline(synthetic_df):
    result = run_probabilistic_forecast_pipeline(synthetic_df)
    for q in range(10, 100, 10):
        assert f"Ppp_{q}" in result.columns