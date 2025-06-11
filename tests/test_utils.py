from solar_forecasting import utils
from solar_forecasting.utils import plot_seasonal_forecasts, plot_probabilistic_forecast, simulate_realtime_forecast
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def synthetic_forecast_df():
    """Create synthetic forecast data with quantiles and measured output."""
    dates = pd.date_range("2024-06-01", periods=48, freq="h")
    data = {
        "timestamp": dates,
        "pv_output_kw": np.random.uniform(0, 1, size=48)
    }
    for q in range(10, 100, 10):
        data[f"Ppp_{q}"] = np.random.uniform(0, q / 100, size=48)
    return pd.DataFrame(data)

def test_plot_seasonal_forecasts_runs_without_error(synthetic_forecast_df):
    """Smoke test to ensure plot_seasonal_forecasts runs and plots without error."""
    selected_days = {
        "Summer": "2024-06-01",
        "Winter": "2024-06-02"
    }
    # Should not raise any exceptions
    utils.plot_seasonal_forecasts(synthetic_forecast_df, selected_days)


def test_plot_probabilistic_forecast_runs_without_error(synthetic_forecast_df):
    """Smoke test for plot_probabilistic_forecast for one valid day."""
    utils.plot_probabilistic_forecast(
        synthetic_forecast_df,
        date_str="2024-06-01",
        season="Summer"
    )


def test_simulate_realtime_forecast_returns_dataframe(monkeypatch):
    """Unit test for simulate_realtime_forecast with mocked dependencies."""
    index = pd.date_range("2024-01-01", periods=300, freq="h")
    df = pd.DataFrame({
        "temperature_wx": np.random.rand(300),
        "cloudiness": np.random.rand(300),
        "humidity": np.random.rand(300),
        "Ppccs": np.random.rand(300) + 0.5,
        "Pm": np.random.rand(300),
    }, index=index)

    # Patch dependencies from point_forecast_prediction
    monkeypatch.setattr(utils, "prepare_training_data",
                        lambda df, **kwargs: (df[["temperature_wx"]], df["Pm"]))
    monkeypatch.setattr(utils, "train_regression_tree",
                        lambda X, y, **kwargs: "mock_model")
    monkeypatch.setattr(utils, "forecast_point_generation",
                        lambda model, df, **kwargs: df.assign(Ppf=np.random.rand(len(df))))

    result = utils.simulate_realtime_forecast(
        df,
        feature_cols=["temperature_wx", "cloudiness", "humidity"],
        start_date="2024-01-10",
        end_date="2024-01-12"
    )

    assert isinstance(result, pd.DataFrame), "Expected output to be a DataFrame"
    assert not result.empty, "Expected output DataFrame to be non-empty"
    assert set(["Pm", "Ppccs", "Ppf", "lambda_hat"]).issubset(result.columns)
