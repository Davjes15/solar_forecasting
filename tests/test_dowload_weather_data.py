# tests/test_download_weather_hourly_daily.py

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from solar_forecasting.download_weather_data import fetch_nasa_power_weather

@pytest.fixture
def mocked_nasa_response():
    return {
        "properties": {
            "parameter": {
                "T2M": {"2024010101": 5.0, "2024010102": 6.0},
                "RH2M": {"2024010101": 80, "2024010102": 82},
                "CLOUD_AMT": {"2024010101": 50, "2024010102": 40}
            }
        }
    }

@patch("solar_forecasting.download_weather_data.requests.get")
def test_fetch_nasa_power_weather_hourly(mock_get, mocked_nasa_response):
    # Arrange
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mocked_nasa_response
    mock_get.return_value = mock_response

    # Act
    df = fetch_nasa_power_weather(
        latitude=48.0,
        longitude=11.5,
        start="20240101",
        end="20240101",
        parameters=("T2M", "RH2M", "CLOUD_AMT"),
        resolution="hourly"
    )

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"datetime", "temperature", "humidity", "cloudiness"}
    assert df.shape[0] == 2
    assert df["cloudiness"].iloc[0] == 0.5
    assert df["temperature"].iloc[1] == 6.0
    assert df["humidity"].iloc[1] == 82