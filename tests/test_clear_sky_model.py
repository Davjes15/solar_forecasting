import pytest
import pandas as pd
from datetime import datetime
from solar_forecasting.clear_sky_model import calculate_clear_sky_irradiance
from solar_forecasting.plant_config import PVPlantConfig


@pytest.fixture
def example_config() -> PVPlantConfig:
    return PVPlantConfig(
        latitude=48.14951,
        longitude=11.56999,
        elevation=516,
        capacity_kwp=3,
        n_modules=12,
        module_area=1.67,
        tilt=30,
        azimuth=200
    )


@pytest.fixture
def example_dataframe():
    # Create a timezone-aware hourly time series
    timestamps = pd.date_range(
        start="2024-06-21 00:00:00",
        periods=24,
        freq="h",
        tz="UTC"
    )
    return pd.DataFrame({"timestamp": timestamps})


def test_clear_sky_output_schema(example_dataframe, example_config):
    """Ensure the output contains all expected columns."""
    result = calculate_clear_sky_irradiance(example_dataframe, example_config)

    expected_columns = {
        "timestamp",
        "zenith",
        "azimuth_sun",
        "GHI",
        "DNI",
        "DHI",
        "POA_irradiance",
        "Pcs_kW"
    }
    assert expected_columns.issubset(result.columns), \
        f"Missing columns: {expected_columns - set(result.columns)}"


def test_pcs_values_non_negative(example_dataframe, example_config):
    """Ensure all Pcs values are >= 0."""
    result = calculate_clear_sky_irradiance(example_dataframe, example_config)
    assert (result["Pcs_kW"] >= 0).all(), "Some Pcs values are negative"


def test_config_as_dict_equivalence(example_dataframe, example_config):
    """Ensure that using a dict instead of a model produces the same result."""
    result_model = calculate_clear_sky_irradiance(example_dataframe, example_config)
    result_dict = calculate_clear_sky_irradiance(example_dataframe, example_config.model_dump())

    pd.testing.assert_series_equal(result_model["Pcs_kW"], result_dict["Pcs_kW"], check_names=False)