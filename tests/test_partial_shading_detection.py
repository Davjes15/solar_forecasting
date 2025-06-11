import pytest
import pandas as pd
import numpy as np
from solar_forecasting.partial_shading_detection import (
    detect_power_drop_events,
    find_common_shading_angle,
    estimate_kappa_ps,
    apply_kappa_correction,
    partial_shading_detection
)


POWER_COL = "Pm"
ZENITH_COL = "zenith"
PCCS_COL = "Pccs"
OUTPUT_COL = "Ppccs"
CHI_THRESHOLD = 0.05
DELTA_T_HOURS = 1


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Fixture: Returns a clean DataFrame with steadily increasing zenith and constant power."""
    ts = pd.date_range(start="2024-06-01", periods=24, freq="h", tz="UTC")
    zenith = 50 + np.arange(24)
    power = np.full(24, 1.0)
    return pd.DataFrame({
        "timestamp": ts,
        ZENITH_COL: zenith,
        POWER_COL: power,
        PCCS_COL: power * 1.05
    })

@pytest.fixture
def shaded_df(simple_df: pd.DataFrame) -> pd.DataFrame:
    """Fixture: Returns a DataFrame with an artificial power drop to simulate partial shading."""
    df = simple_df.copy()
    df.at[10, POWER_COL] -= 0.2  # simulate a shading event
    return df

@pytest.fixture
def df_list():
    # Simulate 3 days, each with 24 hourly records, with a forced drop at zenith = 65.0
    days = ["2024-06-01", "2024-06-02", "2024-06-03"]
    dfs = []
    for day in days:
        ts = pd.date_range(start=f"{day} 00:00", periods=24, freq="h", tz="UTC")
        zenith = np.linspace(50, 75, 24)
        pm = np.ones(24)
        # Introduce a drop at index 12 (zenith ~ 65.2)
        pm[12] = 0.85  # Drop > 0.05 compared to neighbors (which are 1.0)
        data = pd.DataFrame({
            "timestamp": ts,
            "zenith": zenith,
            "Pm": pm,
            "Pccs": np.ones(24) * 1.05,
        })
        dfs.append(data)
    return dfs


def test_detect_power_drop_events(shaded_df: pd.DataFrame):
    """Test: detect_power_drop_events identifies expected drop."""
    mask = detect_power_drop_events(shaded_df, POWER_COL, DELTA_T_HOURS, CHI_THRESHOLD)
    assert isinstance(mask, pd.Series)
    assert mask.any(), "No power drop events detected when one is present."


def test_find_common_shading_angle(df_list: list[pd.DataFrame]):
    """Test: find_common_shading_angle returns consistent angle from multiple days."""
    theta = find_common_shading_angle(df_list, ZENITH_COL, POWER_COL, DELTA_T_HOURS, CHI_THRESHOLD)
    assert theta is not None
    assert isinstance(theta, (float, np.floating))
    assert 50 <= theta <= 70, "Shading angle seems out of expected range."


def test_estimate_kappa_ps(shaded_df: pd.DataFrame):
    """Test: estimate_kappa_ps computes a valid scaling factor."""
    theta = 59.0  # Known zenith at index 9
    kappa = estimate_kappa_ps(shaded_df, theta, ZENITH_COL, POWER_COL, DELTA_T_HOURS)
    assert kappa is not None
    assert 0 < kappa < 1, "Kappa_ps should be between 0 and 1."


def test_apply_kappa_correction(simple_df: pd.DataFrame):
    """Test: apply_kappa_correction applies scaling for zenith > theta with dz/dt > 0."""
    theta = 55.0
    kappa_ps = 0.9
    result = apply_kappa_correction(simple_df, theta, kappa_ps, ZENITH_COL, PCCS_COL, OUTPUT_COL)
    assert OUTPUT_COL in result.columns
    assert result.loc[10, OUTPUT_COL] <= result.loc[10, PCCS_COL], "Correction not applied properly."
    

def test_partial_shading_detection(df_list: list[pd.DataFrame]):
    """Integration Test: Verify full pipeline runs without errors and modifies output as expected."""
    df_all = pd.concat(df_list, ignore_index=True)
    result = partial_shading_detection(df_all, ZENITH_COL, POWER_COL, PCCS_COL, OUTPUT_COL, DELTA_T_HOURS, CHI_THRESHOLD)

    assert OUTPUT_COL in result.columns
    assert not result[OUTPUT_COL].isnull().any(), "Output contains NaNs"
    assert (result[OUTPUT_COL] <= result[PCCS_COL]).all(), "Corrected power exceeds original"