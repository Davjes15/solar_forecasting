import pytest
import pandas as pd
import numpy as np
from datetime import timedelta
from solar_forecasting.system_efficiency import (
    compute_daily_energy,
    detect_clear_sky_days,
    find_consecutive_clear_windows,
    extract_window_data,
    optimize_kappa_c,
    optimize_kappa_t,
    apply_corrections,
    calibrate_system_efficiency
)


@pytest.fixture
def synthetic_hourly_data():
    hours = 72
    timestamps = pd.date_range("2024-06-01", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "Pm_kW": np.random.uniform(450, 500, size=hours),
        "Pcs_kW": np.random.uniform(460, 520, size=hours),
        "cloudiness_index": np.random.uniform(0, 2, size=hours),
        "temperature_C": np.random.uniform(20, 30, size=hours),
    })

def test_compute_daily_energy(synthetic_hourly_data):
    result = compute_daily_energy(synthetic_hourly_data)
    assert "Pm_day_kWh" in result
    assert "Pcs_day_kWh" in result
    assert "cloudiness_day_index" in result
    assert "rel_energy" in result
    assert len(result) == 3

def test_detect_clear_sky_days(synthetic_hourly_data):
    daily = compute_daily_energy(synthetic_hourly_data)
    clear_days = detect_clear_sky_days(daily, cloud_thresh=3, rel_energy_thresh=0.7)
    assert isinstance(clear_days, list)

def test_find_consecutive_clear_windows():
    dates = pd.date_range("2024-06-01", periods=5, freq="D").date
    windows = find_consecutive_clear_windows(list(dates), window_size=3)
    assert all(len(win) == 2 for win in windows)

def test_extract_window_data(synthetic_hourly_data):
    start = synthetic_hourly_data["timestamp"].min()
    end = synthetic_hourly_data["timestamp"].min() + timedelta(days=2)
    subset = extract_window_data(synthetic_hourly_data, start, end)
    assert not subset.empty
    assert subset["timestamp"].min().date() >= start.date()
    assert subset["timestamp"].max().date() <= end.date()

def test_optimize_kappa_c(synthetic_hourly_data):
    kappa_c = optimize_kappa_c(synthetic_hourly_data)
    assert 0.5 <= kappa_c <= 1.0

def test_optimize_kappa_t(synthetic_hourly_data):
    kappa_t = optimize_kappa_t(synthetic_hourly_data, kappa_c=1.0)
    assert 0.7 <= kappa_t <= 1.0

def test_apply_corrections(synthetic_hourly_data):
    corrected = apply_corrections(synthetic_hourly_data, kappa_c=0.95, kappa_t=0.98)
    assert "Pccs_kW" in corrected
    assert corrected["Pccs_kW"].notna().all()

def test_calibrate_system_efficiency(synthetic_hourly_data):
    calibrated_df, (kappa_c, kappa_t) = calibrate_system_efficiency(synthetic_hourly_data)
    assert "Pccs_kW" in calibrated_df
    assert 0.5 <= kappa_c <= 1.0
    assert 0.7 <= kappa_t <= 1.0