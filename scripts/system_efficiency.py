import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# --------------------------------------------------------------
# 2. PV system efficiency calibration module (κc and κt)
# --------------------------------------------------------------

def compute_daily_energy(df: pd.DataFrame,
    pm_col: str = 'Pm_kW',
    pcs_col: str = 'Pcs_kW',
    cloud_col: str = 'cloudiness_index'
) -> pd.DataFrame:
    """
    Aggregate hourly power measurements to daily total energy (kWh).

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'timestamp' and 'Pm' columns.
        'Pm' is the measured power output, 'Pcs' is the clear-sky power output,
        and 'cloudiness' is the cloud index.
        'timestamp' should be in datetime format.
        'Pm' and 'Pcs' should be in kW o watts (W).
        'cloudiness' should be a numeric value representing cloud index.
        'temperature' should be in Celsius.
    
    Returns:
        pd.DataFrame: Daily energy production with 'date' and 'Pm' columns.
    """
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    daily = df.groupby('date').agg({
        pm_col: 'sum',
        pcs_col: 'sum',
        cloud_col: 'mean'
    }).rename(columns={
        pm_col: 'Pm_day_kWh',
        pcs_col: 'Pcs_day_kWh',
        cloud_col: 'cloudiness_day_index'
    })
    daily['rel_energy'] = daily['Pm_day_kWh'] / daily['Pcs_day_kWh']
    return daily.reset_index()


def detect_clear_sky_days(
        daily_df: pd.DataFrame, 
        cloud_thresh: float = 4, 
        rel_energy_thresh: float = 0.9
) -> list:
    """
    Identify days with low cloudiness and high relative measured energy.

    Parameters:
        daily_df: Daily-aggregated DataFrame
        cloud_thresh: Max average cloud index to consider a day clear
        rel_energy_thresh: Min ratio Pm_day / Pcs_day for clear days

    Returns:
        List of dates that meet clear-sky criteria
    """
    is_clear = (daily_df['cloudiness_day_index'] <= cloud_thresh) & (daily_df['rel_energy'] >= rel_energy_thresh)
    return list(daily_df.loc[is_clear, 'date'])


def find_consecutive_clear_windows(clear_days: list, window_size: int = 3) -> list:
    """
    Identify sequences of N consecutive clear-sky days.

    Parameters:
        clear_days: List of clear day dates (as datetime.date)
        window_size: Number of consecutive days required

    Returns:
        List of tuples with (start_date, end_date)
    """
    clear_days = sorted(pd.to_datetime(clear_days))
    result = []
    for i in range(len(clear_days) - window_size + 1):
        window = clear_days[i:i + window_size]
        if (window[-1] - window[0]).days == window_size - 1:
            result.append((window[0], window[-1]))
    return result


def extract_window_data(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Filter the DataFrame to the date range specified by a 3-day clear-sky window.
    """
    return df[(df['timestamp'].dt.date >= start.date()) & (df['timestamp'].dt.date <= end.date())]


def optimize_kappa_c( df: pd.DataFrame,
    pm_col: str = 'Pm_kW',
    pcs_col: str = 'Pcs_kW'
) -> float:
    """
    Optimize system correction factor κc using MAPE (Mean Absolute Percentage Error) minimization between
    modeled and measured power outputs.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Pcs' and 'Pm' columns.

    Returns:
        float: Optimized κc value.
    """
    """
    Implements part of Formula 6: optimize κc when T <= 25°C.
    The objective function computes MAPE between κc·Pcs and Pm.
    """
    pcs, pm = df[pcs_col].values, df[pm_col].values

    def objective(kc):
        return np.mean(np.abs((kc * pcs - pm) / pm)) * 100

    return minimize_scalar(objective, bounds=(0.5, 1.0), method='bounded').x


def optimize_kappa_t(df: pd.DataFrame,
    kappa_c: float,
    temp_threshold: float = 25,
    pm_col: str = 'Pm_kW',
    pcs_col: str = 'Pcs_kW',
    temp_col: str = 'temperature_C'
) -> float:
    """
    Optimize temperature correction factor κt for periods above the given
    temperature threshold, based on minimizing MAPE.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Pcs', 'Pm', and 'temperature'.
        kappa_c (float): Pre-calibrated system correction factor κc.
        temp_threshold (float): Temperature threshold above which κt is applied.

    Returns:
        float: Optimized κt value.
    """
    """
    Implements Formula 6 with Formula 5 embedded:
    Minimizes MAPE where:
    Pccs = κt·κc·Pcs if T > 25°C, else κc·Pcs.
    """
    pcs = df[pcs_col].values
    pm = df[pm_col].values
    temp = df[temp_col].values

    def objective(kt):
        pccs = np.where(temp > temp_threshold, kt * kappa_c * pcs, kappa_c * pcs)  # Implementting Formula 5
        return np.mean(np.abs((pccs - pm) / pm)) * 100

    return minimize_scalar(objective, bounds=(0.7, 1.0), method='bounded').x  # Implementing Formula 6


def apply_corrections(
    df: pd.DataFrame, 
    kappa_c: float, 
    kappa_t: float, 
    temp_threshold: float = 25,
    pcs_col: str = 'Pcs_kW',
    temp_col: str = 'temperature_C'
) -> pd.DataFrame:
    """
    Apply calibrated κc and κt to the clear-sky power values to produce Pccs.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Pcs' and 'temperature'.
        kappa_c (float): System loss correction factor.
        kappa_t (float): Temperature loss correction factor.
        temp_threshold (float): Temperature threshold above which κt is applied.

    Returns:
        pd.DataFrame: Updated DataFrame with a new 'Pccs' column.
    """
    """
    Applies Formula 5 from the paper:
    Pccs = κt·κc·Pcs when T > 25°C, else κc·Pcs.
    """
    pcs = df[pcs_col].values
    temp = df[temp_col].values
    df['Pccs_kW'] = np.where(temp > temp_threshold, kappa_t * kappa_c * pcs, kappa_c * pcs)
    return df


def calibrate_system_efficiency(
    df: pd.DataFrame, 
    temp_threshold: float = 25, 
    window_hours: int = 72,
    pm_col: str = 'Pm_kW',
    pcs_col: str = 'Pcs_kW',
    cloud_col: str = 'cloudiness_index',
    temp_col: str = 'temperature_C'
) -> tuple[pd.DataFrame, tuple[float, float]]:
    
    """
    Calibrate system efficiency correction factors κc and κt.
    Based on a configurable clear-sky window size defined by:
        - window_hours (default = 72 for 3 days)
        - low cloudiness (forecast data)
        - high relative daily measured energy

    Parameters:
        df (pd.DataFrame): input dataframe with hourly data.
        temp_threshold (float): temperature threshold for κt calibration.
        window_hours (int): size of clear-sky detection window in hours (default 72).

    Returns:
        Tuple[pd.DataFrame, (kappa_c, kappa_t)]
    """
    if window_hours % 24 != 0:
        raise ValueError("window_hours must be divisible by 24 to represent full days")
    df = df.copy()
    daily = compute_daily_energy(df, pm_col=pm_col, pcs_col=pcs_col, cloud_col=cloud_col)
    clear_days = detect_clear_sky_days(daily)
    windows = find_consecutive_clear_windows(clear_days, window_size=window_hours // 24)

    if not windows:
        print("[INFO] No 3-day clear-sky window found. Using defaults κc = κt = 1.0")
        return apply_corrections(df, 1.0, 1.0, temp_threshold, pcs_col=pcs_col, temp_col=temp_col), (1.0, 1.0)

    start_date, end_date = windows[0]  # Use the first valid 3-day window
    window_df = extract_window_data(df, start=start_date, end=end_date)
    avg_temp = window_df[temp_col].mean()

    kappa_c, kappa_t = 1.0, 1.0
    if avg_temp > temp_threshold:
        kappa_t = optimize_kappa_t(window_df, kappa_c, temp_threshold, pm_col=pm_col, pcs_col=pcs_col, temp_col=temp_col)
        print(f"[INFO] Calibrated κt = {kappa_t:.4f} (warm window)")
    else:
        kappa_c = optimize_kappa_c(window_df, pm_col=pm_col, pcs_col=pcs_col)
        print(f"[INFO] Calibrated κc = {kappa_c:.4f} (cool window)")

    df = apply_corrections(df, kappa_c, kappa_t, temp_threshold, pcs_col=pcs_col, temp_col=temp_col)
    return df, (kappa_c, kappa_t)