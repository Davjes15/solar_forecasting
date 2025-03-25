import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype
from typing import List, Optional

# --------------------------------------------------------------
# Implements the Partial Shading Detection block based on
# Section 3.4 and Equations (7)–(10) from the paper
# --------------------------------------------------------------

def detect_power_drop_events(
    df: pd.DataFrame,
    power_col: str = "Pm",
    delta_t_hours: int = 1,
    chi_threshold: float = 0.05
) -> pd.Series:
    """
    Detects timestamps with abrupt power changes that may indicate the onset of partial shading,
    based on Eq. (7) in the paper.
    |Pm(t + Δt) − Pm(t)| ≥ χ·Δt

    Parameters:
        df (pd.DataFrame): Input dataframe with datetime index and measured power.
        power_col (str): Column name for measured PV power.
        delta_t_hours (int): Time interval Δt (default = 1 hour).
        chi_threshold (float): Threshold χ for detecting abrupt power change.

    Returns:
        pd.Series: Boolean series where True indicates a shading onset candidate.
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    df["delta_P"] = df_sorted[power_col].diff(periods=delta_t_hours).abs()  # ΔP(t) = |P(t) - P(t-Δt)|
    shading_candidates = df["delta_P"] >= (chi_threshold * delta_t_hours)  # Eq. (7)
    return shading_candidates


def find_common_shading_angle(
    df_list: List[pd.DataFrame],
    zenith_col: str = "zenith",
    power_col: str = "Pm",
    delta_t_hours: int = 1,
    chi_threshold: float = 0.05
) -> Optional[float]:
    """
    Estimates the zenith angle θ where partial shading consistently starts
    according to Eqs. (8)

    Parameters:
        df_list (list): List of 3 DataFrames (consecutive clear-sky days).
        zenith_col (str): Column name for solar zenith angle Z(t).
        power_col (str): Measured power column.
        delta_t_hours (int): Time interval Δt for κ_ps computation (default 1 hour).
        chi_threshold (float): Threshold χ for detecting abrupt power change.

    Returns:
        Optional[float]: Estimated θ or None if not consistent.
    """
    # Identify candidate θ values for each day
    candidate_sets = []
    for df in df_list:
        drop_mask = detect_power_drop_events(df, power_col, delta_t_hours, chi_threshold)
        zenith_angles = df.loc[drop_mask, zenith_col].round(1).values
        candidate_sets.append(set(zenith_angles))

    # Find common zenith angles across all three days
    common_angles = set.intersection(*candidate_sets)
    return min(common_angles) if common_angles else None


def estimate_kappa_ps(
    df: pd.DataFrame,
    theta: float,
    zenith_col: str = "zenith",
    power_col: str = "Pm",
    delta_t: int = 1
) -> Optional[float]:
    """
    Estimate κ_ps = |Pm(t + Δt) − Pm(t)| / Pm(t) (Eq. 9).

    Parameters:
        df (pd.DataFrame): Data with zenith and power columns.
        theta (float): Zenith angle θ.
        zenith_col (str): Column for zenith.
        power_col (str): Column for power.
        delta_t (int): Time step Δt.

    Returns:
        Optional[float]: κ_ps value.
    """
    candidates = df[np.isclose(df[zenith_col], theta, atol=0.1)]
    for idx in candidates.index:
        if idx + delta_t < len(df):
            pm_t = df.at[idx, power_col]
            pm_dt = df.at[idx + delta_t, power_col]
            if pm_t > 0:
                return abs(pm_dt - pm_t) / pm_t
    return None


def apply_kappa_correction(
    df: pd.DataFrame,
    theta: float,
    kappa_ps: float,
    zenith_col: str = "zenith",
    pccs_col: str = "Pccs",
    output_col: str = "Ppccs"
) -> pd.DataFrame:
    """
    Apply κ_ps correction based on zenith threshold and dz/dt > 0 (Eq. 10).

    Parameters:
        df (pd.DataFrame): Input dataframe.
        theta (float): Zenith angle where shading begins.
        kappa_ps (float): Scaling factor for shaded portion.
        zenith_col (str): Column name with solar zenith angles.
        pccs_col (str): Original clear-sky corrected power (Pccs).
        output_col (str): Output column name for shaded-aware clear sky power (Ppccs).

    Returns:
        pd.DataFrame: Updated DataFrame with new Ppccs column.
    """
    df = df.copy()
    df["zenith_diff"] = df[zenith_col].diff()
    df[output_col] = np.where(
        (df[zenith_col] >= theta) & (df["zenith_diff"] > 0),
        kappa_ps * df[pccs_col],
        df[pccs_col]
    )
    return df


def partial_shadding_detection(
        df: pd.DataFrame,
        zenith_col: str = "zenith",
        power_col: str = "Pm",
        pccs_col: str = "Pccs",
        output_col: str = "Ppccs",
        delta_t_hours: int = 1,
        chi_threshold: float = 0.05
) -> pd.DataFrame:
    
    """
    Run the full partial shading correction pipeline.

    Parameters:
        df (pd.DataFrame): Input data with timestamps and required columns.
        zenith_col (str): Column name for solar zenith angle.
        power_col (str): Column name for measured PV power.
        pccs_col (str): Column name for system-calibrated clear-sky power.
        output_col (str): Output column name for corrected power.
        delta_t (int): Time step (in hours).
        chi (float): Drop threshold constant.

    Returns:
        pd.DataFrame: DataFrame with shading-aware corrected power.
    """
    required = ["timestamp", zenith_col, power_col, pccs_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if not is_datetime64_any_dtype(df['timestamp']):
        raise TypeError("'timestamp' column must be datetime type.")

    daily_dfs = [group for _, group in df.groupby(df['timestamp'].dt.date)]
    if len(daily_dfs) < 3:
        raise ValueError("Need at least 3 clear-sky days.")

    dfs_for_analysis = daily_dfs[:3]

    theta = find_common_shading_angle(
        dfs_for_analysis,
        zenith_col=zenith_col,
        power_col=power_col,
        delta_t_hours=delta_t_hours,
        chi_threshold=chi_threshold
    )
    if theta is None:
        df[output_col] = df[pccs_col]
        return df

    kappa_ps = estimate_kappa_ps(
        df,
        theta,
        zenith_col=zenith_col,
        power_col=power_col,
        delta_t_hours=delta_t_hours
    )
    if kappa_ps is None:
        df[output_col] = df[pccs_col]
        return df

    return apply_kappa_correction(df, theta, kappa_ps, zenith_col, pccs_col, output_col)
