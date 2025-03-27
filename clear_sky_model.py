import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pvlib import location, solarposition, irradiance

# ------------------------------
# 1. Clear Sky Model
# ------------------------------

def calculate_clear_sky_irradiance(df: pd.DataFrame, config: dict, eta_ref: float = 0.20) -> pd.DataFrame:
    """
    Computes clear-sky PV power output (Pcs) using plant parameters and atmospheric conditions.

    Parameters:
    - df (pd.DataFrame): Input dataframe with a 'timestamp' column (timezone-aware).
    - config (dict): Dictionary containing PV system parameters.
    - eta_ref (float): Nominal reference efficiency of the PV modules (default = 0.20).

    Returns:
    - pd.DataFrame: Input dataframe with added columns for solar position, irradiance, and Pcs.
    """
    df = df.copy()

    # Validate and prepare timestamps
    if df['timestamp'].dt.tz is None:
        timestamps = pd.to_datetime(df['timestamp'], utc=True)
    else:
        timestamps = pd.DatetimeIndex(df['timestamp'])

    timestamps = timestamps[:len(df)]

    # Create pvlib Location object
    site = location.Location(
        latitude=config['latitude'],
        longitude=config['longitude'],
        tz='UTC',
        altitude=config['elevation']
    )

    # Step 1: Solar position — zenith and azimuth angles
    solar_pos = solarposition.get_solarposition(
        time=timestamps,
        latitude=site.latitude,
        longitude=site.longitude,
        altitude=site.altitude
    )
    df['zenith'] = solar_pos['zenith'].values
    df['azimuth_sun'] = solar_pos['azimuth'].values

    # Step 2: Clear-sky irradiance using Ineichen model
    # NOTE:
    # In the original paper, extraterrestrial irradiance (I₀) is used in Formulas (1)-(3).
    # We do not compute I₀ explicitly; instead, we rely on pvlib's get_clearsky() function,
    # which internally includes I₀ and transmittance effects.
    clearsky = site.get_clearsky(timestamps, model='ineichen')
    df['GHI'] = clearsky['ghi'].values  # Global Horizontal Irradiance (I_T)
    df['DNI'] = clearsky['dni'].values  # Direct Normal Irradiance (I_d)
    df['DHI'] = clearsky['dhi'].values  # Diffuse Horizontal Irradiance (I_as proxy)

    # Step 3: POA irradiance on tilted surface
    poa = irradiance.get_total_irradiance(
        surface_tilt=config['tilt'],
        surface_azimuth=config['azimuth'],
        dni=clearsky['dni'].values,
        ghi=clearsky['ghi'].values,
        dhi=clearsky['dhi'].values,
        solar_zenith=df['zenith'].values,
        solar_azimuth=df['azimuth_sun'].values
    )
    df['POA_irradiance'] = poa['poa_global']  # total irradiance on panel surface (I_{T,tilt})

    # Step 4: Calculate clear-sky PV power (Pcs)
    total_area = config['n_modules'] * config['module_area']
    df['Pcs'] = eta_ref * total_area * df['POA_irradiance']  # Pcs = η × A × I_tilt

    return df
