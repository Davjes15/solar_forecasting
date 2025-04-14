# download_weather_data.py
# ------------------------------------------------------------------
# Safely downloads weather data from NASA POWER API (hourly or daily)
# and parses it into a DataFrame ready for PV forecasting.
# ------------------------------------------------------------------

import requests
import pandas as pd
from typing import Tuple


def fetch_nasa_power_weather(
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    parameters: Tuple[str] = (
        "T2M",     # Temperature at 2 meters [°C]
        "RH2M",    # Relative Humidity at 2 meters [%]
        "WS2M",    # Wind Speed at 2 meters [m/s]
        "WD2M",    # Wind Direction at 2 meters [°]
        "PS",      # Surface Pressure [kPa]
        "CLOUD_AMT",  # Cloud cover [%]
        "ALLSKY_SFC_SW_DWN",  # Global Horizontal Irradiance [W/m^2]
        "ALLSKY_SFC_LW_DWN",  # Longwave radiation [W/m^2]
        "ALLSKY_KT"            # Clearness index
    )
) -> pd.DataFrame:
    """
    Downloads weather data (hourly or daily fallback) from NASA POWER.

    Parameters:
    - latitude, longitude (float): location coordinates
    - start, end (str): date range in YYYYMMDD format
    - parameters (tuple): variable codes from NASA POWER

    Returns:
    - DataFrame with parsed weather features and timestamp
    """
    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    param_str = ','.join(parameters)
    url = (
        f"{base_url}?parameters={param_str}"
        f"&community=RE&longitude={longitude}&latitude={latitude}"
        f"&start={start}&end={end}&format=JSON&user=demo"
    )

    print(f"Requesting data from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # Peek into the first parameter to determine if data is hourly (dict) or daily (flat)
    param_values = data['properties']['parameter'][parameters[0]]
    example_day = next(iter(param_values))
    example_value = param_values[example_day]

    records = []

    # If the first parameter's value is a nested dict, assume hourly data
    if isinstance(example_value, dict):
        # Hourly data case
        for date, hours in param_values.items():
            for hour, _ in hours.items():
                record = {'datetime': pd.to_datetime(f"{date} {hour}:00", utc=True)}
                for param in parameters:
                    record[param] = data['properties']['parameter'][param][date][hour]
                records.append(record)
    # If the first parameter's value is a float/int, assume daily data
    else:
        # Daily fallback
        for date in param_values:
            record = {'datetime': pd.to_datetime(f"{date[:8]} 12:00", utc=True)}
            for param in parameters:
                record[param] = data['properties']['parameter'][param][date]
            records.append(record)

    df = pd.DataFrame.from_records(records).sort_values("datetime").reset_index(drop=True)

    # Rename columns to match pipeline
    df.rename(columns={
        "T2M": "temperature",
        "RH2M": "humidity",
        "WS2M": "wind_speed",
        "WD2M": "wind_direction",
        "PS": "mean_sea_level_pressure",
        "CLOUD_AMT": "cloudiness",
        "ALLSKY_SFC_SW_DWN": "ghi",
        "ALLSKY_SFC_LW_DWN": "lw_irradiance",
        "ALLSKY_KT": "clearness_index"
    }, inplace=True)

    df["cloudiness"] = df["cloudiness"] / 100.0  # normalize to [0, 1]
    return df


if __name__ == "__main__":
    df_weather = fetch_nasa_power_weather(
        latitude=48.14951,
        longitude=11.56999,
        start="20190101",
        end="20191231"
    )
    print(df_weather.head())
