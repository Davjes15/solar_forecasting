# download_weather_hourly_daily.py
# ------------------------------------------------------------------
# Downloads hourly and daily weather data from NASA POWER API,
# then merges both datasets on timestamp to build a unified hourly DataFrame.
# ------------------------------------------------------------------

import requests
import pandas as pd
from typing import Tuple

def fetch_nasa_power_weather(
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    parameters: Tuple[str],
    resolution: str = "hourly"
) -> pd.DataFrame:
    """
    Fetch weather data from NASA POWER API.

    Parameters:
    - latitude, longitude: Geographic coordinates.
    - start, end: Date range in YYYYMMDD.
    - parameters: Tuple of parameter codes.
    - resolution: Either 'hourly' or 'daily'.

    Returns:
    - DataFrame indexed by datetime.
    """
    base_url = f"https://power.larc.nasa.gov/api/temporal/{resolution}/point"
    param_str = ','.join(parameters)
    url = (
        f"{base_url}?parameters={param_str}"
        f"&community=RE&longitude={longitude}&latitude={latitude}"
        f"&start={start}&end={end}&format=JSON&user=demo&time-standard=UTC"
    )

    print(f"Requesting {resolution} data from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
  
    param_data = data['properties']['parameter']
    records = {}

    for param, values in param_data.items():
        for timestamp, val in values.items():
            fmt = "%Y%m%d%H" if resolution=="hourly" else "%Y%m%d"
            dt = pd.to_datetime(timestamp, format=fmt, utc=True)
            if dt not in records:
                records[dt] = {}
            records[dt][param] = val

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "datetime"
    df = df.sort_index().reset_index()

    # Rename columns to match pipeline
    df.rename(columns={
        "T2M": "temperature",
        "RH2M": "humidity",
        "WS2M": "wind_speed",
        "WD2M": "wind_direction",
        "PS": "mean_sea_level_pressure",
        "CLOUD_AMT": "cloudiness",
    }, inplace=True)

    df["cloudiness"] = df["cloudiness"] / 100.0  # normalize to [0, 1]
    return df


if __name__ == "__main__":
    df_weather = fetch_nasa_power_weather(
        latitude=48.14951,
        longitude=11.56999,
        start="20190101",
        end="20191231",
        parameters=("T2M", "RH2M", "WS2M", "WD2M")
    )
    print(df_weather.head())
