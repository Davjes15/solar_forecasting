from typing import List
from datetime import datetime
from pydantic import BaseModel, Field, condecimal, confloat, constr


class SiteMetadata(BaseModel):
    """
    Describes fixed characteristics of a PV installation.
    """

    latitude: confloat(ge=-90.0, le=90.0) = Field(
        ...,
        description="Latitude of the plant location in decimal degrees (°). "
                    "Valid range: -90.0 to +90.0."
    )
    longitude: confloat(ge=-180.0, le=180.0) = Field(
        ...,
        description="Longitude of the plant location in decimal degrees (°). "
                    "Valid range: -180.0 to +180.0."
    )
    elevation: float = Field(
        ...,
        description="Elevation of the plant location in meters above sea level."
    )
    capacity_kwp: float = Field(
        ...,
        description="Installed peak capacity, in kilowatts peak (kWp)."
    )
    module_area: float = Field(
        ...,
        description="Area of a single solar module, in square meters (m²)."
    )
    tilt: confloat(ge=0.0, le=90.0) = Field(
        ...,
        description="Tilt angle of the solar modules in degrees. "
                    "Valid range: 0° (flat) to 90° (vertical)."
    )
    azimuth: confloat(ge=0.0, le=360.0) = Field(
        ...,
        description="Azimuth angle of the solar modules in degrees, "
                    "where 0° = North, 90° = East, 180° = South, 270° = West. "
                    "Valid range: 0.0 to 360.0."
    )


class Observation(BaseModel):
    """
    One row of time‐series data for PV output and weather conditions.
    """

    timestamp: datetime = Field(
        ...,
        description="Timezone‐aware datetime of the observation (ISO 8601)."
    )
    measured_power: float = Field(
        ...,
        description="Measured PV output, in watts (W) or kilowatts (kW). "
                    "Choose consistent units across your dataset."
    )
    cloud_cover: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Normalized cloudiness fraction, from 0.0 (clear) to 1.0 (fully overcast)."
    )
    temperature: float = Field(
        ...,
        description="Ambient air temperature in degrees Celsius (°C)."
    )
    dew_point: float = Field(
        ...,
        description="Dew point temperature in degrees Celsius (°C)."
    )
    humidity: confloat(ge=0.0, le=100.0) = Field(
        ...,
        description="Relative humidity as a percentage. "
                    "Valid range: 0.0 to 100.0 (%)."
    )
    wind_speed: float = Field(
        ...,
        description="Wind speed in meters per second (m/s)."
    )
    wind_direction: confloat(ge=0.0, le=360.0) = Field(
        ...,
        description="Wind direction in degrees from true north. "
                    "Valid range: 0.0 to 360.0°."
    )
    mean_sea_level_pressure: float = Field(
        ...,
        description="Atmospheric pressure at mean sea level, in hectopascals (hPa)."
    )


class PVTimeSeries(BaseModel):
    """
    Combines a SiteMetadata instance with a list of timestamped observations.
    """

    site: SiteMetadata = Field(
        ...,
        description="Static metadata describing the PV installation."
    )
    records: List[Observation] = Field(
        ...,
        description="Chronological list of PV and weather observations."
    )