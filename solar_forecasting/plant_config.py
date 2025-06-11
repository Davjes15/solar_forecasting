# Description: Configuration file for the plant
from pydantic import BaseModel, Field, field_validator
"""
Plant Configuration for Solar Forecasting
    latitude (float): Latitude of the plant location in decimal degrees.
    longitude (float): Longitude of the plant location in decimal degrees.
    elevation (int): Elevation of the plant location in meters above sea level.
    capacity_kwp (int): Total capacity of the solar plant in kilowatts peak (kWp).
    n_modules (int): Number of solar modules in the plant.
    module_area (float): Area of a single solar module in square meters.
    tilt (int): Tilt angle of the solar modules in degrees.
    azimuth (int): Azimuth angle of the solar modules in degrees (0° = North, 90° = East, 180° = South, 270° = West).
"""

from pydantic import BaseModel, Field, field_validator


class PVPlantConfig(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    elevation: int = Field(..., ge=0, description="Elevation in meters above sea level")
    capacity_kwp: int = Field(..., gt=0)
    n_modules: int = Field(..., gt=0)
    module_area: float = Field(..., gt=0)
    tilt: int = Field(..., ge=0, le=90)
    azimuth: int = Field(..., ge=0, le=360)

    @property
    def total_area(self) -> float:
        """Total area of all modules in square meters."""
        return self.n_modules * self.module_area

    @field_validator("tilt", "azimuth", mode="before")
    @classmethod
    def coerce_to_int(cls, v):
        return int(v)


# Optional: default config instance
default_plant_config = PVPlantConfig(
    latitude=48.14951,
    longitude=11.56999,
    elevation=516,
    capacity_kwp=3,
    n_modules=12,
    module_area=1.67,
    tilt=30,
    azimuth=200
)