# Description: Configuration file for the plant
"""
Dictionary containing the configuration parameters for the solar plant.

Keys:
    latitude (float): Latitude of the plant location in decimal degrees.
    longitude (float): Longitude of the plant location in decimal degrees.
    elevation (int): Elevation of the plant location in meters above sea level.
    capacity_kwp (int): Total capacity of the solar plant in kilowatts peak (kWp).
    n_modules (int): Number of solar modules in the plant.
    module_area (float): Area of a single solar module in square meters.
    tilt (int): Tilt angle of the solar modules in degrees.
    azimuth (int): Azimuth angle of the solar modules in degrees (0° = North, 90° = East, 180° = South, 270° = West).
"""


plant_config = {
    'latitude': 48.14951,
    'longitude': 11.56999,
    'elevation': 516,  # meters
    'capacity_kwp': 3,
    'n_modules': 12,
    'module_area': 1.67,  # m²
    'tilt': 30,  # degrees
    'azimuth': 200  # degrees
}