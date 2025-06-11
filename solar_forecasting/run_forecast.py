import pandas as pd
from solar_forecasting.plant_config import default_plant_config
from solar_forecasting.utils import load_and_prepare_data
from solar_forecasting.download_weather_data import fetch_nasa_power_weather
from solar_forecasting.clear_sky_model import calculate_clear_sky_irradiance
from solar_forecasting.partial_shading_detection import partial_shading_detection
from solar_forecasting.system_efficiency import calibrate_system_efficiency
from solar_forecasting.point_forecast_prediction import (
    prepare_training_data, 
    train_regression_tree, 
    forecast_point_generation )
from solar_forecasting.probabilistic_forecast import run_probabilistic_forecast_pipeline



if __name__ == "__main__":
    # Load config
    cfg = default_plant_config

    # --- Step 1: Download weather data ---
    weather_df = fetch_nasa_power_weather(
        latitude=cfg.latitude,
        longitude=cfg.longitude,
        start="20220101",
        end="20221231",
        parameters=("T2M", "RH2M", "WS2M", "WD2M", "CLOUD_AMT")
    )

    df = load_and_prepare_data("data/pv_data.csv")  # Customize path as needed
    TODO: # Ensure the data is prepared correctly, e.g., timestamps, units, etc.

    # --- Step 2: Compute clear-sky irradiance ---
    irradiance_df = calculate_clear_sky_irradiance(weather_df, cfg.model_dump())

    # --- Step 3: Calibrate system (kappa_c and kappa_t) ---
    calibrated_df, (kappa_c, kappa_t) = calibrate_system_efficiency(irradiance_df)

      # --- Step 3: Detect partial shading (optional) ---
    shading_df = partial_shading_detection(calibrated_df)
    TODO: # Ensure the partial shading detection is applied correctly, e.g., zenith angle, power columns, etc.

    # --- Step 5: Point Forecasting ---
    feature_cols = ["temperature", "cloudiness", "humidity"]
    X_train, y_train = prepare_training_data(calibrated_df, feature_cols, target_col="Pm", reference_col="Ppccs")
    model = train_regression_tree(X_train, y_train)
    point_forecast_df = forecast_point_generation(model, calibrated_df, feature_cols)

    # --- Step 6: Probabilistic Forecasting ---
    final_df = run_probabilistic_forecast_pipeline(point_forecast_df)

    # --- Step 7: Save output ---
    final_df.to_csv("data/forecast_output.csv", index=False)
    print("Forecast pipeline completed. Results saved to 'data/forecast_output.csv'")
