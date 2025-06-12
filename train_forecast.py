import joblib
import os
import datetime
import yaml
import argparse
from solar_forecasting.plant_config import default_plant_config
from solar_forecasting.utils import load_prepare_data
from solar_forecasting.download_weather_data import fetch_nasa_power_weather
from solar_forecasting.clear_sky_model import calculate_clear_sky_irradiance
from solar_forecasting.partial_shading_detection import partial_shading_detection
from solar_forecasting.system_efficiency import calibrate_system_efficiency
from solar_forecasting.point_forecast_prediction import (
    prepare_training_data, 
    train_regression_tree, 
    forecast_point_generation )
from solar_forecasting.probabilistic_forecast import run_probabilistic_forecast_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run solar forecasting pipeline.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to YAML config file")
    parser.add_argument("--start", type=str, default="20190101", help="Start date for weather data (YYYYMMDD)")
    parser.add_argument("--end", type=str, default="20191231", help="End date for weather data (YYYYMMDD)")
    parser.add_argument("--pv-data-path", type=str, default="data/raw/pv_data.csv", help="(Optional) Path to the input PV data CSV")
    # parser.add_argument("--weather-data-path", type=str, default="data/raw/weather_data.csv", help="(Optional) Path to the input weather data CSV")
    parser.add_argument("--output-path", type=str, default="data/processed/forecast_output.csv", help="(Optional) Path to save the forecast results")
    parser.add_argument("--model-path", type=str, default="models/point_forecast_model.pkl", help="(Optional) Path to save the trained model")
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()
    cfg_dict = load_config(args.config)

    # Override with CLI args if provided
    start_date = args.start or cfg_dict["weather_download"]["start"]
    end_date = args.end or cfg_dict["weather_download"]["end"]
    pv_data_path = args.pv_data_path or cfg_dict["input_paths"]["pv_data"]
    output_path = args.output_path or cfg_dict["output_path"]
    model_path = args.model_path or cfg_dict["model_path"]

    
    # Load config
    cfg = default_plant_config

    # --- Step 1: Download weather data ---
    weather_df = fetch_nasa_power_weather(
        latitude=cfg.latitude,
        longitude=cfg.longitude,
        start=start_date,
        end=end_date,
        parameters=tuple(cfg_dict["weather_download"]["parameters"])
    )

    data_df = load_prepare_data(pv_data_path, weather_df) 

    # --- Step 2: Compute clear-sky irradiance ---
    irradiance_df = calculate_clear_sky_irradiance(data_df, cfg.model_dump())

    # --- Step 3: Calibrate system (kappa_c and kappa_t) ---
    calibrated_df, (kappa_c, kappa_t) = calibrate_system_efficiency(
        irradiance_df,
        pm_col="pv_output_kw",
        pcs_col="Pcs_kW",
        cloud_col="cloudiness",
        temp_col="temperature_wx"
    )

    # --- Step 3: Detect partial shading (optional) ---
    shading_df = partial_shading_detection(
        calibrated_df,
        zenith_col="zenith", 
        power_col="pv_output_kw", 
        pccs_col="Pccs_kW", 
        output_col="Ppccs_kW"
    )

    # --- Step 5: Point Forecasting ---
    feature_cols = ["temperature_wx", "cloudiness", "humidity"]
    X_train, y_train = prepare_training_data(
        calibrated_df, 
        feature_cols, 
        target_col="pv_output_kw", 
        reference_col="Ppccs_kW")
    model = train_regression_tree(X_train, y_train)
    # Save the model for later use as a pkl file
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Generate timestamp-based version string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_versioned_path = f"models/forecast_model_{timestamp}.pkl"

    # Save the model
    joblib.dump(model, model_versioned_path)
    print(f"Model saved to {model_versioned_path}")

    point_forecast_df = forecast_point_generation(
        model, 
        calibrated_df, 
        feature_cols,
        reference_col="Ppccs_kW", 
        output_col="Ppf_kW"
    )

    # --- Step 6: Probabilistic Forecasting ---
    final_df = run_probabilistic_forecast_pipeline(
        point_forecast_df,
        forecast_col="Ppf_kW",
        actual_col="pv_output_kw",
        cloud_col="cloudiness",
        timestamp_col="timestamp"
        )

    # --- Step 7: Save output ---
    final_df.to_csv(output_path, index=False)
    print("Forecast pipeline completed. Results saved to 'data/forecast_output.csv'")
