import argparse
import pandas as pd
import joblib
import yaml
from solar_forecasting.download_weather_data import fetch_nasa_power_weather
from solar_forecasting.utils import load_prepare_data
from solar_forecasting.plant_config import default_plant_config


def parse_args():
    parser = argparse.ArgumentParser(description="Predict solar power using trained model")
    parser.add_argument("--input", type=str, required=True, help="Path to new raw PV data CSV")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model .pkl")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to YAML config file")
    parser.add_argument("--output", type=str, default="data/predictions.csv", help="Where to save predictions")
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    cfg = default_plant_config

    # Load new PV data
    new_pv_df = pd.read_csv(args.input, parse_dates=["time"])

    # Download corresponding weather data
    start = new_pv_df["time"].min().strftime("%Y%m%d")
    end = new_pv_df["time"].max().strftime("%Y%m%d")

    weather_df = fetch_nasa_power_weather(
        latitude=cfg.latitude,
        longitude=cfg.longitude,
        start=start,
        end=end,
        parameters=("T2M", "RH2M", "WS2M", "WD2M", "PS", "CLOUD_AMT")
    )

    # Merge PV data with weather features
    merged_df = load_prepare_data(file_path=args.input, df_w=weather_df)

    # Load model
    model = joblib.load(args.model)

    # Select feature columns used in training
    feature_cols = config["features"]  # e.g., ["temperature", "cloudiness", "humidity"]
    X_new = merged_df[feature_cols]

    # Predict
    merged_df["predicted_power"] = model.predict(X_new)

    # Save output
    merged_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
