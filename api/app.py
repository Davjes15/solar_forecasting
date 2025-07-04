from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import yaml
from solar_forecasting.download_weather_data import fetch_nasa_power_weather
from solar_forecasting.utils import load_prepare_data
from solar_forecasting.plant_config import default_plant_config
from pathlib import Path
import os

# Load config at startup
CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/default_config.yaml")
MODEL_PATH = os.getenv("MODEL_PATH", "models/point_forecast_model.pkl")

try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load config: {e}")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

cfg = default_plant_config
feature_cols = config["features"]

# Define request schema
class PVInputRow(BaseModel):
    time: str  # ISO format datetime string
    pv_output_kw: float  # Optional, not used for prediction but expected in data file

class PredictionRequest(BaseModel):
    data: list[PVInputRow]

app = FastAPI()

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        df_pv = pd.DataFrame([row.dict() for row in request.data])
        df_pv["time"] = pd.to_datetime(df_pv["time"])

        start = df_pv["time"].min().strftime("%Y%m%d")
        end = df_pv["time"].max().strftime("%Y%m%d")

        weather_df = fetch_nasa_power_weather(
            latitude=cfg.latitude,
            longitude=cfg.longitude,
            start=start,
            end=end,
            parameters=("T2M", "RH2M", "WS2M", "WD2M", "PS", "CLOUD_AMT")
        )

        # Merge and prepare
        df_pv.to_csv("/tmp/temp_input.csv", index=False)
        merged_df = load_prepare_data("/tmp/temp_input.csv", df_w=weather_df)

        X = merged_df[feature_cols]
        predictions = model.predict(X)

        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))