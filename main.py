from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel
import pandas as pd
import joblib
import io
import json

from solar_forecasting.download_weather_data import fetch_nasa_power_weather
from solar_forecasting.plant_config import default_plant_config
from solar_forecasting.utils import load_config

app = FastAPI()

FEATURES = ["temperature_wx", "cloudiness", "humidity"]


class ManualInput(BaseModel):
    temperature: float
    cloudiness: float  # expected to be between 0 and 1
    humidity: float
    model_path: str


class NASAInput(BaseModel):
    timestamp: str  # Format: YYYYMMDD
    latitude: float = default_plant_config.latitude
    longitude: float = default_plant_config.longitude
    model_path: str


@app.post("/predict/manual")
def predict_manual(data: ManualInput):
    model = joblib.load(data.model_path)
    X = pd.DataFrame([{
        "temperature_wx": data.temperature,
        "cloudiness": data.cloudiness,
        "humidity": data.humidity
    }])
    prediction = model.predict(X)[0]
    return {"predicted_power": prediction}


@app.post("/predict/nasa")
def predict_nasa(data: NASAInput):
    df = fetch_nasa_power_weather(
        latitude=data.latitude,
        longitude=data.longitude,
        start=data.timestamp,
        end=data.timestamp,
        parameters=("T2M", "RH2M", "CLOUD_AMT")
    )
    # Ensure the columns are renamed as expected
    df = df[FEATURES]
    model = joblib.load(data.model_path)
    prediction = model.predict(df)[0]
    return {"predicted_power": prediction}


@app.post("/predict/file")
async def predict_file(
    model_path: str = Form(...),
    file: UploadFile = File(...)
):
    model = joblib.load(model_path)
    contents = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(contents.decode()))
        elif filename.endswith(".json"):
            df = pd.read_json(io.StringIO(contents.decode()))
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

    if not all(col in df.columns for col in FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features. Expected columns: {FEATURES}"
        )

    X = df[FEATURES]
    predictions = model.predict(X)
    return {"predictions": predictions.tolist()}