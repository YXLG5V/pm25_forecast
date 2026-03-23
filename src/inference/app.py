# uvicorn src.inference.app:app --reload
# http://127.0.0.1:8000/forecast
# http://127.0.0.1:8000/docs
# 
from fastapi import FastAPI
from src.inference.service import ForecastService

app = FastAPI()

from pydantic import BaseModel


# class ForecastRequest(BaseModel):
#     location_name: str
#     sensor_id: int
#     lat: float
#     lon: float
#     horizon: int = 12


class ForecastRequest(BaseModel):
    location_name: str = "Gyor Szent Istvan"
    sensor_id: int = 36004
    lat: float = 47.6875
    lon: float = 17.6504
    horizon: int = 12

BASE_CONFIG = {
    "model_path": "./models/model.pkl",
    "features_path": "./artifacts/features.pkl",
    "categories_path": "./artifacts/location_mapping.pkl",
    "lag_hours": 48
}


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/forecast")
def forecast(req: ForecastRequest):

    config = {
        **BASE_CONFIG,
        "location_name": req.location_name,
        "sensor_id": req.sensor_id,
        "lat": req.lat,
        "lon": req.lon,
        "horizon": req.horizon
    }

    service = ForecastService(config)

    df = service.get_forecast()

    return {
        "location": req.location_name,
        "forecast": df.to_dict(orient="records")
    }