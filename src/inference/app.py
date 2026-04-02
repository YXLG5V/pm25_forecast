# uvicorn src.inference.app:app --reload
# http://127.0.0.1:8000/docs
# 
from fastapi import FastAPI
from src.inference.service import ForecastService
from fastapi.responses import FileResponse
from pathlib import Path
from src._pollutant_client import STATIONS

app = FastAPI()

from pydantic import BaseModel


# class ForecastRequest(BaseModel):
#     location_name: str
#     horizon: int = 12


class ForecastRequest(BaseModel):
    location_name: str = "Gyor Szent Istvan"
    horizon: int = 12

BASE_CONFIG = {
    "model_path": "./models/model.pkl",
    "features_path": "./artifacts/features.pkl",
    "location_map": "./artifacts/location_mapping.pkl",
    "lag_hours": 48
}

service = ForecastService(BASE_CONFIG)

@app.get("/demo")
def ui():
    
    BASE_DIR = Path(__file__).resolve().parent

    return FileResponse(BASE_DIR / "templates" / "index.html")

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/forecast")
def forecast(req: ForecastRequest):

    service.config = {
        **service.config,
        "location_name": req.location_name,
        "horizon": req.horizon
    }
    
    result = service.get_forecast()

    station = STATIONS[req.location_name]

    return {
        "location": req.location_name,
        "lat": station["lat"],
        "lon": station["lon"],
        "history": result["history"].to_dict(orient="records"),
        "forecast": result["forecast"].to_dict(orient="records"),
        "explanations": result["explanations"]
    }