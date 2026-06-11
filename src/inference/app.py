# uvicorn src.inference.app:app --reload
# http://127.0.0.1:8000/docs

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel

from src.inference.service import (
    ForecastService,
    prepare_shared_data
)

from src._pollutant_client import STATIONS

app = FastAPI()


# =========================================================
# REQUEST MODEL
# =========================================================

class ForecastRequest(BaseModel):
    location_name: str = "Győr - Ifjúság krt."
    horizon: int = 12


# =========================================================
# CONFIG
# =========================================================

BASE_CONFIG = {
    "features_path": "./artifacts/features.pkl",
    "location_map": "./artifacts/location_mapping.pkl",
    "lag_hours": 48
}

MODEL_REGISTRY = {
    "best": "./models/model.pkl",
    "lgbm": "./models/lgbm.pkl",
    "neuralnet": "./models/neuralnet.pkl",
}


# =========================================================
# LOAD SERVICES
# =========================================================

SERVICES = {}

for model_name, model_path in MODEL_REGISTRY.items():

    config = {
        **BASE_CONFIG,
        "model_path": model_path
    }

    SERVICES[model_name] = ForecastService(config)

print("All models loaded.")


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/demo")
def ui():

    BASE_DIR = Path(__file__).resolve().parent

    return FileResponse(
        BASE_DIR / "templates" / "index.html"
    )


# =========================================================
# FORECAST
# =========================================================

@app.post("/forecast")
def forecast(
    req: ForecastRequest,
    model: str = "best"
):

    if model not in MODEL_REGISTRY:
        model = "best"

    service = SERVICES[model]

    # =====================================================
    # SHARED DATA FETCH
    # =====================================================

    shared = prepare_shared_data(
        location_name=req.location_name,
        lag_hours=service.config["lag_hours"],
        weather_lat=service.WEATHER_LAT,
        weather_lon=service.WEATHER_LON,
        pipeline=service.pipeline
    )

    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    result = service.predict_from_prepared(
        history=shared["history"],
        history_tail=shared["history_tail"],
        weather_fc=shared["weather_fc"],
        horizon=req.horizon
    )

    # =====================================================
    # STATION METADATA
    # =====================================================

    station = STATIONS[req.location_name]

    return {
        "location": req.location_name,
        "model": model,

        "lat": station["lat"],
        "lon": station["lon"],

        "history": result["history"].to_dict(
            orient="records"
        ),

        "forecast": result["forecast"].to_dict(
            orient="records"
        ),

        "recommended_window": result[
            "recommended_window"
        ],

        "explanations": result["explanations"]
    }


@app.post("/forecast_all")
def forecast_all(req: ForecastRequest):

    base_service = SERVICES["best"]

    shared = prepare_shared_data(
        location_name=req.location_name,
        lag_hours=BASE_CONFIG["lag_hours"],
        weather_lat=base_service.WEATHER_LAT,
        weather_lon=base_service.WEATHER_LON,
        pipeline=base_service.pipeline
    )

    station = STATIONS[req.location_name]

    results = {}

    for model_name, service in SERVICES.items():

        result = service.predict_from_prepared(
            history=shared["history"],
            history_tail=shared["history_tail"],
            weather_fc=shared["weather_fc"],
            horizon=req.horizon
        )

        results[model_name] = {
            "forecast": result["forecast"].to_dict(
                orient="records"
            )
        }

        if model_name == "best":

            results[model_name]["history"] = (
                result["history"]
                .to_dict(orient="records")
            )

            results[model_name]["recommended_window"] = (
                result["recommended_window"]
            )

            results[model_name]["explanations"] = (
                result["explanations"]
            )

    return {
        "location": req.location_name,
        "lat": station["lat"],
        "lon": station["lon"],
        "models": results
    }