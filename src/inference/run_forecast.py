# run_forecast.py

from src.inference.service import ForecastService

config = {
    "model_path": "./models/model.pkl",
    "features_path": "./artifacts/features.pkl",
    "location_map": "./artifacts/location_mapping.pkl",
    "location_name": "Gyor Szent Istvan",
    "lat": 47.6875,
    "lon": 17.6504,
    "horizon": 12,
    "lag_hours": 48
}

service = ForecastService(config)

forecast = service.run()

print(forecast)