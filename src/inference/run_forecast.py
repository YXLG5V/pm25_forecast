# run_forecast.py

from src.inference.service import ForecastService

config = {
    "model_path": "./models/model_hgb_pm25.pkl",
    "features_path": "./models/features.pkl",
    "categories_path": "./models/location_categories.pkl",
    "sensor_id": 36004,
    "location_name": "Gyor Szent Istvan",
    "lat": 47.6875,
    "lon": 17.6504,
    "horizon": 12,
    "lag_hours": 24
}

service = ForecastService(config)

forecast = service.run()

print(forecast)