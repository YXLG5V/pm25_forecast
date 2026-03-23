from src._api_client import fetch_recent_pollutants
from src._weather_client import (
    fetch_weather_history,
    fetch_weather_forecast
)

from .artifacts import ModelArtifacts
from .model import PM25Model
from .pipeline import ForecastPipeline


class ForecastService:
    def run(self):
        return self.get_forecast()

    def __init__(self, config):

        self.config = config

        # --- artifacts ---
        artifacts = ModelArtifacts(
            model_path=config["model_path"],
            features_path=config["features_path"],
            categories_path=config["categories_path"]
        )

        # --- model ---
        self.model = PM25Model(artifacts)

        # --- pipeline ---
        self.pipeline = ForecastPipeline(self.model)

    def get_forecast(self):

        cfg = self.config

        pollutants = fetch_recent_pollutants(
            pm25_sensor_id=cfg["sensor_id"],
            hours=cfg["lag_hours"] + 3
        )

        pollutants["location"] = cfg["location_name"]
        pollutants["latitude"] = cfg["lat"]
        pollutants["longitude"] = cfg["lon"]

        weather_hist = fetch_weather_history(
            hours=cfg["lag_hours"] + 3
        )

        weather_fc = fetch_weather_forecast()

        history = self.pipeline.build_history(
            pollutants,
            weather_hist
        )

        forecast = self.pipeline.forecast(
            history,
            weather_fc,
            cfg["horizon"]
        )

        return forecast