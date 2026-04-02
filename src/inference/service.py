from src._pollutant_client import fetch_station_pollutants
from src._weather_client import (
    fetch_weather_history,
    fetch_weather_forecast
)

from .artifacts import ModelArtifacts
from .model import PM25Model
from .pipeline import ForecastPipeline


class ForecastService:

    WEATHER_LAT = 47.6875
    WEATHER_LON = 17.6504

    def run(self):
        return self.get_forecast()
    
    def __init__(self, config):

        self.config = config

        # --- artifacts ---
        artifacts = ModelArtifacts(
            model_path=config["model_path"],
            features_path=config["features_path"],
            location_map=config["location_map"]
        )

        import shap

        self.model = artifacts

        # 1. pipeline
        self.pipeline = ForecastPipeline(self.model)

        # 2. modell kibontása pipeline-ból
        model = self.model.model
        if hasattr(model, "named_steps"):
            model = model.named_steps[list(model.named_steps.keys())[-1]]

        # 3. explainer létrehozása
        self.explainer = shap.TreeExplainer(model)

        # 4. átadás pipeline-nak
        self.pipeline.explainer = self.explainer

    def get_forecast(self):

        cfg = self.config

        assert cfg["location_name"] in self.model.categories, \
            f"Unknown location: {cfg['location_name']}"

        pollutants = fetch_station_pollutants(
            location_name=cfg["location_name"],
            hours=cfg["lag_hours"]
        )

        weather_hist = fetch_weather_history(
            self.WEATHER_LAT,
            self.WEATHER_LON,
            hours=cfg["lag_hours"] + 3
        )

        weather_hist = weather_hist.reset_index()

        weather_fc = fetch_weather_forecast(
            self.WEATHER_LAT,
            self.WEATHER_LON
        )
        
        history = self.pipeline.build_history(
            pollutants,
            weather_hist
        )


        history = history.dropna(subset=["pm25"])


        history_tail = (
            history.sort_values("datetime")
                .tail(12)[["datetime", "pm25"]]
        )

        forecast = self.pipeline.forecast(
            history,
            weather_fc,
            cfg["horizon"]
        )

        return {
            "history": history_tail,
            "forecast": forecast,
            "explanations": forecast[["datetime", "effects"]].to_dict(orient="records")
        }