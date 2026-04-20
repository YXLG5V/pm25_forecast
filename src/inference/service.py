from src._pollutant_client import fetch_station_pollutants
from src._weather_client import (
    fetch_weather_history,
    fetch_weather_forecast
)

from .artifacts import ModelArtifacts
from .model import PM25Model
from .pipeline import ForecastPipeline
import pandas as pd


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

        weather_hist["datetime"] = pd.to_datetime(weather_hist["datetime"], utc=True)
        history["datetime"] = pd.to_datetime(history["datetime"], utc=True)

        history = history.merge(
            weather_hist[["datetime", "weather_code"]],
            on="datetime",
            how="left"
        )

        history["datetime"] = (
            pd.to_datetime(history["datetime"])
            .dt.tz_convert("Europe/Budapest")
        )

        weather_fc.index = (
            pd.to_datetime(weather_fc.index)
            .tz_convert("Europe/Budapest")
        )

        history = history.dropna(subset=["pm25"])


        history_tail = (
            history.sort_values("datetime")
                .tail(12)[["datetime", "pm25", "weather_code"]]
        )

        forecast_df, window = self.pipeline.forecast(
            history,
            weather_fc,
            cfg["horizon"]
        )

        forecast_df = forecast_df.merge(
            weather_fc[["weather_code"]],
            left_on="datetime",
            right_index=True,
            how="left"
        )

       # ===== WEATHER-AWARE RECOMMENDATION =====

        GOOD_WEATHER = {0, 1, 2, 3}
        PM_THRESHOLD = 15

        recommendation_text = None
        representative_code = None

        if window is not None and not forecast_df.empty:

            start = pd.to_datetime(window["start"], utc=True).tz_convert("Europe/Budapest")
            end = pd.to_datetime(window["end"], utc=True).tz_convert("Europe/Budapest")

            window_df = forecast_df[
                (forecast_df["datetime"] >= start) &
                (forecast_df["datetime"] <= end)
            ]

            if not window_df.empty:

                avg_pm = window_df["pm25_pred"].mean()

                # ===== weather_ratio =====
                valid_weather = window_df["weather_code"].dropna()

                if not valid_weather.empty:
                    weather_ratio = valid_weather.isin(GOOD_WEATHER).mean()
                else:
                    weather_ratio = 0

                # ===== reprezentatív weather_code (idő szerinti közép) =====
                mid_idx = len(window_df) // 2
                mid_row = window_df.iloc[mid_idx]

                if pd.notna(mid_row["weather_code"]):
                    representative_code = int(mid_row["weather_code"])
                else:
                    # fallback: első valid érték
                    valid_codes = window_df["weather_code"].dropna()
                    if not valid_codes.empty:
                        representative_code = int(valid_codes.iloc[0])
                    else:
                        representative_code = None

                pm_ok = avg_pm < PM_THRESHOLD
                weather_ok = weather_ratio >= 0.5

                if pm_ok and weather_ok:
                    recommendation_text = "🌿 Best time for outdoor activities and ventilation"

                elif pm_ok:
                    recommendation_text = "🪟 Best time for ventilation"

                elif weather_ok:
                    recommendation_text = "⚠️ Air quality is poor – avoid outdoor activities"

                else:
                    recommendation_text = "🚫 Poor air quality – keep windows closed"

            window = dict(window)
            window["recommendation_text"] = recommendation_text
            window["weather_code"] = representative_code
            

        return {
            "history": history_tail,
            "forecast": forecast_df,
            "recommended_window": window,
            "explanations": forecast_df[["datetime", "effects"]].to_dict(orient="records"),
        }