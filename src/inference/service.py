from src._pollutant_client import fetch_station_pollutants
from src._weather_client import (
    fetch_weather_history,
    fetch_weather_forecast
)

from .artifacts import ModelArtifacts
from .model import PM25Model
from .pipeline import ForecastPipeline

import pandas as pd
import joblib
import shap


# =========================================================
# GLOBAL SHAP CACHE
# =========================================================

SHAP_MODEL_PATH = "./models/lgbm.pkl"

SHAP_PIPELINE = joblib.load(SHAP_MODEL_PATH)

if hasattr(SHAP_PIPELINE, "named_steps"):
    SHAP_MODEL = SHAP_PIPELINE.named_steps["model"]
else:
    SHAP_MODEL = SHAP_PIPELINE

GLOBAL_EXPLAINER = shap.TreeExplainer(SHAP_MODEL)


# =========================================================
# SHARED DATA PREPARATION
# =========================================================

def prepare_shared_data(
    location_name,
    lag_hours,
    weather_lat,
    weather_lon,
    pipeline
):

    pollutants = fetch_station_pollutants(
        location_name=location_name,
        hours=lag_hours
    )

    weather_hist = fetch_weather_history(
        weather_lat,
        weather_lon,
        hours=lag_hours + 3
    )

    weather_hist = weather_hist.reset_index()

    weather_fc = fetch_weather_forecast(
        weather_lat,
        weather_lon
    )

    # ===== BUILD HISTORY =====

    history = pipeline.build_history(
        pollutants,
        weather_hist
    )

    # ===== WEATHER CODE MERGE =====

    weather_hist["datetime"] = pd.to_datetime(
        weather_hist["datetime"],
        utc=True
    )

    history["datetime"] = pd.to_datetime(
        history["datetime"],
        utc=True
    )

    history = history.merge(
        weather_hist[["datetime", "weather_code"]],
        on="datetime",
        how="left"
    )

    # ===== TIMEZONE =====

    history["datetime"] = (
        pd.to_datetime(history["datetime"])
        .dt.tz_convert("Europe/Budapest")
    )

    weather_fc.index = (
        pd.to_datetime(weather_fc.index)
        .tz_convert("Europe/Budapest")
    )

    # ===== CLEAN =====

    history = history.dropna(subset=["pm25"])

    history_tail = (
        history
        .sort_values("datetime")
        .tail(12)[["datetime", "pm25", "weather_code"]]
    )

    return {
        "history": history,
        "history_tail": history_tail,
        "weather_fc": weather_fc
    }


# =========================================================
# FORECAST SERVICE
# =========================================================

class ForecastService:

    WEATHER_LAT = 47.6875
    WEATHER_LON = 17.6504

    def __init__(self, config):

        self.config = config

        # ===== LOAD ARTIFACTS =====

        artifacts = ModelArtifacts(
            model_path=config["model_path"],
            features_path=config["features_path"],
            location_map=config["location_map"]
        )

        # ===== MODEL =====

        self.model = PM25Model(artifacts)

        # ===== PIPELINE =====

        self.pipeline = ForecastPipeline(self.model)

        # ===== SHAP =====

        self.explainer = GLOBAL_EXPLAINER
        self.pipeline.explainer = self.explainer

    def run(self):
        return self.get_forecast()

    # =====================================================
    # OLD API COMPATIBLE METHOD
    # =====================================================

    def get_forecast(self):

        cfg = self.config

        assert cfg["location_name"] in self.model.categories, \
            f"Unknown location: {cfg['location_name']}"

        shared = prepare_shared_data(
            location_name=cfg["location_name"],
            lag_hours=cfg["lag_hours"],
            weather_lat=self.WEATHER_LAT,
            weather_lon=self.WEATHER_LON,
            pipeline=self.pipeline
        )

        return self.predict_from_prepared(
            history=shared["history"],
            history_tail=shared["history_tail"],
            weather_fc=shared["weather_fc"],
            horizon=cfg["horizon"]
        )

    # =====================================================
    # SHARED PIPELINE PREDICTION
    # =====================================================

    def predict_from_prepared(
        self,
        history,
        history_tail,
        weather_fc,
        horizon
    ):

        # IMPORTANT:
        # recursive forecasting modifies history
        # therefore local copy is required

        history = history.copy()

        forecast_df, window = self.pipeline.forecast(
            history,
            weather_fc,
            horizon
        )

        # ===== WEATHER CODE MERGE =====

        forecast_df = forecast_df.merge(
            weather_fc[["weather_code"]],
            left_on="datetime",
            right_index=True,
            how="left"
        )

        # =================================================
        # WEATHER-AWARE RECOMMENDATION
        # =================================================

        GOOD_WEATHER = {0, 1, 2, 3}
        PM_THRESHOLD = 15

        recommendation_text = None
        representative_code = None

        if window is not None and not forecast_df.empty:

            start = (
                pd.to_datetime(window["start"], utc=True)
                .tz_convert("Europe/Budapest")
            )

            end = (
                pd.to_datetime(window["end"], utc=True)
                .tz_convert("Europe/Budapest")
            )

            window_df = forecast_df[
                (forecast_df["datetime"] >= start) &
                (forecast_df["datetime"] <= end)
            ]

            if not window_df.empty:

                avg_pm = window_df["pm25_pred"].mean()

                valid_weather = (
                    window_df["weather_code"]
                    .dropna()
                )

                # ===== REPRESENTATIVE WEATHER =====

                mid_idx = len(window_df) // 2
                mid_row = window_df.iloc[mid_idx]

                if pd.notna(mid_row["weather_code"]):

                    representative_code = int(
                        mid_row["weather_code"]
                    )

                else:

                    valid_codes = (
                        window_df["weather_code"]
                        .dropna()
                    )

                    if not valid_codes.empty:
                        representative_code = int(
                            valid_codes.iloc[0]
                        )

                # ===== RULES =====

                pm_ok = avg_pm < PM_THRESHOLD

                weather_ok = (
                    valid_weather.isin(GOOD_WEATHER).all()
                )

                if pm_ok and weather_ok:

                    recommendation_text = (
                        "🌿 Best time for outdoor activities and ventilation"
                    )

                elif pm_ok:

                    recommendation_text = (
                        "🪟 Best time for ventilation"
                    )

                elif weather_ok:

                    recommendation_text = (
                        "⚠️ Air quality is poor – avoid outdoor activities"
                    )

                else:

                    recommendation_text = (
                        "🚫 Poor air quality – keep windows closed"
                    )

            window = dict(window)

            window["recommendation_text"] = recommendation_text
            window["weather_code"] = representative_code

        # =================================================
        # RESPONSE
        # =================================================

        return {
            "history": history_tail,
            "forecast": forecast_df,
            "recommended_window": window,
            "explanations": forecast_df[
                ["datetime", "effects"]
            ].to_dict(orient="records"),
        }