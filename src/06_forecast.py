# ============================================================
# 06_FORECAST.PY
# ============================================================

import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt

from _api_client import fetch_recent_pollutants
from _weather_client import (
    fetch_weather_history,
    fetch_weather_forecast
)

from _preprocessing import build_base_dataset
from _feature_engineering import build_features


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "./models/model.pkl"
FEATURES_PATH = "./artifacts/features.pkl"

LAT = 47.6875
LON = 17.6504

LOCATION_NAME = "Gyor Szent Istvan"
SENSOR_ID = 36004

HORIZON = 12
LAG_HOURS = 48


# ============================================================
# LOAD MODEL + ARTIFACTS
# ============================================================

model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURES_PATH)


# ============================================================
# DOWNLOAD DATA
# ============================================================

pollutants = fetch_recent_pollutants(
    location_name=LOCATION_NAME,
    pm25_sensor_id=SENSOR_ID,
    hours=LAG_HOURS + 3
)

# --- FIX LOCATION + COORDS ---
pollutants["location"] = LOCATION_NAME
pollutants["latitude"] = LAT
pollutants["longitude"] = LON

# --- force UTC ---
pollutants = pollutants.reset_index() 
pollutants["datetime"] = pd.to_datetime(pollutants["datetime"], utc=True)

weather_hist = fetch_weather_history(hours=LAG_HOURS + 3)
weather_hist = weather_hist.reset_index() 
weather_hist["datetime"] = pd.to_datetime(weather_hist["datetime"], utc=True)

weather_fc = fetch_weather_forecast()
weather_fc.index = pd.to_datetime(weather_fc.index, utc=True)


# ============================================================
# BUILD BASE DATASET
# ============================================================

history = build_base_dataset(
    pollution=pollutants,
    weather=weather_hist
)

history_real = history.copy()


# ============================================================
# RECURSIVE FORECAST
# ============================================================

current_time = history["datetime"].max()
predictions = []

for step in range(1, HORIZON + 1):

    future_time = current_time + timedelta(hours=step)

    future = history.iloc[-1:].copy()
    future["datetime"] = future_time

    if pd.isna(future["pm25"].iloc[0]):
        future["pm25"] = history["pm25"].ffill().iloc[-1]

    # =========================================================
    # WEATHER
    # =========================================================
    idx = weather_fc.index.get_indexer([future_time], method="nearest")[0]
    weather = weather_fc.iloc[idx]

    future["temperature"] = weather.temperature
    future["humidity"] = weather.humidity
    future["wind_speed"] = weather.wind_speed
    future["precipitation"] = weather.precipitation

    future["location"] = LOCATION_NAME


    # =========================================================
    # APPEND HISTORY
    # =========================================================
    history = pd.concat([history, future], ignore_index=True)
    history = history.sort_values(["location", "datetime"]).reset_index(drop=True)


    # =========================================================
    # FEATURE BUILD
    # =========================================================
    df_feat_input = history.copy()

    df_features = build_features(df_feat_input, fit=False)

    X = df_features.iloc[[-1]][FEATURES]


    # =========================================================
    # PREDICT
    # =========================================================
    pred_log = model.predict(X)[0]
    pred = np.maximum(0, np.expm1(pred_log))

    # =========================================================
    # WRITE BACK (recursive)
    # =========================================================
    history.loc[
        history["datetime"] == future_time,
        "pm25"
    ] = pred

    predictions.append({
        "datetime": future_time,
        "location": LOCATION_NAME,
        "pm25_pred": pred
    })


# ============================================================
# SAVE
# ============================================================

forecast_df = pd.DataFrame(predictions)

forecast_df["datetime_local"] = (
    forecast_df["datetime"]
    .dt.tz_convert("Europe/Budapest")
)

forecast_df.to_csv("./data/latest_forecast.csv", index=False)


# ============================================================
# PLOT
# ============================================================

plt.figure()

recent_time = history_real["datetime"].tail(12).dt.tz_convert("Europe/Budapest")
recent_pm = history_real["pm25"].tail(12)

plt.plot(recent_time, recent_pm, marker="o", label="recent")

forecast_start_time = history_real["datetime"].iloc[-1].tz_convert("Europe/Budapest")
forecast_start_pm = history_real["pm25"].iloc[-1]

start_point = pd.DataFrame({
    "datetime_local": [forecast_start_time],
    "pm25_pred": [forecast_start_pm]
})

plot_df = pd.concat([
    start_point,
    forecast_df[["datetime_local", "pm25_pred"]]
])

plt.plot(
    plot_df["datetime_local"],
    plot_df["pm25_pred"],
    marker="o",
    label="forecast"
)

plt.axvline(forecast_start_time, linestyle="--", color="gray")

plt.xlabel("Time")
plt.ylabel("PM2.5")
plt.title("Recursive 12h PM2.5 Forecast")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()