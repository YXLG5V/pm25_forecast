# ============================================================
# 07_WALKFORWARD_VALIDATION.PY
# ============================================================

import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt

from _pollutant_client import fetch_station_pollutants
from _weather_client import (
    fetch_weather_history,
    fetch_weather_forecast
)
from _preprocessing import (
    build_base_dataset,
    interpolate_station
)
from _feature_engineering import build_features


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "./models/model.pkl"
FEATURES_PATH = "./artifacts/features.pkl"
LOCATION_MAPPING_PATH = "./artifacts/location_mapping.pkl"

LOCATION_NAME = "Gyor Szent Istvan"
LAT = 47.6875
LON = 17.6504

HORIZON = 12
LAG_HOURS = 48
VALIDATION_HOURS = 120


# ============================================================
# LOAD
# ============================================================

model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURES_PATH)
mapping = joblib.load(LOCATION_MAPPING_PATH)

assert LOCATION_NAME in mapping


# ============================================================
# DATA (IDENTICAL LOGIC AS 06)
# ============================================================

pollutants = fetch_station_pollutants(
    location_name=LOCATION_NAME,
    hours=VALIDATION_HOURS + LAG_HOURS + HORIZON
)

pollutants["datetime"] = pd.to_datetime(
    pollutants["datetime"], utc=True
)

weather_hist = fetch_weather_history(
    LAT, LON,
    hours=VALIDATION_HOURS + LAG_HOURS + 3
)

weather_hist = weather_hist.reset_index()

weather_fc = fetch_weather_forecast(LAT, LON)


# ============================================================
# BASE DATASET
# ============================================================

history_full = build_base_dataset(
    pollution=pollutants,
    weather=weather_hist
).sort_values("datetime")


# ============================================================
# WALKFORWARD
# ============================================================

results = []

start_idx = LAG_HOURS
end_idx = len(history_full) - HORIZON

for i in tqdm(range(start_idx, end_idx)):

    history = history_full.iloc[:i].copy()
    current_time = history["datetime"].max()

    preds = []

    # ========================================================
    # RECURSIVE FORECAST (IDENTICAL TO 06_FORECAST)
    # ========================================================

    for step in range(1, HORIZON + 1):

        future_time = current_time + timedelta(hours=step)

        # --- last row copy ---
        future = history.iloc[-1:].copy()
        future["datetime"] = future_time

        # --- WEATHER (STRICT MATCH) ---
        weather_slice = weather_fc.loc[:future_time]

        if weather_slice.empty:
            weather = weather_fc.iloc[-1]
        else:
            weather = weather_slice.iloc[-1]

        future["temperature"] = weather.temperature
        future["humidity"] = weather.humidity
        future["wind_speed"] = weather.wind_speed
        future["precipitation"] = weather.precipitation

        # --- append ---
        history = pd.concat([history, future], ignore_index=True)
        history = interpolate_station(history)
        history = history.sort_values(
            ["location", "datetime"]
        ).reset_index(drop=True)

        # --- FEATURE ENGINEERING ---
        df_features = build_features(history.copy(), fit=False)

        # --- STRICT SAME ROW SELECTION ---
        X = df_features.iloc[[-1]][FEATURES]

        # --- PREDICTION ---

        if isinstance(model, dict):

            ensemble_preds = []
            for m in model.values():
                p = np.maximum(0, np.expm1(m.predict(X)[0]))
                ensemble_preds.append(p)
            
            pred = np.mean(ensemble_preds)

        else:
            
            pred = np.maximum(0, np.expm1(model.predict(X)[0]))


        # --- write back ---
        history.loc[
            history["datetime"] == future_time,
            "pm25"
        ] = pred

        preds.append(pred)

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    future_real = history_full.iloc[i:i+HORIZON]["pm25"].values

    for h in range(HORIZON):
        results.append({
            "forecast_origin": current_time,
            "horizon": h + 1,
            "y_true": future_real[h],
            "y_pred": preds[h]
        })


# ============================================================
# EVALUATION
# ============================================================

df_res = pd.DataFrame(results)


def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))


def mae(y, yhat):
    return np.mean(np.abs(y - yhat))


summary = df_res.groupby("horizon").apply(
    lambda g: pd.Series({
        "RMSE": rmse(g.y_true, g.y_pred),
        "MAE": mae(g.y_true, g.y_pred)
    })
)

print("\n=== METRICS BY HORIZON ===")
print(summary)


# ============================================================
# PLOTS
# ============================================================

summary.plot(marker="o")
plt.title("Error vs Forecast Horizon")
plt.grid()
plt.show()


plt.figure()
plt.scatter(df_res["y_true"], df_res["y_pred"], alpha=0.3)

plt.plot(
    [df_res.y_true.min(), df_res.y_true.max()],
    [df_res.y_true.min(), df_res.y_true.max()],
)

plt.title("Predicted vs Actual")
plt.show()


df_res["error"] = df_res["y_pred"] - df_res["y_true"]

df_res.groupby("forecast_origin")["error"].mean().plot()
plt.title("Mean Error Over Time")
plt.grid()
plt.show()