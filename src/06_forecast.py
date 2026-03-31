# ============================================================
# 06_FORECAST.PY
# ============================================================

# Ez a script egy teljes ML forecasting pipeline:
# 1. adat letöltés
# 2. preprocessing
# 3. feature engineering
# 4. recursive prediction (autoregresszív módon)
# 5. mentés + plot

# A forecast autoregresszív módon működik:
# - minden lépésben az utolsó ismert állapotból indulunk
# - a modell predikciója visszakerül az inputba
# - így a jövőbeli értékek egymásra épülnek

# A pipeline minden iterációban újraszámolja a feature-öket,
# hogy a lag és rolling változók konzisztensen frissüljenek.

import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt

from _pollutant_client import fetch_station_pollutants      # pollutáns letöltés
from _weather_client import (
    fetch_weather_history,                           # múlt időjárás
    fetch_weather_forecast                           # jövő időjárás
)
from _preprocessing import (
    build_base_dataset,        # cleaning + merge
    interpolate_station
)
from _feature_engineering import build_features      # feature creation
from datetime import datetime

# Modell és feature lista
MODEL_PATH = "./models/model.pkl"
FEATURES_PATH = "./artifacts/features.pkl"

LOCATION_NAME = "Gyor Szent Istvan"
# Gyor 2 Ifjusag
# Gyor Ifjusag
# Gyor Szent Istvan
# VS

# Elérhető még, de tanításkor nem volt használva
# Gyor 1 Szent Istvan
# OMSZ LRK Mobil

LAT = 47.6875
LON = 17.6504

HORIZON = 12      # hány órát jósolunk előre
LAG_HOURS = 48    # mennyi múlt kell feature-ökhöz

DEBUG = True

def dbg(msg, level="DEBUG"):
    if DEBUG:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] [{level}] {msg}")

# Modell betöltése
dbg("Modell és feature lista betöltése", level="INIT")
model = joblib.load(MODEL_PATH)

# Feature lista
FEATURES = joblib.load(FEATURES_PATH)
dbg(f"Features count: {len(FEATURES)}",  level="INIT")

# Lokáció
mapping = joblib.load("./artifacts/location_mapping.pkl")
assert LOCATION_NAME in mapping, \
    f"Unknown location: {LOCATION_NAME}. Not in training mapping."



# múltbeli levegőminőség adatok
dbg("Pollution letöltés indul", level="DATA")
pollutants = fetch_station_pollutants(
    location_name=LOCATION_NAME,
    hours=48
)

# shape = (időpontok száma, változók)
dbg(f"Pollution shape: {pollutants.shape}", level = "DATA")
# tail → legfrissebb adatok ellenőrzése
dbg(f"Pollution head:\n{pollutants.tail(3)}", level = "DATA")

# datetime normalizálás (UTC)
pollutants["datetime"] = pd.to_datetime(pollutants["datetime"], utc=True)



# múlt időjárás → feature engineeringhez kell
dbg("Weather history letöltés", level = "DATA")
weather_hist = fetch_weather_history(LAT, LON, hours=LAG_HOURS + 3)
weather_hist = weather_hist.reset_index()

# ellenőrzés
dbg(f"Weather hist shape: {weather_hist.shape}", level = "DATA")

# jövő időjárás → forecast input
dbg("[DATA] Weather forecast letöltés")
weather_fc = fetch_weather_forecast(LAT, LON)

# látjuk, meddig van forecast adatunk
dbg(f"Weather forecast range: {weather_fc.index.min()} → {weather_fc.index.max()}", level = "DATA")

# ============================================================
# BASE DATASET
# ============================================================
# - merge pollution + weather
# - tisztítás
# - resample
# - interpoláció

dbg("Base dataset építés", level = "PIPELINE")

history = build_base_dataset(
    pollution=pollutants,
    weather=weather_hist
)

history = interpolate_station(history)

# ezt csak plothoz
history_real = history.copy()

# debug: shape + utolsó sorok
dbg(f"History shape: {history.shape}", level="PIPELINE")
dbg(f"Last rows:\n{history.tail(3)}", level = "PIPELINE")

# ============================================================
# REKURZÍV FORECAST
# ============================================================
# last known data
# +1 óra → future row
# weather hozzáadás
# feature build
# model.predict
# pm25 visszaírás history-ba
# következő lépés ezt használja

# → minden új predikció visszakerül inputként

current_time = history["datetime"].max()
predictions = []

dbg(f"Start time: {current_time}", level = "FORECAST")

for step in range(1, HORIZON + 1):

    dbg("\n" + "="*60)
    dbg(f"[STEP {step}]")

    # minden iteráció +1 óra
    future_time = current_time + timedelta(hours=step)
    dbg(f"Future time: {future_time}")

    # utolsó ismert sor másolása
    future = history.iloc[-1:].copy()

    # új időpont
    future["datetime"] = future_time

    dbg(f"Last known pm25: {future['pm25'].iloc[0]}")

    # jövő időjárás kiválasztása
    weather_slice = weather_fc.loc[:future_time]
    if weather_slice.empty:
        weather = weather_fc.iloc[-1]
    else:
        weather = weather_slice.iloc[-1]

    dbg(f"Weather values: T={weather.temperature}, H={weather.humidity}, W={weather.wind_speed}, P={weather.precipitation}")

    # beírjuk a jövő sorba
    future["temperature"] = weather.temperature
    future["humidity"] = weather.humidity
    future["wind_speed"] = weather.wind_speed
    future["precipitation"] = weather.precipitation

    # hozzáadjuk a jövő sort
    history = pd.concat([history, future], ignore_index=True)

    # rendezés
    history = history.sort_values(["location", "datetime"]).reset_index(drop=True)

    dbg(f"History length: {len(history)}")

    # újraszámoljuk az összes feature-t
    # → lag-ek miatt KELL
    df_features = build_features(history.copy(), fit=False)

    # csak az utolsó sor (aktuális future pont)
    X = df_features.iloc[[-1]][FEATURES]

    dbg(f"Feature vector:\n{X.iloc[0].to_dict()}")

    # ha NaN → probléma
    if X.isna().any().any():
        dbg("NaN a feature vectorban!", level = "WARN")

    #Ensemble?    
    if isinstance(model, dict):
    
        preds = []
        for m in model.values():
            p = np.maximum(0, np.expm1(m.predict(X)[0]))
            preds.append(p)
        
        pred = np.mean(preds)

    else:
        # modell log-space-ben tanult, visszaalakítás
        pred = np.maximum(0, np.expm1(model.predict(X)[0]))
    
    dbg(f"Pred value: {pred}")

    # sanity check (extrém értékek)
    if pred > 500:
        dbg("Extrém magas PM2.5!", level = "ALERT")

    
    # a predikció visszakerül a history-ba
    # → következő step ezt fogja használni lagként
    history.loc[
        history["datetime"] == future_time,
        "pm25"
    ] = pred

    #Eredmény tárolás
    predictions.append({
        "datetime": future_time,
        "location": LOCATION_NAME,
        "pm25_pred": pred
    })

# list → DataFrame
forecast_df = pd.DataFrame(predictions)

# lokális időzóna (vizualizációhoz)
forecast_df["datetime_local"] = (
    forecast_df["datetime"].dt.tz_convert("Europe/Budapest")
)

dbg(f"Forecast:\n{forecast_df}", level = "OUTPUT")

#mentés
forecast_df.to_csv("./data/latest_forecast.csv", index=False)
dbg("[OUTPUT] CSV mentve")

#plot
plt.figure()

# múlt (utolsó 12 óra)
recent_time = history_real["datetime"].tail(12).dt.tz_convert("Europe/Budapest")
recent_pm = history_real["pm25"].tail(12)

plt.plot(recent_time, recent_pm, marker="o", label="recent")

# forecast kezdőpont
forecast_start_time = history_real["datetime"].iloc[-1].tz_convert("Europe/Budapest")
forecast_start_pm = history_real["pm25"].iloc[-1]

start_point = pd.DataFrame({
    "datetime_local": [forecast_start_time],
    "pm25_pred": [forecast_start_pm]
})

# forecast görbe
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

# elválasztó vonal
plt.axvline(forecast_start_time, linestyle="--", color="gray")

plt.xlabel("Time")
plt.ylabel("PM2.5")
plt.title("Recursive 12h PM2.5 Forecast")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# model = joblib.load("./models/model.pkl")

