# ============================================================
# 03_PREPROCESS.PY
# Build ML dataset from raw pollutants + weather
# ============================================================

import pandas as pd
import joblib

from _preprocessing import (
    build_base_dataset
)

from _feature_engineering import (
    build_features
)

# ============================================================
# LOAD RAW DATA
# ============================================================

print("Loading raw datasets...")

pollution = pd.read_csv("./data/raw/pollutants.csv", parse_dates=["datetime"])
weather = pd.read_csv("./data/raw/weather.csv")

print("Pollutants shape:", pollution.shape)
print("Weather shape:", weather.shape)

# ============================================================
# BUILD BASE DATASET
# ============================================================

print("\nBuilding base dataset...")

df = build_base_dataset(
    pollution=pollution,
    weather=weather
)

print("Base dataset shape:", df.shape)
df.to_csv("./data/preprocessed/eda_preprocessed.csv", index=False)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nBuilding features...")

df = build_features(df, fit=True)

print("Dataset with features:", df.shape)


FEATURES = [

    # ------------------------------------------------
    # AIR POLLUTION
    # ------------------------------------------------
    "pm10",
    "no2",
    "so2",

    # ------------------------------------------------
    # PM MEMORY
    # ------------------------------------------------
    "pm25_lag1",
    "pm25_lag3",
    "pm25_lag6",
    "pm25_lag24",
    "pm25_roll6",
    "pm25_roll24",
    "pm25_trend_3h",
    "pm25_std_12h",

    # ------------------------------------------------
    # TIME
    # ------------------------------------------------
    "hour",
    "hour_sin",
    "hour_cos",
    "month",
    "month_sin",
    "month_cos",
    "weekend_flag",
    "heating_season_flag",

    # ------------------------------------------------
    # WEATHER CURRENT
    # ------------------------------------------------
    "temperature",
    "humidity",
    "wind_speed",
    "precipitation",

    # ------------------------------------------------
    # WEATHER DYNAMICS
    # ------------------------------------------------
    "temp_change_3h",
    "humidity_change_3h",
    "wind_change_3h",

    # ------------------------------------------------
    # PHYSICAL INTERACTIONS
    # ------------------------------------------------
    "stagnation_index",
    "temp_wind_interaction",
    "mixing_index",
    "ventilation_index",
    "stagnation_hours_6h",

    # ------------------------------------------------
    # SPATIAL
    # ------------------------------------------------
    "location_id",
    "lat_norm",
    "lon_norm"
]

# ============================================================
# SAVE DATASET
# ============================================================
df.to_parquet("./data/preprocessed/preprocessed_with_FE.parquet")
print("\nDataset saved.")

print("Dataset shape:", df.shape)
print("Time range:", df.index.min(), "→", df.index.max())

# ============================================================
# SAVE ARTIFACTS
# ============================================================
joblib.dump(FEATURES,"./artifacts/features.pkl")
print("\nSaved feature list.")
