# ============================================================
# PREPROCESSING PIPELINE
## ============================================================

import pandas as pd
import numpy as np


# ============================================================
# CLEAN LOCATION STRINGS
# ============================================================

def clean_locations(df):

    df["location"] = (
        df["location"]
        .astype(str)
        .str.replace('"', '', regex=False)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.strip()
    )

    return df


# ============================================================
# MERGE WEATHER
# ============================================================

def merge_weather(pollution, weather):

    weather["datetime"] = pd.to_datetime(weather["datetime"], utc=True)
    pollution["datetime"] = pd.to_datetime(pollution["datetime"], utc=True)

    weather = weather.drop_duplicates("datetime")
    df = pollution.merge(weather, on="datetime", how="left")
    df = df.sort_values("datetime")

    return df


# ============================================================
# DROP UNUSED FEATURES
# ============================================================

def drop_unused(df):

    df = df.drop(columns=["co", "o3"], errors="ignore")

    return df


# ============================================================
# RESAMPLE HOURLY PER STATION
# ============================================================

def resample_hourly(df):

    pollutant_cols = ["no2", "pm10", "pm25", "so2"]
    weather_cols   = ["temperature", "humidity", "wind_speed", "precipitation"]
    coord_cols     = ["latitude", "longitude"]

    df_original = df.copy()

    available_cols = [c for c in pollutant_cols if c in df_original.columns]

    # --- pollutáns ---
    df_poll = (
        df_original[["location", "datetime"] + available_cols]
        .set_index(["location", "datetime"])
        .groupby(level="location")
        .resample("1h", level="datetime")
        .mean()
        .reset_index()
    )

    # --- weather ---
    df_weather = (
        df_original[["datetime"] + weather_cols]
        .drop_duplicates("datetime")
        .set_index("datetime")
        .resample("1h")
        .mean()
        .reset_index()
    )

    # --- merge ---
    df = df_poll.merge(df_weather, on="datetime", how="left")

    # --- coords (eredetiből!) ---
    coords = df_original[["location"] + coord_cols].drop_duplicates()

    df = df.merge(coords, on="location", how="left")


    return df


# ============================================================
# CLEAN INVALID POLLUTANTS
# ============================================================

def clean_pollutants(df):

    pollution_cols = ["no2", "pm10", "pm25", "so2"]

    for col in pollution_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    return df


# ============================================================
# INTERPOLATE PER STATION
# ============================================================

def interpolate_station(df):

    # --- WEATHER interpoláció (globális, nem location szerint) ---
    weather_cols = ["temperature", "humidity", "wind_speed", "precipitation"]

    df = df.sort_values("datetime")

    df[weather_cols] = (
        df[weather_cols]
        .interpolate(limit=3, limit_direction="forward")
    )

    # --- POLLUTÁNS interpoláció ---
    df = df.sort_values(["location", "datetime"])
    cols_to_interp = [c for c in ["pm25", "pm10", "no2", "so2"] if c in df.columns]

    df[cols_to_interp] = (
        df.groupby("location")[cols_to_interp]
        .apply(lambda x: x.interpolate(limit=3, limit_direction="forward"))
        .reset_index(level=0, drop=True)
    )

    return df


# ============================================================
# BUILD BASE DATASET
# ============================================================

def build_base_dataset(pollution, weather):

    EXPECTED_POLLUTANTS = ["pm25", "pm10", "no2", "so2"]

    for col in EXPECTED_POLLUTANTS:
        if col not in pollution.columns:
            pollution[col] = np.nan

    pollution = clean_locations(pollution)

    df = merge_weather(pollution, weather)
    df = drop_unused(df)
    df = clean_pollutants(df)
    df = resample_hourly(df)
    df = interpolate_station(df)

    df = df.sort_values(["location", "datetime"])

    return df