import numpy as np
import joblib as joblib
import pandas as pd

# basic: location,datetime,latitude,longitude,no2,pm10,pm25,so2,temperature,humidity,wind_speed,precipitation

# ------------------------------------------------
# PM MEMORY
# ------------------------------------------------
# pm25_lag1	előző óra PM2.5
# pm25_lag3	3 órával korábbi érték
# pm25_lag6	6 órával korábbi érték
# pm25_lag24	tegnap ugyanebben az órában
# pm25_roll6	utolsó 6 óra átlaga
# pm25_roll24	utolsó 24 óra átlaga
# pm25_trend_3h	3 órás változás iránya
# pm25_std_12h	12 órás szórás

def add_pm_features(df):
    df["pm25_lag1"] = df.groupby("location")["pm25"].shift(1)
    df["pm25_lag3"] = df.groupby("location")["pm25"].shift(3)
    df["pm25_lag6"] = df.groupby("location")["pm25"].shift(6)
    df["pm25_lag24"] = df.groupby("location")["pm25"].shift(24)

    df["pm25_roll6"] = (
        df.groupby("location")["pm25"]
        .transform(lambda x: x.rolling(6).mean().shift(1))
    )

    df["pm25_roll24"] = (
        df.groupby("location")["pm25"]
        .transform(lambda x: x.rolling(24).mean().shift(1))
    )

    df["pm25_trend_3h"] = df.groupby("location")["pm25"].diff(3)

    df["pm25_std_12h"] = (
        df.groupby("location")["pm25"]
        .transform(lambda x: x.rolling(12).std().shift(1))
    )

    return df


# ------------------------------------------------
# TIME FEATURES
# ------------------------------------------------
# hour	napszak (0–23)
# hour_sin	ciklikus napszak komponens
# hour_cos	ciklikus napszak komponens
# month	évszak jelleg
# month_sin	ciklikus éves komponens
# month_cos	ciklikus éves komponens
# weekend_flag	hétvége bináris jelző
# heating_season_flag fűtési időszak

def add_time_features(df):

    dt = pd.to_datetime(df["datetime"])

    df["hour"] = dt.dt.hour
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    df["month"] = dt.dt.month
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)

    df["dow"] = dt.dt.dayofweek
    df["weekend_flag"] = (df["dow"] >= 5).astype(int)

    df["heating_season_flag"] = dt.dt.month.isin([10,11,12,1,2,3,4]).astype(int)

    df = df.drop(columns=["dow"])

    return df


# ------------------------------------------------
# WEATHER
# ------------------------------------------------
# temp_change_3h	hőmérséklet változás 3 órán belül
# humidity_change_3h	páratartalom változás 3 órán belül
# wind_change_3h	szélsebesség változás 3 órán belül
# stagnation_index	stagnáló légköri állapot mértéke
# temp_wind_interaction	hőmérséklet és szél kombinációja
# mixing_index	légkeveredés indikátora
# ventilation_index	szél + keveredés kombináció
# stagnation_hours_6h	stagnáló órák száma

def add_weather_features(df):

# --- változások ---
    df["temp_change_3h"] = (
        df.groupby("location")["temperature"].diff(3)
    )

    df["humidity_change_3h"] = (
        df.groupby("location")["humidity"].diff(3)
    )

    df["wind_change_3h"] = (
        df.groupby("location")["wind_speed"].diff(3)
    )

    # --- fizikai indexek ---
    df["stagnation_index"] = (
        df["humidity"] / (df["wind_speed"] + 0.5)
    )

    df["temp_wind_interaction"] = (
        df["temperature"] * df["wind_speed"]
    )

    df["mixing_index"] = (
        (df["temperature"] + 273.15) *
        (df["wind_speed"] + 1.0) /
        (df["humidity"] + 20.0)
    )

    df["ventilation_index"] = (
        df["wind_speed"] *
        (df["temperature"] + 273.15)
    )

    # --- stagnáció ---
    df["low_wind_flag"] = (df["wind_speed"] < 2).astype(int)

    df["stagnation_hours_6h"] = (
        df.groupby("location")["low_wind_flag"]
        .transform(lambda x: x.rolling(6).sum().shift(1))
    )

    df.drop(columns=["low_wind_flag"], inplace=True)

    return df


# ------------------------------------------------
# SPATIAL
# ------------------------------------------------
# lat_norm
# lon_norm

def add_spatial_features(df, fit=False):

    import joblib
    import os

    # =========================================================
    # FIT MODE (TRAIN)
    # =========================================================
    if fit:
        LAT_MEAN = df["latitude"].mean()
        LON_MEAN = df["longitude"].mean()

        locations = sorted(df["location"].unique())
        mapping = {loc: i for i, loc in enumerate(locations)}

        # mentés
        joblib.dump(mapping, "./artifacts/location_mapping.pkl")
        joblib.dump(LAT_MEAN, "./artifacts/lat_mean.pkl")
        joblib.dump(LON_MEAN, "./artifacts/lon_mean.pkl")

    # =========================================================
    # INFERENCE MODE (FORECAST / EVAL)
    # =========================================================
    else:
        LAT_MEAN = joblib.load("./artifacts/lat_mean.pkl")
        LON_MEAN = joblib.load("./artifacts/lon_mean.pkl")
        mapping = joblib.load("./artifacts/location_mapping.pkl")

    # =========================================================
    # APPLY
    # =========================================================
    df["lat_norm"] = df["latitude"] - LAT_MEAN
    df["lon_norm"] = df["longitude"] - LON_MEAN
    df["location_id"] = df["location"].map(mapping).fillna(-1)

    return df

def add_features(df):
    df = add_pm_features(df)
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_spatial_features(df)
    return df

# ============================================================
# FEATURE ENGINEERING
# ============================================================


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

def build_features(df, fit=False):

    df = df.sort_values(["location", "datetime"])
    
    df = add_pm_features(df)
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_spatial_features(df, fit=fit)

    return df