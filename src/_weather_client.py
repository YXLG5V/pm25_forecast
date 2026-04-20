# ======================================================
# WEATHER_CLIENT.PY
# 
# ======================================================

import requests
import pandas as pd
from datetime import datetime, timedelta, UTC

# ======================================================
# COMMON PARSER
# ======================================================

def _parse_weather_json(data):

    df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
        "humidity": data["hourly"]["relative_humidity_2m"],
        "wind_speed": data["hourly"]["wind_speed_10m"],
        "precipitation": data["hourly"]["precipitation"],
        "weather_code": data["hourly"]["weather_code"]
    })

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    return df.set_index("datetime")


# ======================================================
# WEATHER FORECAST (FUTURE)
# ======================================================

def fetch_weather_forecast(lat, lon):

    print("Downloading weather forecast...")

    url = "https://api.open-meteo.com/v1/forecast"

    params = dict(
        latitude=lat,
        longitude=lon,
        hourly=[
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
            "weather_code"
        ],
        forecast_days=2,
        timezone="UTC"
    )

    data = requests.get(url, params=params).json()

    return _parse_weather_json(data)


# ======================================================
# WEATHER HISTORY
# ======================================================

def fetch_weather_history(lat, lon, hours=24):

    print("Downloading historical weather...")

    end = datetime.now(UTC)
    start = end - timedelta(hours=hours)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = dict(
        latitude=lat,
        longitude=lon,
        start_date=start.date().isoformat(),
        end_date=end.date().isoformat(),
        hourly=[
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
            "weather_code"
        ],
        timezone="UTC"
    )

    data = requests.get(url, params=params, timeout=10).json()

    df = _parse_weather_json(data)

    return df.loc[start:end]