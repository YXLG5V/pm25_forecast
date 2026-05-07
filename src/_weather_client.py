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

    if "hourly" not in data:

        print("Invalid weather API response:")
        print(data)

        raise RuntimeError(
            "Weather API response missing 'hourly'"
        )

    hourly = data["hourly"]

    df = pd.DataFrame({
        "datetime": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "wind_speed": hourly["wind_speed_10m"],
        "precipitation": hourly["precipitation"],
        "weather_code": hourly["weather_code"]
    })

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

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

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

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

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    df = _parse_weather_json(data)

    return df.loc[start:end]