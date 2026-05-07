# ======================================================
# WEATHER_CLIENT.PY
# 
# ======================================================

import requests
import pandas as pd
from datetime import datetime, timedelta, UTC

WEATHER_FORECAST_CACHE = None
WEATHER_FORECAST_CACHE_TIME = None
MOCK_FORECAST_DATA = {"latitude":47.68,"longitude":17.66,"generationtime_ms":0.1386404037475586,"utc_offset_seconds":0,"timezone":"GMT","timezone_abbreviation":"GMT","elevation":117.0,"hourly_units":{"time":"iso8601","temperature_2m":"°C","relative_humidity_2m":"%","wind_speed_10m":"km/h","precipitation":"mm","weather_code":"wmo code"},"hourly":{"time":["2026-05-07T00:00","2026-05-07T01:00","2026-05-07T02:00","2026-05-07T03:00","2026-05-07T04:00","2026-05-07T05:00","2026-05-07T06:00","2026-05-07T07:00","2026-05-07T08:00","2026-05-07T09:00","2026-05-07T10:00","2026-05-07T11:00","2026-05-07T12:00","2026-05-07T13:00","2026-05-07T14:00","2026-05-07T15:00","2026-05-07T16:00","2026-05-07T17:00","2026-05-07T18:00","2026-05-07T19:00","2026-05-07T20:00","2026-05-07T21:00","2026-05-07T22:00","2026-05-07T23:00","2026-05-08T00:00","2026-05-08T01:00","2026-05-08T02:00","2026-05-08T03:00","2026-05-08T04:00","2026-05-08T05:00","2026-05-08T06:00","2026-05-08T07:00","2026-05-08T08:00","2026-05-08T09:00","2026-05-08T10:00","2026-05-08T11:00","2026-05-08T12:00","2026-05-08T13:00","2026-05-08T14:00","2026-05-08T15:00","2026-05-08T16:00","2026-05-08T17:00","2026-05-08T18:00","2026-05-08T19:00","2026-05-08T20:00","2026-05-08T21:00","2026-05-08T22:00","2026-05-08T23:00"],"temperature_2m":[17.4,16.9,16.2,15.4,14.9,15.2,15.9,17.1,17.8,19.0,19.5,20.0,19.1,18.4,18.5,19.3,19.3,18.0,17.2,15.9,14.7,14.0,13.6,13.6,13.3,12.6,12.1,11.4,11.0,11.8,13.3,15.2,17.1,18.8,20.2,21.1,21.9,21.9,22.6,22.1,21.9,21.5,20.5,19.0,17.2,16.0,14.9,14.2],"relative_humidity_2m":[65,66,68,74,76,75,72,58,53,43,40,41,49,60,70,58,58,66,69,73,80,84,83,83,84,89,90,92,91,84,77,69,61,53,42,38,33,35,31,35,35,35,39,45,50,50,55,61],"wind_speed_10m":[18.8,18.2,13.2,9.7,8.7,6.8,6.8,7.6,8.9,8.7,8.6,6.5,3.8,4.3,6.4,9.0,10.4,10.2,6.1,3.9,6.4,5.4,6.8,1.8,7.2,6.0,7.2,7.4,6.4,5.8,8.0,6.4,7.4,9.2,9.7,9.4,9.2,15.6,8.7,12.3,10.4,10.1,7.1,5.1,4.9,6.0,5.3,2.8],"precipitation":[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.10,0.80,0.10,0.10,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],"weather_code":[3,3,3,3,3,3,3,3,3,3,3,3,61,61,61,61,3,2,1,1,1,3,3,3,3,2,2,1,1,1,1,0,0,1,0,0,1,1,3,3,2,1,3,3,3,3,3,3]}}

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

    global WEATHER_FORECAST_CACHE
    global WEATHER_FORECAST_CACHE_TIME

    now = datetime.now(UTC)

    # 15 perc cache
    if (
        WEATHER_FORECAST_CACHE is not None and
        WEATHER_FORECAST_CACHE_TIME is not None and
        (now - WEATHER_FORECAST_CACHE_TIME).total_seconds() < 900
    ):

        print("Using cached weather forecast")

        return WEATHER_FORECAST_CACHE.copy()

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

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

    except Exception as e:

        print("Weather API failed:")
        print(e)

        print("Using mock forecast data")

        data = MOCK_FORECAST_DATA

    df = _parse_weather_json(data)

    WEATHER_FORECAST_CACHE = df
    WEATHER_FORECAST_CACHE_TIME = now

    return df.copy()


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