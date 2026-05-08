# ======================================================
# WEATHER_CLIENT.PY
# ======================================================

import requests
import pandas as pd

from datetime import (
    datetime,
    timedelta,
    UTC
)

import os


# ======================================================
# CACHE
# ======================================================

WEATHER_FORECAST_CACHE = None
WEATHER_FORECAST_CACHE_TIME = None


# ======================================================
# ENV
# ======================================================

OPENWEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY"
)


# ======================================================
# DEBUG LOG
# ======================================================

def log(msg):

    print(
        f"[WEATHER] "
        f"{datetime.now(UTC).isoformat()} "
        f"{msg}",
        flush=True
    )


# ======================================================
# OPENWEATHER -> WMO MAP
# ======================================================

OWM_TO_WMO = {

    # clear
    800: 0,

    # partly cloudy
    801: 1,
    802: 2,

    # cloudy
    803: 3,
    804: 3,

    # drizzle
    300: 51,
    301: 53,
    302: 55,

    # rain
    500: 61,
    501: 61,
    502: 63,
    503: 65,
    504: 65,

    # thunderstorm
    200: 95,
    201: 95,
    202: 96,
}


# ======================================================
# MOCK
# ======================================================

MOCK_FORECAST_DATA = {
    "hourly": {
        "time": [],
        "temperature_2m": [],
        "relative_humidity_2m": [],
        "wind_speed_10m": [],
        "precipitation": [],
        "weather_code": []
    }
}


# ======================================================
# COMMON PARSER
# ======================================================

def _parse_weather_json(data):

    if "hourly" not in data:

        log(
            "ERROR: Missing hourly field"
        )

        raise RuntimeError(
            "Weather API response missing hourly"
        )

    hourly = data["hourly"]

    df = pd.DataFrame({

        "datetime":
            hourly["time"],

        "temperature":
            hourly["temperature_2m"],

        "humidity":
            hourly["relative_humidity_2m"],

        "wind_speed":
            hourly["wind_speed_10m"],

        "precipitation":
            hourly["precipitation"],

        "weather_code":
            hourly["weather_code"]
    })

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True
    )

    df = df.set_index("datetime")

    log(
        f"Parsed dataframe shape: "
        f"{df.shape}"
    )

    return df


# ======================================================
# OPENWEATHER FALLBACK
# ======================================================

def _fetch_openweather_fallback(
    lat,
    lon
):

    log(
        "Trying OpenWeather fallback"
    )

    if not OPENWEATHER_API_KEY:

        raise RuntimeError(
            "OPENWEATHER_API_KEY missing"
        )

    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
    )

    params = dict(
        lat=lat,
        lon=lon,
        APPID=OPENWEATHER_API_KEY,
        units="metric"
    )

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    log(
        f"OpenWeather status: "
        f"{response.status_code}"
    )

    response.raise_for_status()

    raw = response.json()

    if "list" not in raw:

        raise RuntimeError(
            "OpenWeather missing list field"
        )

    rows = []

    for row in raw["list"]:

        weather_id = (
            row["weather"][0]["id"]
        )

        rows.append({

            "datetime":
                datetime.fromtimestamp(
                    row["dt"],
                    UTC
                ),

            "temperature":
                row["main"]["temp"],

            "humidity":
                row["main"]["humidity"],

            # m/s -> km/h
            "wind_speed":
                row["wind"]["speed"] * 3.6,

            # 3h rain
            "precipitation":
                row.get(
                    "rain",
                    {}
                ).get("3h", 0.0),

            "weather_code":
                OWM_TO_WMO.get(
                    weather_id,
                    3
                )
        })

    df = pd.DataFrame(rows)

    log(
        f"Fallback rows: "
        f"{len(df)}"
    )

    df = df.set_index("datetime")

    # ==================================================
    # 3H -> 1H
    # ==================================================

    df = (
        df
        .resample("1h")
        .interpolate(method="linear")
    )

    log(
        f"Fallback resampled rows: "
        f"{len(df)}"
    )

    df["weather_code"] = (
        df["weather_code"]
        .round()
        .astype(int)
    )

    # ==================================================
    # OPEN-METEO FORMAT
    # ==================================================

    data = {
        "hourly": {

            "time": [
                dt.isoformat()
                for dt in df.index
            ],

            "temperature_2m":
                df["temperature"].tolist(),

            "relative_humidity_2m":
                df["humidity"].tolist(),

            "wind_speed_10m":
                df["wind_speed"].tolist(),

            "precipitation":
                df["precipitation"].tolist(),

            "weather_code":
                df["weather_code"].tolist()
        }
    }

    return data


# ======================================================
# WEATHER FORECAST
# ======================================================

def fetch_weather_forecast(
    lat,
    lon
):

    global WEATHER_FORECAST_CACHE
    global WEATHER_FORECAST_CACHE_TIME

    now = datetime.now(UTC)

    log(
        f"Forecast request "
        f"lat={lat} lon={lon}"
    )

    # ==================================================
    # CACHE
    # ==================================================

    if (
        WEATHER_FORECAST_CACHE is not None and
        WEATHER_FORECAST_CACHE_TIME is not None and
        (
            now - WEATHER_FORECAST_CACHE_TIME
        ).total_seconds() < 900
    ):

        age = (
            now - WEATHER_FORECAST_CACHE_TIME
        ).total_seconds()

        log(
            f"Using cache "
            f"(age={age:.1f}s)"
        )

        return WEATHER_FORECAST_CACHE.copy()

    log(
        "Cache miss -> downloading"
    )

    # ==================================================
    # OPEN-METEO
    # ==================================================

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

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

        log(
            "Calling Open-Meteo"
        )

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        log(
            f"Open-Meteo status: "
            f"{response.status_code}"
        )

        response.raise_for_status()

        data = response.json()

        source = "openmeteo"

        log(
            "Using Open-Meteo"
        )

    except Exception as e:

        log(
            f"Open-Meteo failed: {e}"
        )

        try:

            data = _fetch_openweather_fallback(
                lat,
                lon
            )

            source = "openweather"

            log(
                "Using OpenWeather fallback"
            )

        except Exception as fallback_error:

            log(
                f"Fallback failed: "
                f"{fallback_error}"
            )

            log(
                "Using MOCK data"
            )

            data = MOCK_FORECAST_DATA

            source = "mock"

    # ==================================================
    # NORMALIZE
    # ==================================================

    df = _parse_weather_json(data)

    df["weather_source"] = source

    log(
        f"Weather source used: "
        f"{source}"
    )

    log(
        f"Final dataframe shape: "
        f"{df.shape}"
    )

    WEATHER_FORECAST_CACHE = df
    WEATHER_FORECAST_CACHE_TIME = now

    log(
        "Forecast cached"
    )

    return df.copy()


# ======================================================
# WEATHER HISTORY
# ======================================================

def fetch_weather_history(
    lat,
    lon,
    hours=24
):

    log(
        f"History request "
        f"hours={hours}"
    )

    end = datetime.now(UTC)

    start = (
        end - timedelta(hours=hours)
    )

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
    )

    params = dict(

        latitude=lat,

        longitude=lon,

        start_date=
            start.date().isoformat(),

        end_date=
            end.date().isoformat(),

        hourly=[
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
            "weather_code"
        ],

        timezone="UTC"
    )

    log(
        "Calling archive API"
    )

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    log(
        f"Archive status: "
        f"{response.status_code}"
    )

    response.raise_for_status()

    data = response.json()

    df = _parse_weather_json(data)

    df = df.loc[start:end]

    log(
        f"History dataframe shape: "
        f"{df.shape}"
    )

    return df