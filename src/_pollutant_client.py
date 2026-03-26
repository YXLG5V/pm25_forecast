# ==========================================================
# SIMPLE STATION POLLUTION FETCHER (WITH STATION REGISTRY)
# ==========================================================

import os
import time
from datetime import datetime, timedelta, UTC

import pandas as pd
from dotenv import load_dotenv
from openaq import OpenAQ
from openaq.shared.exceptions import RateLimitError
import httpx


# ==========================================================
# API INIT
# ==========================================================

load_dotenv("./config/.env")

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAQ_API_KEY not found in .env")

client = OpenAQ(api_key=API_KEY)


# ==========================================================
# STATION REGISTRY (FROM DISCOVERY)
# ==========================================================

STATIONS = {

    "Gyor Ifjusag": {
        "lat": 47.676111000033536,
        "lon": 17.649736000397873,
        "sensors": {
            "co":   [9263],
            "no2":  [9343],
            "o3":   [9311],
            "pm10": [9310],
            "pm25": [35960],
            "so2":  [9309],
        }
    },

    "Gyor Szent Istvan": {
        "lat": 47.68527800000957,
        "lon": 17.639444000391087,
        "sensors": {
            "co":   [9308],
            "no2":  [9307],
            "pm10": [9306],
            "pm25": [36004],
            "so2":  [9345],
        }
    },

    "Gyor 1 Szent Istvan": {
        "lat": 47.68537,
        "lon": 17.63955,
        "sensors": {
            "co":   [4272948],
            "no2":  [4273202],
            "o3":   [4272790],
            "pm10": [4272849],
            "pm25": [4273241],
            "so2":  [4273114],
        }
    },

    "OMSZ LRK Mobil": {
        "lat": 47.694032,
        "lon": 17.73908,
        "sensors": {
            "co":   [4272915],
            "no2":  [4273233],
            "o3":   [4273083],
            "pm10": [4272687],
            "pm25": [4273314],
            "so2":  [4273296],
        }
    },

    "Gyor 2 Ifjusag": {
        "lat": 47.67717,
        "lon": 17.65782,
        "sensors": {
            "co":   [6871130],
            "no2":  [7120205],
            "o3":   [6343097],
            "pm10": [7040998],
            "pm25": [6343098],
            "so2":  [7040717],
        }
    },

    "VS": {
        "lat": 47.76648848572361,
        "lon": 17.65766666921887,
        "sensors": {
            "pm1": [7971335, 7978268],
            "pm10": [7971326, 7978286],
            "pm25": [7490659],
            # nem klasszikus pollutánsok:
            # "relativehumidity": [7971477],
            # "temperature": [7490639],
        }
    }
}


# ==========================================================
# SAFE API CALL
# ==========================================================

def safe_call(**kwargs):

    retry = 0

    while True:
        try:
            return client.measurements.list(**kwargs)

        except RateLimitError:
            print("Rate limit hit → waiting 25s")
            time.sleep(25)

        except (httpx.ReadTimeout, TimeoutError):
            wait = min(60, 2 ** retry)
            print(f"Timeout → retry in {wait}s")
            time.sleep(wait)
            retry += 1


# ==========================================================
# MAIN FETCH FUNCTION (SIMPLIFIED API)
# ==========================================================

def fetch_station_pollutants(
    location_name: str,
    hours: int = 48
):
    """
    Single-station pollutant downloader
    using internal station registry
    """

    if location_name not in STATIONS:
        raise ValueError(f"Unknown location: {location_name}")

    station = STATIONS[location_name]

    lat = station["lat"]
    lon = station["lon"]
    sensors = station["sensors"]

    end = datetime.now(UTC)
    start = end - timedelta(hours=hours)

    print(f"Downloading pollutants: {start} → {end}")
    print(f"Location: {location_name}")

    records = []

    # ------------------------------------------------------
    # LOOP OVER POLLUTANTS
    # ------------------------------------------------------
    for param, sensor_ids in sensors.items():

        for sensor_id in sensor_ids:

            print(f"Fetching {param} | sensor {sensor_id}")

            resp = safe_call(
                sensors_id=sensor_id,
                datetime_from=start.isoformat(),
                datetime_to=end.isoformat(),
                limit=1000
            )

            for m in resp.results:
                records.append({
                    "datetime": m.period.datetime_from.utc,
                    "parameter": param,
                    "value": m.value
                })

    if not records:
        raise RuntimeError("No pollutant data downloaded")

    df = pd.DataFrame(records)

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # ------------------------------------------------------
    # WIDE FORMAT
    # ------------------------------------------------------
    df = (
        df
        .pivot_table(
            index="datetime",
            columns="parameter",
            values="value",
            aggfunc="mean"
        )
        .sort_index()
    )

    # ------------------------------------------------------
    # HOURLY RESAMPLE
    # ------------------------------------------------------
    df = df.resample("1h").mean()

    # ------------------------------------------------------
    # ADD META
    # ------------------------------------------------------
    df["location"] = location_name
    df["latitude"] = lat
    df["longitude"] = lon

    df = df.reset_index()

    return df


# ==========================================================
# CLEANUP
# ==========================================================

def close_client():
    client.close()