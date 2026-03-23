# ==========================================================
# OpenAQ API CLIENT
# ==========================================================

import os
import time
from datetime import datetime, timedelta, UTC

import pandas as pd
from dotenv import load_dotenv

from openaq import OpenAQ
from openaq.shared.exceptions import RateLimitError
import httpx


# ======================================================
# CONFIG
# ======================================================

load_dotenv("./config/.env")

API_KEY = os.getenv("OPENAQ_API_KEY")

client = OpenAQ(api_key=API_KEY)

LAT = 47.6875
LON = 17.6504
RADIUS_METERS = 25000


# ======================================================
# SAFE API CALL
# ======================================================

def safe_measurement_call(**kwargs):

    retry = 0

    while True:

        try:
            return client.measurements.list(**kwargs)

        except RateLimitError:
            print("Rate limit hit → waiting...")
            time.sleep(25)

        except (httpx.ReadTimeout, TimeoutError):

            wait = min(60, 2 ** retry)

            print(f"Timeout → retry in {wait}s")

            time.sleep(wait)
            retry += 1


# ======================================================
# LOCATION DISCOVERY
# ======================================================

def get_locations():
    """Find monitoring stations near Győr"""

    resp = client.locations.list(
        coordinates=(LAT, LON),
        radius=RADIUS_METERS,
        limit=1000
    )

    if not resp.results:
        raise RuntimeError("No locations found")

    return resp.results


def get_location_by_name(location_name):

    resp = client.locations.list(
        coordinates=(LAT, LON),
        radius=RADIUS_METERS,
        limit=1000
    )

    matches = [
        loc for loc in resp.results
        if location_name.lower() in loc.name.lower()
    ]

    if not matches:
        raise RuntimeError(f"No match for {location_name}")

    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous: {location_name}")

    return matches[0]


# ======================================================
# MAIN DATA DOWNLOAD
# ======================================================

def fetch_recent_pollutants(location_name, pm25_sensor_id, hours=24):

    end = datetime.now(UTC)
    start = end - timedelta(hours=hours)

    print(f"Downloading pollutant data: {start} → {end}")

    regional_parameters = ["pm10", "no2", "co", "o3", "so2"]

    locations = [get_location_by_name(location_name)]

    sensor_map = {}

    # --------------------------------------------------
    # SENSOR DISCOVERY (regional pollutants)
    # --------------------------------------------------

    for loc in locations:
        for s in loc.sensors:

            name = s.parameter.name

            if name in regional_parameters:
                sensor_map.setdefault(name, []).append(s.id)

    records = []

    # --------------------------------------------------
    # DOWNLOAD REGIONAL POLLUTANTS
    # --------------------------------------------------

    for param, sensor_ids in sensor_map.items():

        for sensor_id in sensor_ids:

            resp = safe_measurement_call(
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

    # --------------------------------------------------
    # DOWNLOAD STATION PM2.5
    # --------------------------------------------------

    resp = safe_measurement_call(
        sensors_id=pm25_sensor_id,
        datetime_from=start.isoformat(),
        datetime_to=end.isoformat(),
        limit=1000
    )

    for m in resp.results:

        records.append({
            "datetime": m.period.datetime_from.utc,
            "parameter": "pm25",
            "value": m.value
        })

    if not records:
        raise RuntimeError("No pollutant data downloaded")

    df = pd.DataFrame(records)

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # --------------------------------------------------
    # WIDE FORMAT
    # --------------------------------------------------

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

    # --------------------------------------------------
    # HOURLY ALIGNMENT
    # --------------------------------------------------

    df = df.resample("1h").mean()

    return df


# ======================================================
# CLOSE CLIENT
# ======================================================

def close_client():
    client.close()