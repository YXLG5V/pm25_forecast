import os
import time
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from openaq import OpenAQ
from openaq.shared.exceptions import RateLimitError
import httpx


# ======================================================
# CONFIG
# ======================================================

load_dotenv("./config/.env")
print("API_KEY loaded:", bool(os.getenv("OPENAQ_API_KEY")))


API_KEY = os.getenv("OPENAQ_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAQ_API_KEY missing from .env")

client = OpenAQ(api_key=API_KEY)

PARAMETERS = {"pm25", "pm10", "no2", "o3", "so2", "co"}

LAT = 47.6875
LON = 17.6504
RADIUS_METERS = 25000


# ======================================================
# SAFE API CALL
# ======================================================

def safe_measurement_call(**kwargs):

    retry = 0
    max_retry = 8

    while True:
        try:
            return client.measurements.list(**kwargs)

        except RateLimitError as e:
            wait_time = 25
            msg = str(e)

            if "resets in" in msg:
                try:
                    wait_time = int(msg.split("resets in")[1].split()[0]) + 2
                except Exception:
                    pass

            print(f"[RateLimit] waiting {wait_time}s...")
            time.sleep(wait_time)

        except (httpx.ReadTimeout, TimeoutError, httpx.HTTPError):

            if retry >= max_retry:
                raise RuntimeError("Too many retries")

            wait = min(60, 2 ** retry)
            print(f"[Retry] {retry+1}/{max_retry} in {wait}s...")
            time.sleep(wait)
            retry += 1


# ======================================================
# LOCATIONS
# ======================================================

def get_locations():

    resp = client.locations.list(
        coordinates=(LAT, LON),
        radius=RADIUS_METERS,
        limit=1000
    )

    if not resp.results:
        raise RuntimeError("No locations found.")

    print(f"Locations found: {len(resp.results)}")
    return resp.results


# ======================================================
# SENSOR INDEX
# ======================================================

def build_sensor_index(locations):

    sensors = []

    for loc in locations:
        coords = loc.coordinates

        for s in loc.sensors:
            if s.parameter.name in PARAMETERS:
                sensors.append({
                    "sensor_id": s.id,
                    "location": loc.name,
                    "latitude": coords.latitude if coords else None,
                    "longitude": coords.longitude if coords else None,
                })

    print(f"Sensors selected: {len(sensors)}")
    return sensors


# ======================================================
# DOWNLOAD ONE SENSOR
# ======================================================

def fetch_sensor(sensor_info, start, end):

    records = []
    chunk_days = 30
    current = start

    while current < end:

        chunk_end = min(current + timedelta(days=chunk_days), end)

        resp = safe_measurement_call(
            sensors_id=sensor_info["sensor_id"],
            datetime_from=current.isoformat(),
            datetime_to=chunk_end.isoformat(),
            limit=1000
        )

        for m in resp.results:
            records.append({
                "datetime": m.period.datetime_from.utc,
                "parameter": m.parameter.name,
                "value": m.value,
                "location": sensor_info["location"],
                "latitude": sensor_info["latitude"],
                "longitude": sensor_info["longitude"],
            })

        current = chunk_end
        time.sleep(0.15)

    return records


# ======================================================
# MAIN PIPELINE
# ======================================================

def fetch_all(start, end):

    print("TIME WINDOW:", start, "→", end)

    locations = get_locations()
    sensors = build_sensor_index(locations)

    all_records = []

    for sensor in tqdm(sensors, desc="Downloading sensors"):
        all_records.extend(fetch_sensor(sensor, start, end))

    df = pd.DataFrame(all_records)

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    df_wide = df.pivot_table(
        index=["datetime", "location", "latitude", "longitude"],
        columns="parameter",
        values="value",
        aggfunc="mean"
    ).reset_index()

    return df_wide.sort_values("datetime")


# ======================================================
# MAIN
# ======================================================

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    args = parser.parse_args()

    start = pd.to_datetime(args.start_date, utc=True)
    end   = pd.to_datetime(args.end_date, utc=True)

    df = fetch_all(start, end)

    output = "./data/raw/pollutants.csv"
    df.to_csv(output, index=False)

    client.close()

    print(f"Saved → {output}")