# ==========================================================
# SENSOR DISCOVERY TOOL (BY COORDINATES)
# ==========================================================

import os
from dotenv import load_dotenv
from openaq import OpenAQ


# ==========================================================
# CONFIG
# ==========================================================

load_dotenv("./config/.env")

API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAQ_API_KEY not found")

client = OpenAQ(api_key=API_KEY)

LAT = 47.6875
LON = 17.6504
RADIUS_METERS = 25000


# ==========================================================
# DISCOVERY FUNCTION
# ==========================================================

def discover_sensors(lat, lon, radius=25000):

    print(f"\nSearching locations near ({lat}, {lon})...\n")

    resp = client.locations.list(
        coordinates=(lat, lon),
        radius=radius,
        limit=1000
    )

    if not resp.results:
        print("No locations found")
        return {}

    sensor_map = {}

    # ------------------------------------------------------
    # LOOP LOCATIONS
    # ------------------------------------------------------
    for loc in resp.results:

        print("=" * 60)
        print(f"Location: {loc.name}")
        print(f"Coords: {loc.coordinates.latitude}, {loc.coordinates.longitude}")

        if not loc.sensors:
            print("  No sensors")
            continue

        # --------------------------------------------------
        # LOOP SENSORS
        # --------------------------------------------------
        for s in loc.sensors:

            param = s.parameter.name
            sensor_id = s.id

            print(f"  - {param} | sensor_id={sensor_id}")

            sensor_map.setdefault(param, []).append({
                "sensor_id": sensor_id,
                "location": loc.name
            })

    return sensor_map


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":

    sensor_map = discover_sensors(LAT, LON)

    print("\n" + "#" * 60)
    print("RAW SENSOR MAP:")
    print(sensor_map)

    client.close()